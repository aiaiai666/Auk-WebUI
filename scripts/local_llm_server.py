#!/usr/bin/env python3
"""
Serve the local ``ckpts/Qwen2.5-Omni-3B`` as an OpenAI-compatible Chat Completions
endpoint, so the AuK Prompt Enhancer works without a cloud LLM.

Usage::

    python scripts/local_llm_server.py                       # 127.0.0.1:8000
    python scripts/local_llm_server.py --port 8123 --served-model-name qwen-omni-3b

Then point ``.env`` at it::

    LLM_API_KEY=local
    LLM_BASE_URL=http://127.0.0.1:8000/v1
    LLM_MODEL_NAME=qwen-omni-3b

Design notes
------------
* Only the Thinker **text** path is served. The Prompt Enhancer sends plain text
  messages (``auk.infer.pe`` builds ``list[dict[str, str]]``), so the vision tower
  and the Talker are dropped at load time -- same pruning ``AukInfer`` does.
* ``generation_config.eos_token_id`` is ``None`` in this checkpoint, so
  ``generate()`` never stops on its own and runs to ``max_tokens``. The chat
  template terminates turns with ``<|im_end|>`` (151645), so that id is passed
  explicitly; without it every reply would be followed by a runaway
  ``Human:/Assistant:`` continuation that breaks ``pe._extract_json``.
* ``max_tokens`` is capped by ``--max-new-tokens`` (default 1024, ~45s on a 4090)
  to stay inside ``pe``'s 120s per-call timeout.
* Any ``api_key`` is accepted: the endpoint is unauthenticated and bound to
  loopback by default.
* One generation at a time, guarded by an ``asyncio`` lock, so concurrent
  WebUI requests queue instead of thrashing VRAM.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, Qwen2_5OmniThinkerForConditionalGeneration


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("auk.local_llm")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "ckpts" / "Qwen2.5-Omni-3B"
IM_END_ID = 151645
ENDOFTEXT_ID = 151643
PRUNED_SUBMODULES = ("visual", "talker")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = 0.0
    stream: bool = False


class LocalLLM:
    """A single Qwen2.5-Omni Thinker exposed through the Chat Completions schema."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        served_model_name: str = "qwen-omni-3b",
        max_new_tokens: int = 1024,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"model path not found: {self.model_path}")
        self.device = device
        self.served_model_name = served_model_name
        self.max_new_tokens = max_new_tokens
        self._lock = asyncio.Lock()

        logger.info(f"Loading Qwen2.5-Omni Thinker from {self.model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        )
        for name in PRUNED_SUBMODULES:
            if getattr(model, name, None) is not None:
                delattr(model, name)
                setattr(model, name, None)
        self.model = model.to(device).eval()
        self.eos_token_ids = [IM_END_ID, ENDOFTEXT_ID]
        if device.startswith("cuda"):
            logger.info(f"Resident VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        logger.info("Ready. POST /v1/chat/completions, GET /v1/models, GET /health")

    def _render_prompt(self, messages: list[ChatMessage]) -> str:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        return self.tokenizer.apply_chat_template(payload, tokenize=False, add_generation_prompt=True)

    def _generate_sync(self, prompt: str, max_new_tokens: int, temperature: float) -> tuple[str, int, bool]:
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.eos_token_ids,
            "pad_token_id": ENDOFTEXT_ID,
        }
        if temperature and temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            gen_kwargs["do_sample"] = False
        try:
            with torch.no_grad():
                output = self.model.generate(**inputs, **gen_kwargs)
            new_ids = output[0][prompt_tokens:]
            hit_eos = bool(new_ids.numel()) and int(new_ids[-1]) in self.eos_token_ids
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        finally:
            # The caching allocator only grows during long generations. Without this the
            # server's *reserved* VRAM climbs towards the whole card (measured 19.3 GiB for a
            # 7.6 GiB model) and starves the AuK WebUI, which needs ~10.9 GiB of its own.
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        for stop in ("<|im_end|>", "<|endoftext|>"):
            text = text.split(stop)[0]
        return text.strip(), int(new_ids.numel()), hit_eos

    async def chat_completion(self, req: ChatCompletionRequest) -> dict:
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        if req.stream:
            raise HTTPException(status_code=400, detail="streaming is not supported")
        if req.model and req.model != self.served_model_name:
            raise HTTPException(status_code=404, detail=f"unknown model: {req.model}")

        budget = self.max_new_tokens if not req.max_tokens else min(int(req.max_tokens), self.max_new_tokens)
        prompt = self._render_prompt(req.messages)
        started = time.perf_counter()
        async with self._lock:
            text, completion_tokens, hit_eos = await asyncio.to_thread(
                self._generate_sync, prompt, budget, float(req.temperature or 0.0)
            )
        elapsed = time.perf_counter() - started
        reserved = torch.cuda.memory_reserved() / 1024**3 if self.device.startswith("cuda") else 0.0
        logger.info(f"generated {completion_tokens} tokens in {elapsed:.1f}s (eos={hit_eos}, reserved={reserved:.2f} GiB)")

        if not text:
            raise HTTPException(status_code=500, detail="model returned an empty completion")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop" if hit_eos else "length",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_tokens,
            },
        }

    def models_payload(self) -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }


def build_app(llm: LocalLLM) -> FastAPI:
    app = FastAPI(title="AuK local LLM", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        return {"status": "ok", "model": llm.served_model_name, "vram_gib": round(vram, 2)}

    @app.get("/v1/models")
    def models() -> dict:
        return llm.models_payload()

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> dict:
        return await llm.chat_completion(req)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve ckpts/Qwen2.5-Omni-3B as an OpenAI-compatible endpoint.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="local Qwen2.5-Omni snapshot")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (loopback by default; the endpoint is unauthenticated)")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument("--device", default=None, help="cuda | cpu (defaults to cuda when available)")
    parser.add_argument("--served-model-name", default="qwen-omni-3b", help="model id reported to clients")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="hard cap on tokens per reply")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    llm = LocalLLM(
        args.model_path,
        device=device,
        served_model_name=args.served_model_name,
        max_new_tokens=args.max_new_tokens,
    )
    uvicorn.run(build_app(llm), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
