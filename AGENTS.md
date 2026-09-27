# Repository Guidelines

AuK is a 1.5B Python (3.10) foundation model for speech generation and editing. Condensed setup: `pip install -e ".[gradio]"`, then read [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) before opening an issue or PR.

## Project Structure & Module Organization

The package uses a `src/` layout:

- `src/auk/infer/` — entry points: `infer_cli.py` (`auk-infer`), `infer_gradio.py` (`auk-gradio`), engine `infer_auk.py` (`AukInfer`), Prompt Enhancer `pe.py` + `pe.config.yaml`.
- `src/auk/model/` — diffusion backbone (`cfm_edit.py`, `flux2_edit.py`, `modules.py`) and the bundled VAE under `model/vae/`.
- `src/auk/train/train.py` — fine-tuning loop; `scripts/train.sh` launches it.
- `comfyui/` — ComfyUI nodes and `auk.json` workflow; `docs/` — guides; `assets/` — demo audio; `ckpts/` — weights (gitignored, downloaded at runtime).
- `webui/app.py` — **secondary-development WebUI** (`python -m webui.app`); it is a standalone panel built on `src/auk/infer/` and is not imported by `auk-gradio`. Its Examples tab parses every example command out of `README.md` at page load, so docs and UI stay in sync.
- `scripts/download_models.py` — downloads weights into `ckpts/` (aria2c multi-threaded) and verifies each file's SHA-256 against its Git-LFS pointer.
- `scripts/local_llm_server.py` — serves `ckpts/Qwen2.5-Omni-3B` as an OpenAI-compatible `/v1/chat/completions` so the Prompt Enhancer needs no cloud LLM; `start_app.sh` auto-starts it when `.env`'s `LLM_BASE_URL` points at loopback.
- `start_app.sh` — one-shot launcher: loads `.env`, frees port 7860, clears other GPU processes (exempting the local LLM server by port), auto-starts that server, then starts `webui/app.py` on `0.0.0.0:7860`.
- `outputs/` — generated wav files, named `outputs_YYYYMMDD_HHMMSS[_NNN].wav`.
- `ckpts/AuK` (Base), `ckpts/AuK-Flash` (distilled), `ckpts/Qwen2.5-Omni-3B` (encoder + local LLM), `ckpts/SenseVoiceSmall` (PE fallback ASR) — weights; `ckpts/AuK-Flash/vae.safetensors` is a **hardlink** of the Base VAE (identical SHA-256). `scripts/download_models.py --variant {auk,flash,qwen,asr}` fetches a subset and merges entries into `ckpts/.checksums.sha256`; `--reserve <GiB>` tunes the disk head-check. `SenseVoiceSmall` is resolved from `ckpts/` via `Path(__file__).parents[3]`, not the CWD, so the WebUI, CLI, and `auk-gradio` all find it; set `AUK_ASR_MODEL` to override.

## Build, Test, and Development Commands

```bash
python3 -m pip install -e ".[gradio]"   # extras: [gradio] [comfyui] [train]
pre-commit install                       # ruff + YAML hooks
ruff check . && ruff format --check .   # lint and format gate
python3 -m compileall -q src             # syntax check
auk-infer --audio ref.wav --instruction "..." --output out.wav
bash scripts/train.sh                    # fine-tuning (needs the [train] extra)
```

Downloading weights and running the secondary-development WebUI:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py            # aria2c download into ckpts/ + SHA-256 verify
python scripts/download_models.py --verify-only

./start_app.sh 7860                          # loads .env, frees 7860, clears GPU, starts webui/app.py
python -m webui.app --host 0.0.0.0 --port 7860   # manual start (no .env, no port/GPU cleanup)

python scripts/local_llm_server.py --port 8000   # Qwen2.5-Omni-3B as an OpenAI-compatible endpoint
```

The WebUI **lazily** loads a variant on first generation and never at startup; keep `ckpts/` next to `config.yaml`, since `config.yaml` resolves `ckpts/Qwen2.5-Omni-3B` relative to the CWD. Full details: [`docs/WEBUI.md`](docs/WEBUI.md).

There is no project-wide test suite. Run the smallest checks that cover your change and paste the exact commands and results into the PR. Changes to inference, model loading, sampling, audio I/O, or checkpoints require at least one end-to-end run per affected variant — record checkpoint revision, GPU, input, output, and memory. Gradio edits must also pass:

```bash
python3 -c "
import auk.infer.infer_gradio as g
g.CKPT_PATHS['AuK (Base)'] = 'ckpts/AuK/auk_base.safetensors'
g.CONFIG_PATHS['AuK (Base)'] = 'ckpts/AuK/config.yaml'
assert g.build_demo()
"
```

`CKPT_PATHS` is empty at import time (only `main()`'s argparse fills it), so a bare `assert build_demo()` raises `RuntimeError: No valid model checkpoint was configured.` in every checkout — seed the dict as above.

## Coding Style & Naming Conventions

Ruff (line length 130, target py310) enforces style. Use 4-space indent, `from __future__ import annotations`, and type hints on public signatures. Naming: `PascalCase` classes, `snake_case` functions, leading underscore for private helpers (`_trim_audio`), `UPPER_SNAKE_CASE` constants. Do not broaden the `model/vae/` lint exclusions without justifying it in the PR.

## Commit & Pull Request Guidelines

Commits use Conventional Commits — `feat:`, `fix(pe):`, `docs:` — with a subject plus a body giving motivation and validation. Branches are typically `feat/<slug>` or `fix/<slug>`.

PRs follow `.github/PULL_REQUEST_TEMPLATE.md`: one coherent problem per PR, link the issue with `Closes #123`, list all validation commands with results, and call out quality, latency, memory, and API effects. Update docs when behavior or configuration changes. Never commit weights, datasets, caches, credentials, or generated artifacts.

## Security & Configuration

Copy `.env.example` to `.env` (gitignored) for LLM and ASR credentials and export them from environment variables — never commit keys. Keep secrets, private URLs, and personal audio out of issues and commits; report security vulnerabilities privately to the maintainers.
