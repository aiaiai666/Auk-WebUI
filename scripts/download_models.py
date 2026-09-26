#!/usr/bin/env python3
"""
Download the AuK model checkpoints into ``ckpts/`` and verify their integrity.

Usage::

    export HF_ENDPOINT=https://hf-mirror.com      # or https://huggingface.co
    python scripts/download_models.py                  # download + verify
    python scripts/download_models.py --verify-only    # re-verify only
    python scripts/download_models.py --variant auk    # only the DiT/VAE release

Design notes
------------
* Expected SHA-256 values are **not hardcoded**. They are read from each file's
  Git-LFS pointer (``<endpoint>/<repo>/raw/<rev>/<file>``), which yields
  ``oid sha256:...`` + ``size``. Files that are not LFS-tracked are compared
  against the raw bytes themselves. This keeps the script correct if a repo is
  ever re-uploaded.
* Downloads stream straight into ``ckpts/`` (no HuggingFace cache copy), so peak
  disk usage equals the model size rather than twice it.
* Transfers are resumable via HTTP ``Range`` and land on a ``*.part`` file that
  is atomically renamed once complete.
* ``ckpts/.checksums.sha256`` is written in ``sha256sum -c`` format, so the
  result can be re-checked outside this script too.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

DEFAULT_ENDPOINT = "https://hf-mirror.com"
CHUNK_SIZE = 4 * 1024 * 1024
RESERVE_BYTES = 3 * 1024**3  # keep this much free for the rest of the toolchain
PROGRESS_INTERVAL = 5.0
MAX_INLINE_HASH = 64 * 1024**2  # refuse to hash a big non-LFS file in memory

REPO_ROOT = Path(__file__).resolve().parent.parent

# variant -> (destination subdirectory, repo_id, [filenames])
MODEL_SETS: dict[str, tuple[str, str, list[str]]] = {
    "auk": (
        "AuK",
        "tencent/AuK",
        [
            "config.yaml",
            "vae.safetensors",
            "auk_base.safetensors",
        ],
    ),
    "flash": (
        "AuK-Flash",
        "tencent/AuK-Flash",
        [
            "config.yaml",
            "vae.safetensors",
            "auk_flash.safetensors",
        ],
    ),
    "qwen": (
        "Qwen2.5-Omni-3B",
        "Qwen/Qwen2.5-Omni-3B",
        [
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "chat_template.json",
            "added_tokens.json",
            "merges.txt",
            "vocab.json",
            "model.safetensors.index.json",
            "spk_dict.pt",
            "tokenizer.json",
            "tokenizer_config.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ],
    ),
}


def fmt_bytes(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:6.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def resolve_endpoint(explicit: str) -> str:
    """
    返回一个真正可达的 endpoint。

    容器镜像常在 PID 1 里预设一个构建期残留的 ``HF_ENDPOINT``（域名解析失败）。
    沿用会让全部下载失败，所以这里探测一下，不可达就退回默认加速镜像。
    """
    candidates = [explicit]
    default = os.environ.get("AUK_HF_ENDPOINT", DEFAULT_ENDPOINT)
    if default not in candidates:
        candidates.append(default)

    for candidate in candidates:
        if candidate == DEFAULT_ENDPOINT and not endpoint_reachable(candidate):
            continue
        if endpoint_reachable(candidate):
            if candidate != explicit:
                print(f"  ! endpoint {explicit} 不可达，改用 {candidate}")
            return candidate
    raise SystemExit(
        f"No reachable HuggingFace endpoint (tried {', '.join(candidates)}). Set AUK_HF_ENDPOINT to a working mirror."
    )


def endpoint_reachable(endpoint: str) -> bool:
    try:
        response = requests.get(f"{endpoint}/", timeout=8, allow_redirects=False)
        return response.status_code < 500
    except requests.RequestException:
        return False


def bytes_still_to_fetch(dest: Path, expected_size: int) -> int:
    """
    还要下载多少字节。

    已存在的 ``*.part``（aria2c 断点续传文件）与目标文件本身都算已到位，
    否则中断续跑时预检会把整份文件当成全新下载，平白多要一倍空间。
    """
    partial = dest.with_name(dest.name + ".part")
    if partial.is_file():
        have = min(partial.stat().st_size, expected_size)
    elif dest.is_file():
        have = min(dest.stat().st_size, expected_size)
    else:
        have = 0
    return max(0, expected_size - have)


def find_hash_twin(ckpt_dir: Path, sha: str) -> Path | None:
    """
    在既有 manifest 里找一个 SHA-256 相同的文件，用于硬链接复用。

    AuK 与 AuK-Flash 的 ``vae.safetensors`` 是同一个文件（哈希一致），
    硬链接过去即可，不占额外磁盘。返回 None 表示没有可复制的。
    """
    manifest = ckpt_dir / ".checksums.sha256"
    if not manifest.is_file():
        return None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] == sha:
            candidate = ckpt_dir / parts[1].strip()
            if candidate.is_file():
                return candidate
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_variants(selected: list[str] | None) -> list[str]:
    if not selected or "all" in selected:
        return list(MODEL_SETS)
    ordered = [name for name in MODEL_SETS if name in selected]
    if not ordered:
        raise SystemExit(f"No known variant in {selected!r}; choose from {list(MODEL_SETS)} or 'all'.")
    return ordered


def resolve_revision(session: requests.Session, endpoint: str, repo_id: str) -> str:
    """Pin to the current main commit so a re-run is reproducible."""
    try:
        response = session.get(f"{endpoint}/api/models/{repo_id}", timeout=30)
        if response.status_code == 200:
            sha = response.json().get("sha")
            if sha:
                return sha
    except (requests.RequestException, ValueError):
        pass
    print(f"  ! could not resolve a commit sha for {repo_id}; falling back to 'main'")
    return "main"


def expected_spec(session: requests.Session, endpoint: str, repo_id: str, revision: str, filename: str) -> tuple[str, int]:
    """Return ``(sha256, size)`` for a remote file, without downloading the payload."""
    url = f"{endpoint}/{repo_id}/raw/{revision}/{quote(filename)}"
    response = session.get(url, timeout=120)
    response.raise_for_status()
    body = response.content
    if body.startswith(b"version https://git-lfs.github.com/spec/v1"):
        oid = size = None
        for line in body.decode("utf-8", "replace").splitlines():
            if line.startswith("oid sha256:"):
                oid = line.split(":", 1)[1].strip()
            elif line.startswith("size "):
                size = int(line.split()[1])
        if not oid or size is None:
            raise RuntimeError(f"Malformed Git-LFS pointer for {repo_id}/{filename}")
        return oid, size
    # Not an LFS pointer: this raw response body *is* the expected file content.
    if len(body) > MAX_INLINE_HASH:
        raise RuntimeError(f"{repo_id}/{filename} is a large non-LFS file; refusing to hash it in memory.")
    return hashlib.sha256(body).hexdigest(), len(body)


def check_file(path: Path, expected_sha: str) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    digest = sha256_file(path)
    if digest != expected_sha:
        return False, f"sha256 mismatch (got {digest[:16]}…, want {expected_sha[:16]}…)"
    return True, f"{fmt_bytes(path.stat().st_size)} sha256 ok"


def _stream_to_file(response: requests.Response, part: Path, have: int, expected_size: int, label: str) -> bool:
    """Stream ``response`` into ``part``. Returns True when the transfer was resumed."""
    if response.status_code not in (200, 206):
        raise RuntimeError(f"{label}: unexpected HTTP {response.status_code}")
    resumed = have > 0 and response.status_code == 206
    done = have
    started = time.time()
    last_print = started
    mode = "ab" if have else "wb"
    with open(part, mode) as handle:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            handle.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last_print >= PROGRESS_INTERVAL and expected_size:
                speed = (done - have) / max(now - started, 1e-6)
                eta = (expected_size - done) / speed if speed > 0 else 0.0
                print(
                    f"    {done / expected_size * 100:5.1f}%  {fmt_bytes(done)}/{fmt_bytes(expected_size)}"
                    f"  {fmt_bytes(speed)}/s  eta {eta:5.0f}s",
                    flush=True,
                )
                last_print = now
    if part.stat().st_size != expected_size:
        raise RuntimeError(f"{label}: incomplete download ({part.stat().st_size} != {expected_size})")
    return resumed


def aria2_available() -> bool:
    return shutil.which("aria2c") is not None


def download_aria2(url: str, dest: Path, expected_size: int, threads: int) -> str:
    """
    用 aria2c 多线程 + 续传下载。

    ``-x/-s`` 拆分多连接，``--continue`` 断点续传，``--auto-file-renaming=false``
    保证落名可控。resume 状态由 aria2c 自己的 ``<out>.aria2`` 控制文件维护。
    """
    part = dest.with_name(dest.name + ".part")
    command = [
        "aria2c",
        "--dir",
        str(dest.parent),
        "--out",
        part.name,
        "--continue=true",
        "--max-tries=8",
        "--retry-wait=3",
        "--timeout=60",
        "--split",
        str(threads),
        "--max-connection-per-server",
        str(threads),
        "--min-split-size",
        "1M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--console-log-level=warn",
        "--summary-interval=10",
        url,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"aria2c exited with {result.returncode}")
    if not part.is_file() or part.stat().st_size != expected_size:
        raise RuntimeError(f"incomplete download ({part.stat().st_size if part.exists() else 0} != {expected_size})")
    resumed = part.with_name(part.name + ".aria2").exists()
    os.replace(part, dest)
    # aria2c 的控制文件只在未完成时存在；这里清理可能残留的元数据
    part.with_name(part.name + ".aria2").unlink(missing_ok=True)
    return "resumed" if resumed else "downloaded"


def download_curl(session: requests.Session, url: str, dest: Path, expected_size: int, label: str) -> str:
    """单线程续传兜底（没有 aria2c 时使用）。"""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}

    with session.get(url, stream=True, headers=headers, timeout=60) as response:
        if have and response.status_code != 206:
            # server ignored our Range (or the partial is stale) -> restart cleanly
            part.unlink(missing_ok=True)
            have = 0
            response.close()
            with session.get(url, stream=True, timeout=60) as fresh:
                _stream_to_file(fresh, part, 0, expected_size, label)
                resumed = False
        else:
            response.raise_for_status()
            resumed = _stream_to_file(response, part, have, expected_size, label)

    os.replace(part, dest)
    return "resumed" if resumed else "downloaded"


def download(session: requests.Session, url: str, dest: Path, expected_size: int, label: str, threads: int = 16) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == expected_size:
        return "already present"
    if aria2_available():
        return download_aria2(url, dest, expected_size, threads)
    return download_curl(session, url, dest, expected_size, label)


def write_manifest(entries: dict[str, str], ckpt_dir: Path) -> Path:
    """
    把本次校验通过的条目**合并**进现有 manifest 后写回。

    必须是合并而不是覆盖：按 ``--variant`` 分批下载时，后一次运行不能把先前
    已登记的文件冲掉，否则 ``--verify-only`` 只校验到本次的文件。
    """
    manifest = ckpt_dir / ".checksums.sha256"
    merged: dict[str, str] = {}
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                merged[parts[1].strip()] = parts[0]
    merged.update(entries)

    with open(manifest, "w", encoding="utf-8") as handle:
        for relative, sha in sorted(merged.items()):
            handle.write(f"{sha}  {relative}\n")
    return manifest


def verify_manifest(ckpt_dir: Path) -> int:
    manifest = ckpt_dir / ".checksums.sha256"
    if not manifest.is_file():
        print(f"[ERROR] no manifest at {manifest}; run the download first.")
        return 1
    failures = 0
    rows = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in rows:
        sha, relative = line.split(None, 1)
        ok, detail = check_file(ckpt_dir / relative, sha)
        print(f"  {'OK  ' if ok else 'FAIL'} {relative} — {detail}")
        failures += 0 if ok else 1
    print(f"\n{'✅' if not failures else '❌'} verified {len(rows)} file(s), {failures} failure(s)")
    return 1 if failures else 0


def build_plan(session: requests.Session, endpoint: str, variant_names: list[str], ckpt_dir: Path) -> list[dict]:
    """Resolve revision, expected sha256 and size for every file in the manifest."""
    plan: list[dict] = []
    for variant in variant_names:
        dest_subdir, repo_id, filenames = MODEL_SETS[variant]
        revision = resolve_revision(session, endpoint, repo_id)
        print(f"[{variant}] {repo_id} @ {revision[:12]}")
        for filename in filenames:
            sha, size = expected_spec(session, endpoint, repo_id, revision, filename)
            dest = ckpt_dir / dest_subdir / filename
            plan.append(
                {
                    "variant": variant,
                    "repo_id": repo_id,
                    "revision": revision,
                    "filename": filename,
                    "sha": sha,
                    "size": size,
                    "dest": dest,
                }
            )
            print(f"    {filename:<34} {fmt_bytes(size):>12}  {sha[:16]}…")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download AuK checkpoints into ckpts/ and verify their SHA-256 integrity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=[*MODEL_SETS, "all"],
        default=None,
        help="which model set to download (repeatable; default: all)",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="HuggingFace-compatible endpoint (mirror); AUK_HF_ENDPOINT overrides the default",
    )
    parser.add_argument("--ckpt-dir", default=str(REPO_ROOT / "ckpts"), help="destination root")
    parser.add_argument("--verify-only", action="store_true", help="only re-check the existing files")
    parser.add_argument("--force", action="store_true", help="re-download even if the hash already matches")
    parser.add_argument("--yes", action="store_true", help="skip the interactive size confirmation")
    parser.add_argument(
        "--reserve",
        type=float,
        default=RESERVE_BYTES,
        help=f"required free space left after download, in GiB (default {RESERVE_BYTES / 1024**3:.0f})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="aria2c connections per file (multi-threaded download + resume)",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    endpoint = resolve_endpoint(args.endpoint.rstrip("/"))

    if args.verify_only:
        return verify_manifest(ckpt_dir)

    variant_names = resolve_variants(args.variant)
    session = requests.Session()
    session.headers.update({"User-Agent": "auk-download-models/1.0"})

    print(f"Endpoint : {endpoint}")
    print(f"Variants : {', '.join(variant_names)}")
    print(f"Target   : {ckpt_dir}")
    print(
        f"Downloader : {'aria2c (multi-threaded, %d connections)' % args.threads if aria2_available() else 'curl fallback (single-threaded)'}\n"
    )

    plan = build_plan(session, endpoint, variant_names, ckpt_dir)

    # Smallest first, so an interrupted run leaves the most useful files behind.
    plan.sort(key=lambda row: row["size"])

    # 统计真正还要落盘的量：已存在/已部分下载的不算；能硬链接复用的也几乎不占空间
    missing = 0
    hardlink_bytes = 0
    for row in plan:
        dest: Path = row["dest"]
        if dest.is_file() and dest.stat().st_size == row["size"]:
            continue
        twin = None if args.force else find_hash_twin(ckpt_dir, row["sha"])
        if twin is not None and twin != dest:
            hardlink_bytes += row["size"]
            continue
        missing += bytes_still_to_fetch(dest, row["size"])

    free = shutil.disk_usage(ckpt_dir).free
    reserve = int(args.reserve * 1024**3)
    detail = f"{fmt_bytes(missing)} new"
    if hardlink_bytes:
        detail += f" + {fmt_bytes(hardlink_bytes)} hardlinked (≈0 disk)"
    print(f"\nRequired : {detail} / {fmt_bytes(free)} free (reserve {fmt_bytes(reserve)})")
    if missing + reserve > free:
        print(f"[ERROR] not enough free space in {ckpt_dir}; aborting before downloading.")
        print(f"         need {fmt_bytes(missing + reserve)}, have {fmt_bytes(free)}.")
        return 1
    if not args.yes:
        if input("Proceed with download? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    manifest_entries: dict[str, str] = {}
    failures = 0
    for row in plan:
        label = f"{row['repo_id']}/{row['filename']}"
        url = f"{endpoint}/{row['repo_id']}/resolve/{row['revision']}/{quote(row['filename'])}"
        dest: Path = row["dest"]
        try:
            if args.force and dest.is_file():
                dest.unlink()
            # 同一文件的另一份副本已存在且校验通过 -> 硬链接复用，零额外磁盘
            twin = None if args.force else find_hash_twin(ckpt_dir, row["sha"])
            if twin is not None and twin != dest:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    os.link(twin, dest)
                state = f"hardlink from {twin.relative_to(ckpt_dir)}"
            else:
                state = download(session, url, dest, row["size"], label, threads=args.threads)
            ok, detail = check_file(dest, row["sha"])
            if not ok:
                raise RuntimeError(detail)
            manifest_entries[str(dest.relative_to(ckpt_dir))] = row["sha"]
            print(f"  ✓ {dest.relative_to(ckpt_dir)}  [{state}] {detail}")
        except Exception as exc:  # noqa: BLE001 - report and keep going with the rest
            failures += 1
            print(f"  ✗ {label}: {type(exc).__name__}: {exc}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(manifest_entries, ckpt_dir)
    print(f"\nWrote {manifest} ({len(manifest_entries)} entries)")
    if failures:
        print(f"❌ {failures} file(s) failed; re-run to retry (transfers are resumable).")
        return 1
    print("✅ all files present and SHA-256 verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
