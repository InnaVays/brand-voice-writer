from __future__ import annotations
import hashlib, os, urllib.request, contextlib
from pathlib import Path
from typing import Optional

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _download_url(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(dest_path.suffix + ".part")
    with contextlib.ExitStack() as stack:
        stack.enter_context(open(tmp, "wb"))
        urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest_path)

def ensure_local(local_path: str) -> str:
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(f"Local model not found: {p}")
    return str(p)

def ensure_url(url: str, dest_dir: str, filename: str, sha256: str | None = None) -> str:
    dest = Path(dest_dir) / filename
    if dest.exists() and (not sha256 or _sha256(dest) == sha256):
        return str(dest)
    _download_url(url, dest)
    if sha256 and _sha256(dest).lower() != sha256.lower():
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {filename}")
    return str(dest)

def ensure_hf(repo_id: str, filename: str, dest_dir: str) -> str:
    """
    Download from Hugging Face Hub using huggingface_hub. Requires `pip install huggingface_hub`.
    If the repo is gated, user must `huggingface-cli login` or set HF_TOKEN.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError("huggingface_hub is not installed. `pip install huggingface_hub`") from e

    dest_dir_p = Path(dest_dir); dest_dir_p.mkdir(parents=True, exist_ok=True)
    # This stores under HF cache; we copy/link into our models cache for stable path
    cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
    target = dest_dir_p / filename
    if not target.exists():
        # copy the file into our models cache dir for predictable path
        from shutil import copy2
        copy2(cached_path, target)
    return str(target)

def ensure_model(source: str, dest_dir: str, *, url: str = "", filename: str = "", repo_id: str = "", local_path: str = "", sha256: str | None = None) -> str:
    """
    Resolve a model path based on the given source type:
    - "local": use existing file
    - "url":   download from direct URL
    - "hf":    download via huggingface_hub
    """
    if source == "local":
        if not local_path:
            raise ValueError("local_path must be set when source='local'")
        return ensure_local(local_path)
    elif source == "url":
        if not url or not filename:
            raise ValueError("url and filename must be set when source='url'")
        return ensure_url(url, dest_dir, filename, sha256)
    elif source == "hf":
        if not repo_id or not filename:
            raise ValueError("repo_id and filename must be set when source='hf'")
        return ensure_hf(repo_id, filename, dest_dir)
    else:
        raise ValueError(f"Unknown model source: {source}")