from __future__ import annotations
import hashlib, os, urllib.request
from pathlib import Path
from typing import Optional

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_download(url: str, dest_dir: str, filename: str, sha256: str | None = None) -> str:
    dest = Path(dest_dir); dest.mkdir(parents=True, exist_ok=True)
    out = dest / filename
    if out.exists() and (not sha256 or _sha256(out) == sha256):
        return str(out)
    # download
    tmp = out.with_suffix(out.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # you can add progress hooks if you like
    if sha256:
        got = _sha256(tmp)
        if got.lower() != sha256.lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {filename}: {got} != {sha256}")
    tmp.replace(out)
    return str(out)
