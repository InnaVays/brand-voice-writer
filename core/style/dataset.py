# core/style/dataset.py
from __future__ import annotations
import glob
import os
from typing import List


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_local_corpus(dir_path: str) -> List[str]:
    """
    Load all .txt and .md files from a directory.
    """

    SUPPORTED_EXT = {".txt", ".md"}

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    files = []
    for ext in SUPPORTED_EXT:
        files.extend(glob.glob(os.path.join(dir_path, f"**/*{ext}"), recursive=True))
    texts = []
    for p in sorted(files):
        try:
            t = read_text_file(p).strip()
            if t:
                texts.append(t)
        except Exception:
            # skip unreadable files
            continue
    return texts

def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    """
    Super chunking by words.
    """
    words = text.split()
    out = []
    for i in range(0, len(words), max_tokens):
        out.append(" ".join(words[i:i + max_tokens]))
    return out

def build_corpus(dir_path: str, max_tokens: int = 300) -> List[str]:
    """
    Load and chunk all texts. Returns a list of chunks.
    """
    docs = load_local_corpus(dir_path)
    chunks: List[str] = []
    for t in docs:
        chunks.extend(chunk_text(t, max_tokens=max_tokens))
    return chunks
