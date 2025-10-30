# pipeline/data.py
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Iterable

from pypdf import PdfReader
from docx import Document as DocxDocument


@dataclass
class Chunk:
    text: str
    words: int


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for p in reader.pages:
        try:
            parts.append(p.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def extract_text_from_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text])


def read_docs(paths: List[str]) -> str:
    buff = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            buff.append(extract_text_from_pdf(p))
        elif ext in (".docx", ".doc"):
            buff.append(extract_text_from_docx(p))
        elif ext in (".txt", ".md"):
            buff.append(open(p, "r", encoding="utf-8", errors="ignore").read())
        else:
            # skip unsupported
            continue
    text = "\n".join(buff)
    # normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def words_iter(s: str) -> Iterable[str]:
    for w in re.findall(r"[A-Za-zА-Яа-яЁё0-9\-]+", s):
        yield w


def chunk_text_by_words(s: str, target_words: int = 120, stride_words: int = 90) -> List[Chunk]:
    w = list(words_iter(s))
    chunks: List[Chunk] = []
    i = 0
    n = len(w)
    while i < n:
        j = min(i + target_words, n)
        piece = " ".join(w[i:j])
        if len(piece.split()) >= max(30, int(0.6 * target_words)):  # discard too-short scraps
            chunks.append(Chunk(text=piece, words=len(piece.split())))
        i += stride_words
    return chunks


def ensure_at_least_k_chunks(chunks: List[Chunk], k: int = 120) -> List[Chunk]:
    if len(chunks) >= k:
        return chunks[:k]
    # reuse with offset shuffles
    out = list(chunks)
    idx = 0
    while len(out) < k and chunks:
        out.append(chunks[idx % len(chunks)])
        idx += 1
    return out
