from __future__ import annotations
import re
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42

SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))

def split_sentences(text: str) -> list[str]:
    parts = SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]
