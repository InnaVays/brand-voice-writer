from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict
import docx2txt
from pypdf import PdfReader
from core.utils.text import normalize_whitespace

ACCEPT = {".docx", ".pdf", ".txt", ".md"}

class DocLoader:
    def load_dir(self, path: str | Path) -> List[Dict]:
        base = Path(path)
        if not base.exists():
            return []
        items: List[Dict] = []
        for p in base.rglob("*"):
            if not p.is_file(): 
                continue
            if p.suffix.lower() not in ACCEPT:
                continue
            txt = self._read_one(p)
            if not txt:
                continue
            items.append({
                "id": p.stem,
                "text": normalize_whitespace(txt),
                "meta": {"path": str(p), "ext": p.suffix.lower()}
            })
        return items

    def _read_one(self, p: Path) -> str:
        ext = p.suffix.lower()
        if ext == ".txt" or ext == ".md":
            return p.read_text(encoding="utf-8", errors="ignore")
        if ext == ".docx":
            return docx2txt.process(str(p)) or ""
        if ext == ".pdf":
            try:
                reader = PdfReader(str(p))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                return ""
        return ""
