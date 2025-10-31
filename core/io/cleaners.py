from __future__ import annotations
import re
from core.utils.text import normalize_whitespace

HEADER_FOOTER_RE = re.compile(r"^Page \d+ of \d+$", re.IGNORECASE)

class Cleaner:
    def normalize(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            l = line.strip()
            if not l:
                lines.append("")
                continue
            if HEADER_FOOTER_RE.match(l):
                continue
            lines.append(l)
        txt = "\n".join(lines)
        return normalize_whitespace(txt)
