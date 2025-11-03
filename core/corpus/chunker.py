from __future__ import annotations
from typing import List
from core.utils.text import split_sentences, word_count

class Chunker:
    def make_chunks(self, text: str, target_words: int = 60) -> List[str]:
        sents = split_sentences(text)
        chunks, cur, cur_wc = [], [], 0
        for s in sents:
            wc = word_count(s)
            if cur_wc + wc > target_words and cur:
                chunks.append(" ".join(cur))
                cur, cur_wc = [], 0
            cur.append(s)
            cur_wc += wc
        if cur:
            chunks.append(" ".join(cur))
        return [c for c in chunks if word_count(c) >= max(40, int(0.3 * target_words))]

    def sample_target(self, chunks: List[str], target_n: int = 60) -> List[str]:
        if len(chunks) <= target_n:
            return chunks
        # stratify by length (simple): keep every k-th
        k = max(1, len(chunks) // target_n)
        return [c for i, c in enumerate(chunks) if i % k == 0][:target_n]
