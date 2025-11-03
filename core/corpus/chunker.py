from __future__ import annotations
from typing import List
from core.utils.text import split_sentences

class Chunker:
    def make_chunks(
        self,
        text: str,   
        sentences_per_chunk: int = 3 
    ) -> List[str]:
        sents = split_sentences(text)
        chunks: List[str] = []
        cur: List[str] = []
        cur_count = 0

        for s in sents:
            cur.append(s)
            cur_count += 1
            if cur_count >= sentences_per_chunk:
                chunks.append(" ".join(cur))
                cur = []
                cur_count = 0
                
        if cur:
            chunks.append(" ".join(cur))
        
        return chunks

    def sample_target(self, chunks: List[str], target_n: int = 60) -> List[str]:
        if len(chunks) <= target_n:
            return chunks
        # stratify by index: keep every k-th
        k = max(1, len(chunks) // target_n)
        return [c for i, c in enumerate(chunks) if i % k == 0][:target_n]
