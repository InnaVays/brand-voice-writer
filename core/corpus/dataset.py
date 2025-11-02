from __future__ import annotations
import json
from typing import List, Tuple

class DatasetBuilder:
    def build_jsonl(self, chunks_with_gist: List[Tuple[str, str]], out_path: str) -> None:
        """
        Instruction-tuning pairs:
        - instruction: fixed rewrite instruction
        - input: chunk + its gist (helps keep meaning)
        - output: the original chunk (identity target helps focus soft prompt on style)
        """
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk, gist in chunks_with_gist:
                rec = {
                    "instruction": "Rewrite the input in the user's voice while preserving meaning.",
                    "input": f"{gist}",
                    "output": chunk
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
