from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import json, os, time

@dataclass
class RunMetrics:
    seconds: float
    n_examples: int
    virtual_tokens: int
    epochs: int
    batch_size: int
    lr: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

def save_run_json(d: Dict[str, Any], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"run_{int(time.time())}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    return path
