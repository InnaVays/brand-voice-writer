from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, Any, Optional

REGISTRY_FILE = "styles.json"

class StyleRegistry:
    """
    Tracks styles, dataset paths, and trained adapters.
    artifacts/registry/styles.json:
    {
      "styles": {
        "fintech": {"dataset": "data/datasets/fintech.jsonl", "adapter_dir": "artifacts/softprompt/fintech"},
        "compliance": {...}
      }
    }
    """
    def __init__(self, registry_dir: str):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.registry_dir / REGISTRY_FILE
        self.data: Dict[str, Any] = {"styles": {}}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass

    def upsert_style(self, style_id: str, dataset_path: Optional[str] = None, adapter_dir: Optional[str] = None):
        entry = self.data["styles"].get(style_id, {})
        if dataset_path: entry["dataset"] = dataset_path
        if adapter_dir:  entry["adapter_dir"] = adapter_dir
        self.data["styles"][style_id] = entry
        self._save()

    def get(self, style_id: str) -> Dict[str, Any]:
        return self.data["styles"].get(style_id, {})

    def list_styles(self):
        return sorted(self.data["styles"].keys())

    def _save(self):
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
