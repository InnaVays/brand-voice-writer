from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, Any

DEFAULT_REGISTRY = {"models": {}, "styles": {}}

class Registry:
    def __init__(self, path: str):
        self.path = Path(path)
        self.data: Dict[str, Any] = DEFAULT_REGISTRY.copy()
        if self.path.exists():
            self.data.update(json.loads(self.path.read_text(encoding="utf-8")))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- models ----
    def set_model(self, key: str, info: Dict[str, Any]):
        self.data.setdefault("models", {})[key] = info
        self.save()

    def get_model(self, key: str) -> Dict[str, Any] | None:
        return self.data.get("models", {}).get(key)

    # ---- styles ----
    def upsert_style(self, style_id: str, dataset_path: str | None = None, peft_dir: str | None = None, meta: Dict[str, Any] | None = None):
        styles = self.data.setdefault("styles", {})
        s = styles.setdefault(style_id, {})
        if dataset_path: s["dataset"] = dataset_path
        if peft_dir:     s["peft_dir"] = peft_dir
        if meta:         s.setdefault("meta", {}).update(meta)
        self.save()

    def get_style(self, style_id: str) -> Dict[str, Any] | None:
        return self.data.get("styles", {}).get(style_id)

    def list_styles(self) -> dict:
        return self.data.get("styles", {})
