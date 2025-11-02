#from __future__ import annotations
from typing import Optional, Dict, Any
from llama_cpp import Llama
import re
from core.models.downloader import ensure_download
from core.models.registry import Registry

_PROMPT = """You are a concise neutral summarizer.
Summarize the text in exactly 1-2 plain sentences. Be factual and avoid hype or emojis.

[TEXT]
{chunk}
[/TEXT]

Summary:
"""

_WS = re.compile(r"\s+")
def _clean(s: str) -> str:
    s = s.strip()
    s = _WS.sub(" ", s)
    if s and s[-1] not in ".!?…": s += "."
    return s

class GistBuilder:
    """
    llama.cpp-only summarizer with auto-download and registry bookkeeping.
    """
    def __init__(self, cfg: Dict[str, Any], registry: Registry):
        mcfg = cfg["models"]["summarizer"]
        cache = cfg["paths"]["models_cache"]
        local_path = ensure_download(mcfg["url"], cache, mcfg["name"], mcfg.get("sha256") or None)
        # record in registry
        registry.set_model("summarizer", {"path": local_path, "name": mcfg["name"]})
        self.stop = ("[/TEXT]", "###", "</s>", "Summary:", "[END]")
        self.llm = Llama(
            model_path=local_path,
            n_ctx=mcfg.get("n_ctx", 2048),
            n_threads=mcfg.get("n_threads", 6),
            n_gpu_layers=mcfg.get("n_gpu_layers", 0),
            verbose=False,
        )
        self.max_new = int(mcfg.get("max_new", 96))
        self.temperature = float(mcfg.get("temperature", 0.0))
        self.top_p = float(mcfg.get("top_p", 0.9))

    def gist(self, chunk: str, max_chars: int = 4000) -> str:
        prompt = _PROMPT.format(chunk=(chunk or "")[:max_chars])
        out = self.llm(
            prompt,
            max_tokens=self.max_new,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop,
        )
        text = out["choices"][0]["text"].strip()
        if text.lower().startswith("summary:"):
            text = text[len("summary:"):].strip()
        text = text.split("\n")[0].strip()
        return _clean(text)
