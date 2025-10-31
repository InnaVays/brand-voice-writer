from __future__ import annotations
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        if torch.cuda.is_available(): return "cuda"
        try:
            if torch.backends.mps.is_available(): return "mps"
        except Exception:
            pass
        return "cpu"
    return device

def load_gemma(model_id: str, device: str = "auto", attn: bool = True, hidden: bool = True):
    d = resolve_device(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if d == "cuda" else None,
    )
    model.config.output_attentions = attn
    model.config.output_hidden_states = hidden
    model.to(d)
    return tok, model, d
