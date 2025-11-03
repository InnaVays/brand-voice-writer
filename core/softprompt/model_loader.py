from __future__ import annotations
import os
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

def load_model(model_id: str, device: str = "auto", attn: bool = True, hidden: bool = True):
    d = resolve_device(device)

    if d == "cuda":
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            pass

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=(torch.float16 if d == "cuda" else None),
        low_cpu_mem_usage=True,
        attn_implementation="eager",   # 2) force eager attention
    )

    # 3) ensure config reflects this, then enable outputs
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass
    model.config.output_attentions = bool(attn)
    model.config.output_hidden_states = bool(hidden)

    model.to(d)
    return tok, model, d
