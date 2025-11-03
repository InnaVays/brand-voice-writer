from __future__ import annotations
from typing import Tuple
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _resolve_device(device: str = "auto") -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return device


def load_model(
    model_id: str,
    device: str = "auto",
    cache_dir: str = ".hf_cache",
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """
    Loader for decoder-only LMs like Qwen2.5-*B-Instruct.
    """
    os.makedirs(cache_dir, exist_ok=True)
    d = _resolve_device(device)
    dtype = torch.float16 if d == "cuda" else torch.float32


    try:
        tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=True,       
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            dtype=dtype,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        print(f"[load_model] Loaded {model_id} from LOCAL cache ({cache_dir}) on {d}")
    except Exception as e:
        print(f"[load_model] Local cache miss or error: {e}")
        print(f"[load_model] Downloading {model_id} into cache_dir={cache_dir} ...")

        tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            dtype=dtype,
            cache_dir=cache_dir,
        )
        print(f"[load_model] Downloaded {model_id} and cached to {cache_dir} on {d}")

    model.to(d)
    model.eval()
    return tok, model, d
