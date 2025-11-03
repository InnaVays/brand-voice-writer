import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(model_id: str, device: str="cpu"):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        dtype=torch.float32,
        use_safetensors=True,
        device_map={"": "cpu"}
    )
    return tok, model, "cpu"