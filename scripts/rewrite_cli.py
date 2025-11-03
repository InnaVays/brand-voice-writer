from __future__ import annotations

from _bootstrap import add_repo_root
add_repo_root()

import argparse
from pathlib import Path
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ---------- utils ----------

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


def _latest_run_dir(softprompt_root: str, style_id: str) -> Path:
    """
    Returns the run_* for a style.
    artifacts/softprompt/<style>/run_XXX
    """
    style_root = Path(softprompt_root) / style_id
    if not style_root.exists():
        raise SystemExit(f"No adapters found for style '{style_id}' in {style_root}")

    runs = []
    for p in style_root.glob("run_*"):
        if p.is_dir():
            try:
                idx = int(p.name.split("_")[-1])
                runs.append((idx, p))
            except ValueError:
                continue

    if not runs:
        raise SystemExit(f"No run_* directories found in {style_root}")

    runs.sort(key=lambda x: x[0])
    return runs[-1][1] 


def load_styled_model(config_path: str, style_id: str, run: str | None, device: str):
    """
    """
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    model_id = cfg["model_id"]
    softprompt_root = cfg["paths"]["softprompt"]

    d = _resolve_device(device)
    if run:
        adapter_dir = Path(softprompt_root) / style_id / run
    else:
        adapter_dir = _latest_run_dir(softprompt_root, style_id)

    if not adapter_dir.exists():
        raise SystemExit(f"Adapter directory not found: {adapter_dir}")

    print(f"[LOAD] Base model: {model_id}")
    print(f"[LOAD] Adapter:    {adapter_dir}")

    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    dtype = torch.float16 if d == "cuda" else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=dtype,
    ).to(d)
    base.eval()

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(d)
    model.eval()

    return tok, model, d


# ---------- core rewrite ----------

def rewrite_text(tok, model, device: str, text: str, style_id: str,
                 max_new_tokens: int | None = None,
                 temperature: float = 0.0, 
                 top_p: float = 0.9) -> str:
    inp_ids = tok(
        text.strip(),
        add_special_tokens=False
    )["input_ids"]
    inp_len = len(inp_ids)

    target_max = min(inp_len + 20, 64)  

    if max_new_tokens is None:
        max_new_tokens = target_max

    system_prompt = (
        f"You are a brand voice rewriter. "
        f"You rewrite user text into the '{style_id}' brand voice. "
        f"Preserve meaning and approximate length (no more than +50% longer). "
        f"Answer with a SINGLE rewritten version"
        f"Do NOT offer alternatives, options, or explanations. "
        f"Your reply must contain ONLY the rewritten text."
    )

    prompt = (
            f"{system_prompt}\n\n"
            "Original text:\n"
            f"{text.strip()}\n\n"
            "Rewritten text:\n"
        )
    inputs = tok(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,             
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
        )

    generated_ids = out[0]

    prompt_len = inputs["input_ids"].shape[1]

    gen_only_ids = generated_ids[prompt_len:]

    decoded = tok.decode(gen_only_ids, skip_special_tokens=True).strip()

    return decoded


# ---------- CLI main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/app.yaml")
    ap.add_argument("--style", required=True, help="Style ID, e.g. fintech")
    ap.add_argument("--run", default=None, help="Run name, e.g. run_003 (optional, default latest)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--text", default=None, help="Input text to rewrite (if omitted, read from stdin)")
    args = ap.parse_args()

    tok, model, d = load_styled_model(
        config_path=args.config,
        style_id=args.style,
        run=args.run,
        device=args.device,
    )

    if args.text is not None:
        src = args.text
    else:
        print("Enter text to rewrite. Finish with Ctrl+D (Linux/macOS) or Ctrl+Z (Windows):")
        src = "".join(iter(input, ""))

    out = rewrite_text(
        tok, model, d,
        text=src,
        style_id=args.style,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n=== REWRITTEN TEXT ===\n")
    print(out)
    print("\n======================\n")


if __name__ == "__main__":
    main()
