# pipeline/rewrite.py
from __future__ import annotations
import argparse

from model.gemma_peft import (
    load_gemma_base,
    load_prompt_tuning,
    GemmaPeftGenerator,
)
from model.base import GenConfig


PROMPT_TMPL = (
    "Rewrite the following text in the learned brand voice. "
    "Preserve facts, improve clarity, keep length roughly similar.\n\n"
    "INPUT:\n{src}\n\nOUTPUT:\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Path to saved PEFT adapter dir")
    ap.add_argument("--text", required=True, help="Text to rewrite")
    ap.add_argument("--model_id", type=str, default="google/gemma-2b-it")
    args = ap.parse_args()

    tok, base, device = load_gemma_base(args.model_id)
    peft = load_prompt_tuning(base, args.adapter)
    gen = GemmaPeftGenerator(tok, peft, device=device)

    prompt = PROMPT_TMPL.format(src=args.text.strip())
    out = gen.generate(prompt, cfg=GenConfig(max_new_tokens=256, temperature=0.7))
    print(out.text.strip())


if __name__ == "__main__":
    main()
