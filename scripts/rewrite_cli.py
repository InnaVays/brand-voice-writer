from __future__ import annotations
import argparse, os
from core.model.rewriter import GemmaRewriter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peft", required=True, help="Path to artifacts/softprompt/<style_id>")
    ap.add_argument("--model", default="google/gemma-2b-it")
    ap.add_argument("--in", dest="text", required=True, help="Text to rewrite")
    ap.add_argument("--tone", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    rw = GemmaRewriter(model_id=args.model, peft_dir=args.peft, device=args.device)
    out = rw.rewrite(args.text, tone=args.tone)
    print(out)

if __name__ == "__main__":
    main()
