from __future__ import annotations
import argparse, yaml
from core.models.registry import Registry
from core.softprompt.gemma_rewriter import GemmaRewriter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/app.yaml")
    ap.add_argument("--style", required=True, help="style_id saved in registry")
    ap.add_argument("--text", required=True)
    ap.add_argument("--tone", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    reg = Registry(cfg["paths"]["registry"])
    s   = reg.get_style(args.style)
    if not s or "peft_dir" not in s:
        raise SystemExit(f"Style '{args.style}' has no trained PEFT weights. Train it first.")

    model_id = cfg["models"]["rewriter_base"]["hf_id"]
    rw = GemmaRewriter(model_id=model_id, peft_dir=s["peft_dir"], device=cfg["models"]["rewriter_base"].get("device","auto"))
    out = rw.rewrite(args.text, tone=args.tone)
    print(out)

if __name__ == "__main__":
    main()
