from __future__ import annotations

from scripts._bootstrap import add_repo_root
add_repo_root()

import argparse, os
from core.eval.report import visualize_softprompt_matrix, normalized_attention_difference

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peft", required=True, help="Path to artifacts/softprompt/<style_id>")
    ap.add_argument("--text", default="We help small teams make better decisions with data.")
    ap.add_argument("--out", default="artifacts/reports")
    args = ap.parse_args()

    fig1 = visualize_softprompt_matrix(args.peft, os.path.join(args.out, "figures"))
    fig2 = normalized_attention_difference(args.peft, args.text, os.path.join(args.out, "figures"))

    print("Saved figures:")
    print(" -", fig1)
    print(" -", fig2)

if __name__ == "__main__":
    main()
