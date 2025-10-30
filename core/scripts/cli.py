# scripts/cli.py
from __future__ import annotations
import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Convenience wrapper for training/visualization/rewrite/report")
    sub = ap.add_subparsers(dest="cmd")

    t = sub.add_parser("train")
    t.add_argument("--files", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--extra", default="", help="extra flags for train_gemma")

    v = sub.add_parser("viz")
    v.add_argument("--adapter", required=True)

    r = sub.add_parser("rewrite")
    r.add_argument("--adapter", required=True)
    r.add_argument("--text", required=True)

    rep = sub.add_parser("report")
    rep.add_argument("--adapter", required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        cmd = ["python", "-m", "pipeline.train_gemma", "--files"] + args.files + ["--out", args.out] + args.extra.split()
    elif args.cmd == "viz":
        cmd = ["python", "-m", "pipeline.visualize_softprompt", "--adapter", args.adapter, "--out_dir", "figures"]
    elif args.cmd == "rewrite":
        cmd = ["python", "-m", "pipeline.rewrite", "--adapter", args.adapter, "--text", args.text]
    elif args.cmd == "report":
        cmd = ["python", "-m", "pipeline.posttrain_report", "--adapter", args.adapter, "--fig_dir", "figures", "--out", "reports"]
    else:
        ap.print_help()
        sys.exit(1)

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
