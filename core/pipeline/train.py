# pipeline/train_gemma.py
from __future__ import annotations
import argparse
import os
import time
from typing import List, Dict

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from pipeline.data import read_docs, chunk_text_by_words, ensure_at_least_k_chunks
from model.gemma_peft import (
    load_gemma_base,
    attach_prompt_tuning,
    save_prompt_tuning,
)
from pipeline.dataset import PromptTuningDataset


INSTR = (
    "You are a Brand Voice Rewriter. Given a sample of the user's writing and its essence, "
    "you will rewrite new texts in the same style. First, learn the style by reconstructing "
    "the sample. Keep meaning and tone.\n\n"
)


def naive_essence(chunk: str, max_words: int = 40) -> str:
    # very simple extractive essence: first ~2 sentences truncated by words
    words = chunk.split()
    return " ".join(words[:max_words])


def build_examples(chunks: List[str]) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for c in chunks:
        essence = naive_essence(c, max_words=40)
        prompt = (
            f"{INSTR}"
            f"ESSENCE:\n{essence}\n\n"
            f"SAMPLE (style reference):\n{c}\n\n"
            f"OUTPUT (reconstruct in same style):\n"
        )
        target = c  # auto-encode to learn style
        examples.append({"prompt": prompt, "target": target})
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="Paths to user docs (.pdf/.docx/.txt/.md)")
    ap.add_argument("--out", required=True, help="Output dir for adapter (PEFT prompt-tuning)")
    ap.add_argument("--num_virtual_tokens", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--target_chunks", type=int, default=120)
    ap.add_argument("--chunk_words", type=int, default=120)
    ap.add_argument("--stride_words", type=int, default=90)
    ap.add_argument("--model_id", type=str, default="google/gemma-2b-it")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    raw = read_docs(args.files)
    chunks = chunk_text_by_words(raw, target_words=args.chunk_words, stride_words=args.stride_words)
    chunks = [c.text for c in ensure_at_least_k_chunks(chunks, k=args.target_chunks)]
    print(f"[DATA] Using {len(chunks)} chunks (≈{args.chunk_words} words each)")

    tok, base, device = load_gemma_base(args.model_id)
    peft_model = attach_prompt_tuning(base, num_virtual_tokens=args.num_virtual_tokens)

    ex = build_examples(chunks)
    ds = PromptTuningDataset(tok, ex)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    t0 = time.time()
    trainer = Trainer(
        model=peft_model,
        args=TrainingArguments(
            output_dir=args.out,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=20,
            save_steps=200,
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            fp16=False,
            report_to=[],
        ),
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    dt = time.time() - t0
    save_prompt_tuning(peft_model, args.out)
    print(f"[OK] Saved adapter to {args.out} in {dt/60:.1f} min")


if __name__ == "__main__":
    main()
