# pipeline/dataset.py
from __future__ import annotations
from typing import Dict, List
from torch.utils.data import Dataset
import torch


class PromptTuningDataset(Dataset):
    """
    Creates (input_ids, labels) where labels are -100 for the prompt part
    and actual ids for the target part. This trains the model to produce the
    user's original text (autoencoding in their style), conditioning on essence+sample.
    """
    def __init__(self, tokenizer, examples: List[Dict[str, str]]):
        self.tok = tokenizer
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        prompt = ex["prompt"]
        target = ex["target"]

        # construct input: [PROMPT] + target; labels mask prompt
        full = prompt + target
        enc = self.tok(full, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]

        p_enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        p_len = p_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:p_len] = -100  # mask prompt tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }
