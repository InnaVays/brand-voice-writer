from __future__ import annotations
import os
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def visualize_softprompt_matrix(peft_dir: str, out_dir: str):
    _ensure_dir(out_dir)
    # load soft prompt weights
    base = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
    peft = PeftModel.from_pretrained(base, peft_dir)
    w = peft.base_model.prompt_encoder.weight.detach().cpu().numpy()  # (tokens, hidden)
    fig, ax = plt.subplots(figsize=(6, 3))
    vmax = np.percentile(np.abs(w), 99)
    im = ax.imshow(w, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title("Soft Prompt Weights (tokens × hidden_dim)")
    fig.colorbar(im)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "softprompt_matrix.png")
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    return fig_path

def normalized_attention_difference(peft_dir: str, text: str, out_dir: str):
    _ensure_dir(out_dir)
    tok = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    base = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
    base.config.output_attentions = True
    base.eval()

    peft = PeftModel.from_pretrained(base, peft_dir)
    peft.eval()

    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out_base = base(**inputs, output_attentions=True)
        out_peft = peft(**inputs, output_attentions=True)

    # mean over heads & layers, attention to soft virtual tokens is at beginning
    # For PEFT prompt-tuning, virtual tokens are prepended internally before inputs.
    # We'll approximate by comparing overall attention distributions over sequence positions.
    def mean_attn(attn_list):
        # attn_list: list of (batch, heads, q_len, k_len)
        mats = [a.mean(dim=1).squeeze(0) for a in attn_list]  # (q_len, k_len)
        return torch.stack(mats, dim=0).mean(dim=0)  # (q_len, k_len)

    A_base = mean_attn(out_base.attentions)    # (q,k)
    A_peft = mean_attn(out_peft.attentions)    # (q,k)

    # Compare per-query token the mean attention over the first V virtual tokens.
    V = peft.base_model.prompt_encoder.weight.shape[0]
    q_len, k_len = A_peft.shape
    V = min(V, k_len)  # safety
    brand = A_peft[:, :V].mean(dim=1)          # (q_len,)
    neutral = A_base[:, :V].mean(dim=1)        # (q_len,)
    delta = (brand - neutral).cpu().numpy()
    delta = delta / (np.max(np.abs(delta)) + 1e-8)

    # Plot as 1×seq heat strip
    fig, ax = plt.subplots(figsize=(8, 1.2))
    im = ax.imshow(delta.reshape(1, -1), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_yticks([])
    ax.set_title("Normalized Attention Difference (brand vs baseline)")
    fig.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.2)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "attention_diff.png")
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    return fig_path
