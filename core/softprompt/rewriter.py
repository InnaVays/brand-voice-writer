from __future__ import annotations
from typing import Optional
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

class GemmaRewriter:
    def __init__(self, model_id: str, peft_dir: str, device: str = "auto"):
        self.device = self._resolve_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else None
        ).to(self.device)
        self.model = PeftModel.from_pretrained(base, peft_dir).to(self.device)
        self.model.eval()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available(): return "cuda"
            try:
                if torch.backends.mps.is_available(): return "mps"
            except Exception:
                pass
            return "cpu"
        return device

    def rewrite(self, text: str, tone: Optional[str] = None, max_new_tokens: int = 256) -> str:
        tone_str = f" in a {tone} tone" if tone else ""
        prompt = (
            f"Rewrite the following text{tone_str} in the user's voice. "
            f"Preserve meaning and factual content. Improve clarity.\n\n"
            f"Text:\n{text}\n\nRewrite:"
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                eos_token_id=self.tok.eos_token_id
            )
        full = self.tok.decode(out[0], skip_special_tokens=True)
        return full.split("Rewrite:")[-1].strip()
