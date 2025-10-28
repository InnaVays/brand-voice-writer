# core/pipeline/rewriter.py
from __future__ import annotations
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from core.models.base import TextModel, GenerationConfig
from core.models.base import load_local_model
from core.trainer import SoftPromptWeights
from core.pipeline.scorer import evaluate_candidate
from core.style.space import BrandStyleSpace


@dataclass
class TaskSpec:
    """
    Minimal public TaskSpec for the public repo (one simple target form).
    Pro will extend this with Blueprints and sections.
    """
    tone: str = "neutral"    # "neutral" | "friendly" | "formal" | "persuasive" | "storytelling" | "playful"
    max_words: int = 160


class BrandVoiceRewriter:
    """
    Public pipeline:
      - assembles a single prompt (system + content + tags)
      - injects soft prompt weights implicitly (metadata tag)
      - generates N candidates
      - ranks by distance-to-brand (+ semantic if source provided)
    """

    def __init__(
        self,
        brand_space: BrandStyleSpace,
        soft_prompt: Optional[SoftPromptWeights] = None,
        model: Optional[TextModel] = None
    ):
        self.brand_space = brand_space
        self.soft_prompt = soft_prompt
        self.model = model or load_local_model()

    def _soft_prompt_tag(self) -> str:
        if not self.soft_prompt:
            return "[SOFT:0]"
        # Encode a small signature so we can trace 'which soft prompt' was applied
        return f"[SOFT:{self.soft_prompt.tokens}x{self.soft_prompt.dim}]"

    @staticmethod
    def _truncate_words(text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])

    def _build_prompt(self, source_text: str, task: TaskSpec) -> str:
        """
        Minimal text prompt for the public demo.
        No external templates to keep dependencies zero.
        """
        
        source_text = self._truncate_words(source_text.strip(), max_words=task.max_words)
        
        system = textwrap.dedent(f"""
        You are a helpful writing assistant.
        Rewrite the content to improve clarity while preserving meaning.
        Maintain the brand voice encoded by the soft prompt tag and produce a {task.tone} tone.
        Avoid adding new facts. Keep it concise and readable.
        """).strip()

        prompt = (
            f"[SYSTEM]\n{system}\n[/SYSTEM]\n"
            f"{self._soft_prompt_tag()} [TONE:{task.tone}]\n"
            f"[CONTENT]\n{source_text}\n[/CONTENT]"
        )
        return prompt

    def rewrite(
        self,
        source_text: str,
        task: Optional[TaskSpec] = None,
        n_candidates: int = 3,
        alpha_semantic: float = 0.4,
        gen_cfg: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """
        Returns: {
          'best': {'text':..., 'metrics':...},
          'candidates': [{'text':..., 'metrics':...}, ...]
        }
        """
        if task is None:
            task = TaskSpec()
        if gen_cfg is None:
            gen_cfg = GenerationConfig(
                max_tokens=min(512, task.max_words * 2),
                temperature=0.7,
                top_p=0.95,
                num_candidates=n_candidates
            )

        prompt = self._build_prompt(source_text, task)
        gens = self.model.generate_n(prompt, cfg=gen_cfg)

        cands: List[Dict[str, Any]] = []
        for g in gens:
            metrics = evaluate_candidate(
                brand=self.brand_space,
                generated=g.text,
                source=source_text,
                alpha_semantic=alpha_semantic
            )
            cands.append({"text": g.text, "metrics": metrics})

        # pick best by aggregate score
        best = max(cands, key=lambda c: c["metrics"]["score"])
        return {"best": best, "candidates": cands}
