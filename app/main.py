# app/main.py
from __future__ import annotations
import io
import os
import time
from pathlib import Path
from typing import List, Optional

import streamlit as st

from core.style.dataset import build_corpus
from core.style.space import BrandStyleSpace
from core.softprompt.trainer import SoftPromptTrainer, SoftPromptWeights
from core.pipeline.rewriter import BrandVoiceRewriter, TaskSpec
from core.pipeline.eval.report import (
    compute_candidate_table,
    plot_style_distance_hist,
    plot_semantic_similarity_hist,
    render_run_summary_md,
)
from core.tts.gtts_impl import synthesize_tts_mp3


# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
BRAND_DIR = DATA_DIR / "brand"
REPORTS_DIR = DATA_DIR / "reports"
SOFTPROMPT_DIR = ROOT / "core" / "softprompt" / "weights"
for p in [UPLOADS_DIR, BRAND_DIR, REPORTS_DIR, SOFTPROMPT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ---------- helpers ----------
def save_uploaded_files(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Path]:
    saved: List[Path] = []
    for uf in files:
        ext = Path(uf.name).suffix.lower()
        if ext not in {".txt", ".md"}:
            st.warning(f"Skipping unsupported file type: {uf.name}")
            continue
        out = UPLOADS_DIR / uf.name
        out.write_bytes(uf.read())
        saved.append(out)
    return saved


def load_or_fit_brand(space_path: Path, corpus_dir: Path, dim: int = 256) -> BrandStyleSpace:
    if space_path.exists():
        st.success("Loaded existing brand style space.")
        return BrandStyleSpace.from_json(space_path.read_text(encoding="utf-8"))
    st.info("Fitting brand style space from uploaded corpus…")
    chunks = build_corpus(str(corpus_dir), max_tokens=300)
    space = BrandStyleSpace.fit(chunks, dim=dim)
    space_path.write_text(space.to_json(), encoding="utf-8")
    st.success("Brand style space fitted and saved.")
    return space


def load_or_train_softprompt(weights_path: Path, corpus_dir: Path, tokens: int, dim: int) -> SoftPromptWeights:
    if weights_path.exists():
        st.success("Loaded existing soft prompt weights.")
        return SoftPromptTrainer.load(str(weights_path))
    st.info("Training soft prompt from uploaded corpus…")
    chunks = build_corpus(str(corpus_dir), max_tokens=300)
    trainer = SoftPromptTrainer(tokens=tokens, dim=dim, seed=42)
    weights = trainer.fit(chunks)
    SoftPromptTrainer.save(weights, str(weights_path))
    st.success("Soft prompt trained and saved.")
    return weights


# ---------- UI ----------
st.set_page_config(page_title="Brand Voice Rewriter (Public)", layout="wide")
st.title("🖋️ Brand Voice Rewriter — Public Edition")
st.caption("Rewrite your draft in your brand voice. Local, lightweight, and private.")

with st.sidebar:
    st.header("Corpus")
    uploaded = st.file_uploader("Upload .txt / .md files", type=["txt", "md"], accept_multiple_files=True)
    if uploaded:
        saved_paths = save_uploaded_files(uploaded)
        if saved_paths:
            st.success(f"Saved {len(saved_paths)} file(s) to data/uploads/")

    st.header("Model / Training")
    dim = st.number_input("Embedding dimension", min_value=64, max_value=1024, value=256, step=64)
    soft_tokens = st.number_input("Soft prompt tokens", min_value=8, max_value=128, value=40, step=8)
    retrain = st.checkbox("Force retrain soft prompt", value=False)

    st.header("Generation")
    tone = st.selectbox("Tone", ["neutral", "friendly", "formal", "persuasive", "storytelling", "playful"])
    max_words = st.slider("Max words from input to use", min_value=60, max_value=600, value=160, step=20)
    n_candidates = st.slider("Candidates", min_value=1, max_value=6, value=3, step=1)
    alpha_sem = st.slider("Weight: semantic similarity", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    temperature = st.slider("Temperature (heuristic)", min_value=0.0, max_value=1.5, value=0.7, step=0.05)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=123, step=1)

tab1, tab2, tab3, tab4 = st.tabs(["1) Build Brand", "2) Train Soft Prompt", "3) Rewrite", "4) Report"])

brand_space: Optional[BrandStyleSpace] = None
soft_weights: Optional[SoftPromptWeights] = None

space_json_path = BRAND_DIR / "brand_style_space.json"
weights_path = SOFTPROMPT_DIR / "brand_softprompt.json"

with tab1:
    st.subheader("Build Brand Vector")
    st.write("We compute a brand style centroid (μ) and covariance (Σ) from your uploaded corpus.")
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Fit / Load Brand Style Space"):
            if not any(UPLOADS_DIR.iterdir()):
                st.error("Please upload some .txt/.md files in the sidebar.")
            else:
                brand_space = load_or_fit_brand(space_json_path, UPLOADS_DIR, dim=dim)
    with colB:
        if space_json_path.exists():
            st.code(space_json_path.read_text(encoding="utf-8")[:512] + "...\n", language="json")

with tab2:
    st.subheader("Train Soft Prompt")
    st.write("Train a compact soft prompt that represents your brand voice.")
    if st.button("Train / Load Soft Prompt"):
        if not any(UPLOADS_DIR.iterdir()):
            st.error("Please upload corpus files first.")
        else:
            if weights_path.exists() and retrain:
                weights_path.unlink(missing_ok=True)
            soft_weights = load_or_train_softprompt(weights_path, UPLOADS_DIR, tokens=int(soft_tokens), dim=int(dim))

with tab3:
    st.subheader("Rewrite Draft")
    draft = st.text_area("Paste your draft", height=220, placeholder="Paste a paragraph or email here…")

    gen_col1, gen_col2, gen_col3 = st.columns([1, 1, 1])
    with gen_col1:
        run_btn = st.button("Generate")

    if run_btn:
        if not space_json_path.exists() or not weights_path.exists():
            st.error("Please build brand vector and train soft prompt first (tabs 1 & 2).")
        elif not draft.strip():
            st.error("Please paste a draft to rewrite.")
        else:
            brand_space = BrandStyleSpace.from_json(space_json_path.read_text(encoding="utf-8"))
            soft_weights = SoftPromptTrainer.load(str(weights_path))

            task = TaskSpec(tone=tone, max_words=int(max_words))
            pipeline = BrandVoiceRewriter(brand_space=brand_space, soft_prompt=soft_weights)
            start = time.time()
            out = pipeline.rewrite(
                source_text=draft,
                task=task,
                n_candidates=int(n_candidates),
                alpha_semantic=float(alpha_sem),
            )
            elapsed = time.time() - start

            st.success(f"Generated {len(out['candidates'])} candidate(s) in {elapsed:.2f}s")
            table_df = compute_candidate_table(out["candidates"])
            st.dataframe(table_df, use_container_width=True)

            st.markdown("**Best candidate**")
            st.write(out["best"]["text"])

            # Optional TTS
            tts_col1, tts_col2 = st.columns([1, 3])
            with tts_col1:
                if st.button("🔊 Speak (TTS)"):
                    mp3_bytes = synthesize_tts_mp3(out["best"]["text"])
                    if mp3_bytes:
                        st.audio(mp3_bytes, format="audio/mp3")
                    else:
                        st.warning("TTS not available on this environment.")

            # Plots
            st.markdown("---")
            st.markdown("### Charts")
            c1, c2 = st.columns([1, 1])
            with c1:
                fig1 = plot_style_distance_hist(table_df)
                st.pyplot(fig1, use_container_width=True)
            with c2:
                fig2 = plot_semantic_similarity_hist(table_df)
                st.pyplot(fig2, use_container_width=True)

            # Save report
            md = render_run_summary_md(
                table_df=table_df,
                best_text=out["best"]["text"],
                alpha_sem=float(alpha_sem),
            )
            report_name = f"report_{int(time.time())}.md"
            report_path = REPORTS_DIR / report_name
            report_path.write_text(md, encoding="utf-8")
            st.success(f"Saved report: {report_path}")
            st.download_button("Download report (.md)", data=md.encode("utf-8"), file_name=report_name, mime="text/markdown")

with tab4:
    st.subheader("Saved Reports")
    reps = sorted(REPORTS_DIR.glob("*.md"))
    if not reps:
        st.info("No reports yet.")
    else:
        for rp in reps[-10:]:
            st.markdown(f"- {rp.name}")
            with open(rp, "r", encoding="utf-8") as f:
                st.code(f.read()[:1200] + "\n...", language="markdown")
