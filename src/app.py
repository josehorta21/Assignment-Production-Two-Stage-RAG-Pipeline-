"""Streamlit RAG observability workbench."""
from __future__ import annotations

from dataclasses import replace
import streamlit as st
import pandas as pd
from src.config import load_settings
from src.factory import build_pipeline
from src.generator import GroundedGenerator

st.set_page_config(page_title="Northstar | Adaptive RAG Workbench", page_icon="🧭", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1320px; padding-top: 2rem;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.2); border-radius: 12px; padding: 12px;}
.hero {border-radius:18px;padding:20px 26px;background:linear-gradient(120deg,#172437,#25415b);color:white;margin-bottom:18px}
.hero h1 {margin:0; font-size:2rem}.hero p{margin:.3rem 0 0;opacity:.86}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Northstar · Adaptive RAG Workbench</h1><p>Query rewriting · MiniLM vector retrieval · cross-encoder re-ranking · confidence gate · cited web fallback</p></div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading local embedding and cross-encoder models; the first run downloads model files…")
def create_engine():
    return build_pipeline(load_settings())

try:
    engine = create_engine()
except Exception as exc:
    st.error(f"Engine initialization failed: {type(exc).__name__}: {exc}")
    st.info("Run `uv sync` from the project root, ensure model downloads are permitted, and verify the docs/ folder exists.")
    st.stop()

st.sidebar.header("Pipeline controls")
threshold = st.sidebar.slider("Confidence gate threshold", 0.05, 0.95, float(engine.threshold), 0.05)
k1 = st.sidebar.slider("Stage 1 · candidate pool", 5, 30, engine.k1, 1)
k2 = st.sidebar.slider("Stage 2 · final snippets", 3, 5, min(engine.k2, 5), 1)
force_web = st.sidebar.checkbox("Force web route (demo override)", value=False)
mode = st.sidebar.selectbox("Response synthesis", ["extractive", "local"],
                            index=0 if engine.generator.mode == "extractive" else 1,
                            help="Local uses free FLAN-T5; unsupported generated output is discarded.")
st.sidebar.caption("No API keys required. Web results are external, untrusted preview snippets.")
st.sidebar.divider()
st.sidebar.caption("Reference corpus: fictional Northstar engineering handbook (12 Markdown source files).")

st.subheader("Ask a technical question")
samples = {
    "Exact configuration": "What is the maxUnavailable default in Rolling Update Deployment?",
    "Conceptual": "How can we reduce failed requests during rolling deployment?",
    "Web fallback": "What is the latest reported outage affecting AWS today?",
}
chosen = st.selectbox("Example scenarios", ["Write my own", *samples.keys()])
seed = samples.get(chosen, "")
if "history" not in st.session_state:
    st.session_state.history = []
with st.form("ask_form"):
    question = st.text_area("Question", value=seed, height=90,
                            placeholder="e.g. How should we handle read replica lag?")
    use_context = st.checkbox("Use prior user queries as conversation context", value=True)
    submitted = st.form_submit_button("Run two-stage retrieval", type="primary")

if submitted:
    engine.threshold, engine.k1, engine.k2 = threshold, k1, min(k2, k1)
    engine.generator = GroundedGenerator(
        llm=engine.rewriter.llm, mode=mode,
    )
    try:
        with st.spinner("Transforming query → vector retrieval → re-ranking → gated synthesis…"):
            result = engine.run(question, st.session_state.history if use_context else [], force_web=force_web)
        st.session_state.latest_result = result
        st.session_state.history.append(question)
        st.session_state.history = st.session_state.history[-6:]
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Retrieval failed: {type(exc).__name__}: {exc}")

if "latest_result" in st.session_state:
    response = st.session_state.latest_result
    st.divider()
    left, mid, right = st.columns(3)
    left.metric("Execution path", "🟢 INTERNAL_RAG" if response.route_taken == "INTERNAL_RAG" else "🟠 WEB_FALLBACK")
    mid.metric("Top cross-encoder score", f"{response.confidence_score:.3f}")
    right.metric("End-to-end query time", f"{response.latency_ms:.1f} ms")
    st.progress(float(response.confidence_score), text=f"Confidence gate · score {response.confidence_score:.3f} · threshold {response.gate_threshold:.2f}")
    st.info(f"**Routing decision:** {response.gate_reason}")
    st.write("**Original query:**", response.original_query)
    st.write("**Transformed query:**", response.transformed_query)

    a, b = st.columns(2, gap="large")
    with a:
        st.subheader(f"Stage 1 · Bi-encoder (Top {len(response.stage1)})")
        table = [{"Rank": c.stage1_rank, "Document": c.title, "Chunk ID": c.chunk_id,
                  "Cosine similarity": round(c.score, 4)} for c in response.stage1]
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    with b:
        st.subheader(f"Stage 2 · Cross-encoder (Top {len(response.stage2)})")
        table = [{"New rank": c.stage2_rank, "Old rank": c.stage1_rank,
                  "Document": c.title, "Raw logit": round(c.raw_logit or 0, 4),
                  "Sigmoid score": round(c.score, 4)} for c in response.stage2]
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    st.caption("Cosine scores and cross-encoder sigmoid outputs use different scales. Sigmoid is NOT a calibrated probability.")
    st.subheader("Grounded response")
    st.markdown(response.answer)
    if response.warnings:
        for warning in response.warnings:
            st.warning(warning)
    st.subheader("Evidence and citation explorer")
    for n, source in enumerate(response.sources, 1):
        with st.expander(f"[{n}] {source.title or source.chunk_id}", expanded=n == 1):
            st.text(f"Source type: {source.source} | Chunk ID: {source.chunk_id}")
            if source.url:
                st.link_button("Open source page", source.url)
            else:
                st.caption(f"File: docs/{source.metadata.get('file', '?')}")
            st.text(source.content[:1600])
    with st.expander("Structured execution trace (JSON)"):
        st.json(response.model_dump(mode="json"))
    st.download_button("Download trace JSON", data=response.model_dump_json(indent=2),
                       file_name="rag_run_trace.json", mime="application/json")
else:
    st.caption("Choose an example or type your own question to run the live architecture.")
