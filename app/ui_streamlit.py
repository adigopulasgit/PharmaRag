# app/ui_streamlit.py
import os, sys, time
from datetime import datetime
import pandas as pd
import streamlit as st

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
for p in (CURRENT_DIR, ROOT_DIR):
    if p not in sys.path: sys.path.append(p)

from app.llm_service import rag_answer, stream_answer_chunks
from app.retrieval import get_preds_dict
from app.validation import validate_citations
from app.reporting import build_markdown_report

st.set_page_config(page_title="PharmaRAG — RAGOps", page_icon="🧪", layout="wide")
st.title("PharmaRAG — RAGOps Research Assistant")

# --- robust log reader (handles schema changes) ---
def safe_read_logs(path: str) -> pd.DataFrame:
    cols13 = [
        "timestamp_iso","question","answer","k","latency_ms",
        "prompt_eval_count","eval_count","prompt_eval_duration_ms","eval_duration_ms",
        "groundedness","source_count","source_preview","used_fallback"
    ]
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        n = len(df.columns)
        if n == 13:
            df.columns = cols13; return df
        if n == 12:
            df.columns = cols13[:-1]; df["used_fallback"] = 0; return df
        df = pd.read_csv(path, engine="python", on_bad_lines="skip", header=None, names=cols13)
        if len(df) and df.iloc[0].tolist() == cols13: df = df.iloc[1:]
        return df
    except Exception:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip", header=None, names=cols13)
        if len(df) and list(df.iloc[0]) == cols13: df = df.iloc[1:]
        return df

# --- Session history ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar ---
with st.sidebar:
    st.subheader("Retrieval")
    use_hybrid = st.toggle("Hybrid (dense + BM25 + RRF)", value=True)
    use_rerank = st.toggle("Cross-encoder re-ranking", value=False)
    k = st.slider("Top-K evidence", 1, 20, 5)

    st.subheader("Generation")
    model = st.text_input("Ollama model", value="llama3")
    stream = st.toggle("Streaming (live typing)", value=False)
    llm_fallback = st.toggle("LLM-only fallback if no hits (⚠️ may hallucinate)", value=False)

    st.markdown("---")
    log_path = os.path.join(CURRENT_DIR, "logs", "ragops_eval.csv")
    if os.path.exists(log_path):
        df_log = safe_read_logs(log_path)
        st.metric("Runs logged", len(df_log))
        if "groundedness" in df_log.columns and len(df_log):
            st.metric("Avg groundedness", f"{df_log['groundedness'].mean():.2f}")
        st.download_button("Download pipeline logs (CSV)", df_log.to_csv(index=False),
                           file_name="ragops_eval.csv")

# --- Helpers ---
def _tag(meta: dict, idx: int) -> str:
    return (meta or {}).get("id") or (meta or {}).get("dataset") or f"doc{idx}"

def sources_to_df(sources):
    rows = []
    for i, s in enumerate(sources, 1):
        rows.append({
            "tag": _tag(s.get("meta"), i),
            "dataset": (s.get("meta") or {}).get("dataset", ""),
            "score": s.get("score", 0.0),
            "text": s.get("text", "")
        })
    return pd.DataFrame(rows)

# --- Tabs ---
tab1, tab2 = st.tabs(["🔎 Retrieval Evidence", "🤖 LLM Insights"])

# TAB 1
with tab1:
    st.subheader("Query → Retrieved Documents")
    q1 = st.text_input("Enter a retrieval query:", key="retrieval_q")
    cA, cB = st.columns([1,1])
    with cA: run_ret = st.button("Search")
    with cB: clear_hist = st.button("Clear Session History")
    if clear_hist:
        st.session_state.history = []; st.success("History cleared.")

    if run_ret and q1.strip():
        with st.spinner("Searching…"):
            preds = get_preds_dict(q1, k=k, use_hybrid=use_hybrid, use_cross_encoder=use_rerank)
        if preds:
            df_src = sources_to_df(preds)
            st.dataframe(df_src, use_container_width=True, hide_index=True)
            st.bar_chart(df_src[["score"]])
            st.download_button("Download results as CSV", df_src.to_csv(index=False),
                               file_name="retrieval_results.csv", mime="text/csv")
            st.session_state.history.append({
                "tab": "Retrieval", "ts": datetime.utcnow().strftime("%H:%M:%S"),
                "question": q1, "answer": "", "sources": [r["meta"] for r in preds],
                "groundedness": None, "used_fallback": False
            })
        else:
            st.warning("No documents found for this query.")

# TAB 2
with tab2:
    st.subheader("Ask grounded questions (viability, toxicity, etc.)")
    q2 = st.text_area("Enter your question:", height=90, key="llm_q",
                      placeholder="e.g., Which compounds show potential hERG inhibition with supporting assay details?")
    c1, c2 = st.columns([1,1])
    with c1: run_llm = st.button("Generate Answer")
    with c2: report_ph = st.empty()

    if run_llm and q2.strip():
        with st.spinner("Retrieving evidence…"):
            preds = get_preds_dict(q2, k=k, use_hybrid=use_hybrid, use_cross_encoder=use_rerank)

        if stream:
            st.markdown("### Answer")
            ph = st.empty(); acc = ""; t0 = time.perf_counter()
            for chunk in stream_answer_chunks(q2, preds, k=k, model=model, allow_fallback=llm_fallback):
                acc += chunk; ph.markdown(acc)
            res = rag_answer(q2, preds, k=k, model=model, stream=False, allow_fallback=llm_fallback)
        else:
            with st.spinner("Generating grounded answer…"):
                res = rag_answer(q2, preds, k=k, model=model, stream=False, allow_fallback=llm_fallback)
            st.markdown("### Answer")
            if res.get("used_fallback"):
                st.warning("⚠️ LLM-only fallback used (no retrieved evidence). Not grounded.")
            st.markdown(res["answer"])

        # Guardrail
        if not res.get("used_fallback"):
            v = validate_citations(res["answer"], res["sources"])
            if v["ok"]:
                st.success("✅ Citations match retrieved sources.")
            else:
                st.error("❌ Citation check failed — tags in the answer aren’t in retrieved sources.")
                st.caption(f"Cited: {sorted(v['cited'])} | Available: {sorted(v['available'])} | Missing: {sorted(v['missing'])}")

        # Evidence + metrics
        with st.expander("Supporting Evidence (Top-K)"):
            if res["sources"]:
                df_src = sources_to_df(res["sources"])
                st.dataframe(df_src, use_container_width=True, hide_index=True)
            else:
                st.write("_No sources shown for this response._")

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Latency (ms)", f'{res["latency_ms"]}')
        cB.metric("Prompt tokens", f'{res["metrics"].get("prompt_eval_count", 0)}')
        cC.metric("Gen tokens", f'{res["metrics"].get("eval_count", 0)}')
        cD.metric("Groundedness", f'{res["groundedness"]:.2f}')

        # Per-query report
        report_bytes = build_markdown_report(
            question=q2, answer=res["answer"], sources=res["sources"],
            metrics={**res["metrics"], "latency_ms": res["latency_ms"]},
            groundedness=res["groundedness"], used_fallback=bool(res.get("used_fallback"))
        )
        report_ph.download_button("⬇️ Download report (Markdown)", data=report_bytes,
                                  file_name="pharmarag_report.md", mime="text/markdown")

        # History
        st.session_state.history.append({
            "tab": "LLM", "ts": datetime.utcnow().strftime("%H:%M:%S"),
            "question": q2, "answer": res["answer"],
            "sources": [s.get("meta") for s in res["sources"]],
            "groundedness": res["groundedness"],
            "used_fallback": bool(res.get("used_fallback"))
        })

st.markdown("---")
st.caption("PharmaRAG • Phase 4: Hybrid retrieval, reranking, citation guardrails, reports.")
