# app/ui_streamlit.py
import os, sys
import streamlit as st

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
for p in (CURRENT_DIR, ROOT_DIR):
    if p not in sys.path: sys.path.append(p)

from app.retriever import retrieve
from app.llm_service import rag_answer, stream_answer_chunks

st.set_page_config(page_title="PharmaRAG Chat", page_icon="🧬", layout="wide")
st.markdown("<h2 style='text-align:center;'>🧪 PharmaRAG — RAGOps Drug Discovery Assistant</h2>", unsafe_allow_html=True)
st.caption("Live answers grounded on BindingDB, PubChem, ChEMBL and TDC summaries. Wikipedia for general topics.")

# --- Sidebar (settings + history) ---
with st.sidebar:
    st.subheader("⚙️ Settings")
    model = st.text_input("LLM Model", value="llama3")
    stream = st.toggle("Stream response", value=False)
    allow_fallback = st.toggle("Allow LLM fallback when no evidence", value=True)
    st.markdown("---")
    st.subheader("💬 Chat History")
    if "threads" not in st.session_state: st.session_state.threads = []
    if "active_thread" not in st.session_state: st.session_state.active_thread = None
    if "messages" not in st.session_state: st.session_state.messages = []
    if st.session_state.threads:
        for i, th in enumerate(st.session_state.threads):
            if st.button(f"Chat {i+1}: {th['title'][:24]}", key=f"th_{i}"):
                st.session_state.active_thread = i
                st.session_state.messages = th["messages"][:]
    if st.button("➕ New Chat"):
        st.session_state.active_thread = None
        st.session_state.messages = []

# render chat history
for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

prompt = st.chat_input("Ask a pharma question (e.g., 'SMILES from BindingDB for EGFR', 'aspirin properties', 'hERG dataset')…")

def _chips(dsets): return " ".join(f"`{d}`" for d in dsets if d)

def _followups(q: str, datasets: list) -> list:
    ql = q.lower()
    if "herg" in ql:
        return ["Give 5 ligand SMILES for hERG", "Explain hERG_central tasks", "Known hERG blockers?"]
    if "tox21" in ql:
        return ["List the 12 Tox21 tasks", "How to split Tox21 (random vs scaffold)?", "Typical baselines?"]
    if "egfr" in ql:
        return ["Show EGFR ligands with IC50", "What is CHEMBL203?", "EGFR resistance mutations?"]
    if any(x in datasets for x in ["BindingDB","PubChem","ChEMBL","TDC"]):
        return ["Show more compounds", "Any related targets?", "Downloadable sources?"]
    return ["What datasets cover this topic?", "Any assay evidence available?", "Show related compounds"]

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence from web…"):
            preds = retrieve(prompt, k=10, use_web=True, fallback_to_llm=allow_fallback)

        grounded = preds.get("grounded", False)
        dsets = preds.get("datasets_used", [])
        lat = preds.get("latency_ms", 0)
        ev = preds.get("evidence", [])

        if grounded:
            st.success(f"✅ Grounded on { _chips(dsets) }  •  {lat} ms")
        else:
            st.warning("⚠️ LLM-only (no retrieved evidence or out of pharma domain).")

        # Generate the answer
        with st.spinner("Generating answer…"):
            if stream:
                holder = st.empty(); acc = ""
                for chunk in stream_answer_chunks(prompt, ev, model=model, allow_fallback=allow_fallback):
                    acc += chunk; holder.markdown(acc)
                answer = acc
            else:
                out = rag_answer(prompt, ev, model=model, stream=False, allow_fallback=allow_fallback)
                answer = out["answer"]

        st.markdown(answer)

        # Show up to 3 concise evidence bullets (no expander)
        if ev:
            st.markdown("**Supporting evidence (top):**")
            for i, e in enumerate(ev[:3], 1):
                src = (e.get("meta") or {}).get("dataset","")
                txt = (e.get("text") or "").replace("\n"," ")
                st.markdown(f"- **[{src}]** {txt[:200]}{'…' if len(txt)>200 else ''}")

        # Follow-ups
        suggestions = _followups(prompt, dsets)
        st.markdown("**Try next:** " + " • ".join(f"`{s}`" for s in suggestions))

        # Save conversation
        st.session_state.messages.append({"role":"assistant","content":answer})
        if st.session_state.active_thread is None:
            st.session_state.threads.append({"title": prompt, "messages": st.session_state.messages[:]})
            st.session_state.active_thread = len(st.session_state.threads)-1
        else:
            st.session_state.threads[st.session_state.active_thread]["messages"] = st.session_state.messages[:]

st.markdown("---")
st.caption("RAGOps • Provenance-first retrieval with LLM reasoning.")
