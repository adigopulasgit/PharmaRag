import os, sys, re
import streamlit as st

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Path setup ---
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
for p in (CURRENT_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.append(p)

# --- App imports ---
from app.retriever import retrieve
from app.llm_service import rag_answer, stream_answer_chunks
from app.monitor.logger import log_query   # ✅ Phase 6 logger

# --- Optional RDKit for 2D rendering ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKit_OK = True
except Exception:
    RDKit_OK = False

# --- Streamlit config ---
st.set_page_config(page_title="PharmaRAG Chat", page_icon="🧬", layout="wide")
st.markdown(
    "<h2 style='text-align:center;'>🧪 PharmaRAG — RAGOps Drug Discovery Assistant</h2>",
    unsafe_allow_html=True
)
st.caption("Live answers grounded on BindingDB, PubChem, ChEMBL and TDC summaries. Wikipedia for general topics.")

# ---------- Sidebar ----------
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

# ---------- Helpers ----------
def _chips(dsets):
    return " ".join(f"`{d}`" for d in dsets if d)

def _followups(q: str, datasets: list) -> list:
    ql = q.lower()
    if "herg" in ql:
        return ["Give 5 ligand SMILES for hERG", "Explain hERG_central tasks", "Known hERG blockers?"]
    if "tox21" in ql:
        return ["List the 12 Tox21 tasks", "How to split Tox21 (random vs scaffold)?", "Typical baselines?"]
    if "egfr" in ql:
        return ["Show EGFR ligands with IC50", "What is CHEMBL203?", "EGFR resistance mutations?"]
    if any(x in datasets for x in ["BindingDB", "PubChem", "ChEMBL", "TDC"]):
        return ["Show more compounds", "Any related targets?", "Downloadable sources?"]
    return ["What datasets cover this topic?", "Any assay evidence available?", "Show related compounds"]

def _extract_smiles_lines(evidence):
    """Extract up to 4 SMILES strings from evidence text."""
    pairs = []
    for e in evidence:
        txt = (e.get("text") or "")
        m = re.search(r"SMILES=([A-Za-z0-9@+\-\[\]\(\)\\/=#$]+)", txt)
        if m:
            label = (e.get("meta") or {}).get("dataset", "") or "SMILES"
            pairs.append((label, m.group(1)))
    return pairs[:4]

def _render_smiles_gallery(smiles_pairs):
    """Render 2D molecules if RDKit available."""
    if not RDKit_OK or not smiles_pairs:
        return
    cols = st.columns(min(4, len(smiles_pairs)))
    for i, (label, smi) in enumerate(smiles_pairs):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                img = Draw.MolToImage(mol, size=(220, 220))
                with cols[i % len(cols)]:
                    st.image(img, caption=f"{label}: {smi}", use_column_width=False)
        except Exception:
            pass

# ---------- Render chat history ----------
for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

prompt = st.chat_input(
    "Ask a pharma question (e.g., 'SMILES from BindingDB for EGFR', 'aspirin properties', 'hERG dataset')…"
)

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # --- Retrieve evidence ---
        with st.spinner("Retrieving evidence from web…"):
            preds = retrieve(prompt, k=10, use_web=True, fallback_to_llm=allow_fallback)

        grounded = preds.get("grounded", False)
        dsets = preds.get("datasets_used", [])
        lat = preds.get("latency_ms", 0)
        ev = preds.get("evidence", [])

        if grounded:
            st.success(f"✅ Grounded on { _chips(dsets) }  •  {lat} ms")
        else:
            st.warning("⚠️ LLM-only (no retrieved evidence or out of domain).")

        # --- Generate answer ---
        with st.spinner("Generating answer…"):
            if stream:
                holder, acc = st.empty(), ""
                for chunk in stream_answer_chunks(prompt, ev, model=model, allow_fallback=allow_fallback):
                    acc += chunk
                    holder.markdown(acc)
                answer = acc
                out = {"answer": acc}
            else:
                out = rag_answer(prompt, ev, model=model, stream=False, allow_fallback=allow_fallback)
                answer = out["answer"]

        st.markdown(answer)

        # --- Logging (Phase 6) ---
        tokens_used = 0
        try:
            tokens_used = int(out.get("tokens_used", 0))
            if tokens_used == 0 and answer:
                tokens_used = max(1, len(answer) // 4)  # rough estimate
        except Exception:
            pass

        try:
            log_query(prompt, preds, answer, model, tokens_used=tokens_used)
        except Exception as e:
            print(f"⚠️ Logging failed: {e}")

        # --- Render SMILES structures ---
        smiles_pairs = _extract_smiles_lines(ev)
        if smiles_pairs:
            st.markdown("**Molecule structures (from SMILES):**")
            _render_smiles_gallery(smiles_pairs)

        # --- Show supporting evidence ---
        if grounded and ev:
            st.markdown("**Supporting evidence (top):**")
            shown = set()
            for i, e in enumerate(ev, 1):
                txt = (e.get("text") or "").strip().replace("\n", " ")
                if not txt or txt in shown:
                    continue
                shown.add(txt)
                src = (e.get("meta") or {}).get("dataset","")
                st.markdown(f"- **[{src}]** {txt[:200]}{'…' if len(txt)>200 else ''}")
                if len(shown) >= 3:
                    break

        # --- Suggested follow-ups ---
        suggestions = _followups(prompt, dsets)
        st.markdown("**Try next:** " + " • ".join(f"`{s}`" for s in suggestions))

        # --- Save conversation thread ---
        st.session_state.messages.append({"role": "assistant", "content": answer})
        if st.session_state.active_thread is None:
            st.session_state.threads.append({
                "title": prompt,
                "messages": st.session_state.messages[:]
            })
            st.session_state.active_thread = len(st.session_state.threads) - 1
        else:
            st.session_state.threads[st.session_state.active_thread]["messages"] = st.session_state.messages[:]

st.markdown("---")
st.caption("RAGOps • Provenance-first retrieval with LLM reasoning.")
