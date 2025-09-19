from pathlib import Path
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------------------------
# Paths
# ---------------------------
# Use project-root/data/processed
processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

index_path = processed_dir / "evidence_faiss.index"
idmap_path = processed_dir / "id_to_row.pkl"

# ---------------------------
# Load index + metadata
# ---------------------------
if not index_path.exists():
    raise FileNotFoundError(f"FAISS index not found at {index_path}")
if not idmap_path.exists():
    raise FileNotFoundError(f"id_to_row.pkl not found at {idmap_path}")

index = faiss.read_index(str(index_path))
with open(idmap_path, "rb") as f:
    id_to_row = pickle.load(f)

# ---------------------------
# Lazy load models
# ---------------------------
_embedder = None
_summarizer = None

def get_models():
    """Load embedding + summarizer models (only once)."""
    global _embedder, _summarizer
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    if _summarizer is None:
        _summarizer = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device_map="auto"
        )
    return _embedder, _summarizer

# ---------------------------
# Retrieval
# ---------------------------
def retrieve(query: str, k: int = 5):
    """Retrieve top-k hits from FAISS index given a query string."""
    embedder, _ = get_models()
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)

    results = []
    for score, i in zip(scores[0], idxs[0]):
        row = id_to_row[i]
        results.append({
            "ligand": row.get("ligand_name") or row.get("ligand"),
            "smiles": row.get("smiles"),
            "target": row.get("target"),
            "activity_type": row.get("activity_type"),
            "value": row.get("value"),
            "pValue": row.get("pValue"),
            "score": float(score)
        })
    return results

# ---------------------------
# Summarization
# ---------------------------
def summarize(query: str, hits):
    """Summarize retrieved evidence using a T5 model."""
    _, summarizer = get_models()
    if not hits:
        return "No evidence found."

    context = "\n".join(
        f"{h.get('ligand') or h.get('smiles')} | Target={h.get('target')} "
        f"| {h.get('activity_type')}={h.get('value')} | p={h.get('pValue')}"
        for h in hits
    )

    prompt = (
        "You are a scientific assistant. Summarize the evidence below in 3–5 bullets. "
        "Each bullet MUST include a citation with ligand/target/value in parentheses.\n\n"
        f"Question: {query}\nEvidence:\n{context}\n\nSummary:"
    )

    out = summarizer(prompt, max_new_tokens=256, truncation=True)
    return out[0]["generated_text"]

# ---------------------------
# End-to-end RAG pipeline
# ---------------------------
def rag_answer(query: str, k: int = 5):
    """Retrieve evidence + summarize into a structured dict."""
    hits = retrieve(query, k=k)
    summary = summarize(query, hits)
    return {
        "query": query,
        "summary": summary,
        "evidence": hits
    }
