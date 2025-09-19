# retriever.py
import os
import faiss
import pandas as pd
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# -------------------------
# Config (set env vars as needed)
# -------------------------
EMB_MODEL = os.getenv("RAGOPS_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH = os.getenv("RAGOPS_INDEX_PATH", "artifacts/faiss.index")
DOCS_PATH = os.getenv("RAGOPS_DOCS_PATH", "artifacts/docs.parquet")
TOP_K = int(os.getenv("RAGOPS_TOPK", "5"))

# -------------------------
# Load model, index, and docs
# -------------------------
_model = SentenceTransformer(EMB_MODEL)

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")

if not os.path.exists(DOCS_PATH):
    raise FileNotFoundError(f"Docs file not found at {DOCS_PATH}")

_index = faiss.read_index(INDEX_PATH)
_docs = pd.read_parquet(DOCS_PATH)  # expects columns: doc_id, text, source


def embed(texts: List[str]) -> np.ndarray:
    """Create embeddings for a list of texts."""
    return _model.encode(texts, normalize_embeddings=True).astype("float32")


def retrieve(query: str, k: int = TOP_K) -> List[Dict]:
    """Retrieve top-k docs relevant to the query."""
    qv = embed([query])
    D, I = _index.search(qv, k)

    hits = []
    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
        row = _docs.iloc[idx]
        hits.append({
            "rank": rank,
            "doc_id": row.get("doc_id", str(idx)),
            "text": row.get("text", ""),
            "source": row.get("source", ""),
            "score": float(dist)
        })
    return hits
