# app/retrieval.py
"""
Hybrid retrieval (dense + lexical) with RRF merge, optional cross-encoder re-ranking.

Data file: app/data/chunks.jsonl with rows like:
  {"id":"assay_123","dataset":"BindingDB","text":"IC50=85 nM for Q15858 ..."}

Installs:
  pip install sentence-transformers faiss-cpu rank-bm25
"""
import os, json
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")

# ---------- Load corpus ----------
def _load_corpus() -> List[Dict[str, Any]]:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CHUNKS_PATH):
        # minimal placeholder content so app runs
        sample = [
            {"id":"doc1","dataset":"ADMET","text":"Compound A shows potential hERG inhibition in assay X."},
            {"id":"doc2","dataset":"BindingDB","text":"Q15858 (PIK3CG) inhibitors include Compound B with IC50 80 nM."},
            {"id":"doc3","dataset":"ADMET","text":"Lipophilicity indicates moderate risk; solubility favorable for Compound C."}
        ]
        with open(CHUNKS_PATH, "w") as f:
            for r in sample: f.write(json.dumps(r) + "\n")

    corpus = []
    with open(CHUNKS_PATH, "r") as f:
        for line in f:
            try:
                corpus.append(json.loads(line))
            except Exception:
                continue
    return corpus

_CORPUS = _load_corpus()
_TEXTS  = [d.get("text","") for d in _CORPUS]
_IDS    = [d.get("id", f"doc{i}") for i, d in enumerate(_CORPUS, 1)]
_DATA   = [d.get("dataset","") for d in _CORPUS]

# ---------- Dense index ----------
_ST = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_EMB = _ST.encode(_TEXTS, normalize_embeddings=True, convert_to_numpy=True)
_INDEX = faiss.IndexFlatIP(_EMB.shape[1])
_INDEX.add(_EMB)

# ---------- Lexical index ----------
def _tokenize(s: str):
    return s.lower().split()

_BM25 = BM25Okapi([_tokenize(t) for t in _TEXTS])

# ---------- RRF merge ----------
def _rrf(rank_lists: List[List[int]], kappa: int = 60) -> Dict[int, float]:
    """
    rank_lists: list of lists of idx (ordered by rank best->worst)
    returns: idx -> rrf score
    """
    merged = {}
    for lst in rank_lists:
        for rank, idx in enumerate(lst, start=1):
            merged[idx] = merged.get(idx, 0.0) + 1.0 / (kappa + rank)
    return merged

# ---------- Search ----------
def _dense_search(query: str, topn: int = 50) -> List[Tuple[int, float]]:
    qv = _ST.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, idxs = _INDEX.search(qv, topn)
    return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

def _bm25_search(query: str, topn: int = 50) -> List[Tuple[int, float]]:
    scores = _BM25.get_scores(_tokenize(query))
    ranked = np.argsort(scores)[::-1][:topn]
    return [(int(i), float(scores[i])) for i in ranked]

_CE: CrossEncoder = None
def _load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _CE
    if _CE is None:
        _CE = CrossEncoder(model_name)
    return _CE

def _ce_rerank(query: str, idxs: List[int], topm: int = 20) -> List[int]:
    ce = _load_cross_encoder()
    pairs = [(query, _TEXTS[i]) for i in idxs[:topm]]
    scores = ce.predict(pairs).tolist()
    order = np.argsort(scores)[::-1]
    return [idxs[:topm][j] for j in order]

def get_preds_dict(
    question: str,
    k: int = 5,
    use_hybrid: bool = True,
    use_cross_encoder: bool = False,
    dense_topn: int = 50,
    bm25_topn: int = 50
) -> List[Dict[str, Any]]:
    """
    Returns list of {'text','score','meta':{'id','dataset'}} sorted by final score.
    """
    # Dense & lexical
    d_hits = _dense_search(question, dense_topn)
    l_hits = _bm25_search(question, bm25_topn)

    if use_hybrid:
        d_order = [i for i, _ in d_hits]
        l_order = [i for i, _ in l_hits]
        rrf_scores = _rrf([d_order, l_order])
        idxs = [i for i, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)]
    else:
        idxs = [i for i, _ in d_hits]

    # Optional CE re-rank on the top pool
    pool = idxs[:max(k, 20)]
    if use_cross_encoder and pool:
        pool = _ce_rerank(question, pool, topm=max(k, 20))

    # Build results
    dense_map = {i: s for i, s in d_hits}
    bm25_map  = {i: s for i, s in l_hits}
    rrf_map   = _rrf([ [i for i,_ in d_hits], [i for i,_ in l_hits] ]) if use_hybrid else {}

    out = []
    for i in pool[:k]:
        score = rrf_map.get(i, dense_map.get(i, 0.0))
        out.append({
            "text": _TEXTS[i],
            "score": float(score),
            "meta": {"id": _IDS[i], "dataset": _DATA[i]}
        })
    return out
