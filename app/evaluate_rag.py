# app/evaluate_rag.py
"""
Batch retrieval evaluation.
Input CSV: app/data/eval_qrels.csv with columns:
  query_id,query,relevant_ids
Where relevant_ids is a ; separated list of source tags/ids (e.g., "assay_1;assay_9;doc3").

Usage:
  python -m app.evaluate_rag
Outputs:
  app/logs/retrieval_eval.csv  (per-query results)
  summary printed to console
"""
import os, pandas as pd, numpy as np
from typing import List

from app.retrieval import get_preds_dict

CUR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CUR, "data")
LOG_DIR  = os.path.join(CUR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

EVAL_PATH = os.path.join(DATA_DIR, "eval_qrels.csv")
OUT_PATH  = os.path.join(LOG_DIR, "retrl_eval.csv")

def precision_at_k(pred_ids: List[str], rel_ids: List[str], k: int = 5) -> float:
    top = pred_ids[:k]
    return len(set(top) & set(rel_ids)) / max(1, len(top))

def reciprocal_rank(pred_ids: List[str], rel_ids: List[str]) -> float:
    rel = set(rel_ids)
    for i, pid in enumerate(pred_ids, start=1):
        if pid in rel:
            return 1.0 / i
    return 0.0

def _id_of(item: dict, idx: int) -> str:
    meta = item.get("meta") or {}
    return meta.get("id") or meta.get("dataset") or f"doc{idx+1}"

def run(topk: int = 10, use_hybrid: bool = True, use_cross_encoder: bool = False):
    if not os.path.exists(EVAL_PATH):
        # create a tiny starter file if missing
        os.makedirs(DATA_DIR, exist_ok=True)
        pd.DataFrame([
            {"query_id":"q1","query":"PIK3CG IC50 inhibitors","relevant_ids":"doc2"},
            {"query_id":"q2","query":"hERG inhibition evidence","relevant_ids":"doc1"}
        ]).to_csv(EVAL_PATH, index=False)
        print(f"[init] Wrote starter {EVAL_PATH}")

    df = pd.read_csv(EVAL_PATH)
    rows = []
    for _, r in df.iterrows():
        qid = str(r["query_id"])
        q   = str(r["query"])
        rel = [x.strip() for x in str(r["relevant_ids"]).split(";") if x.strip()]

        hits = get_preds_dict(q, k=topk, use_hybrid=use_hybrid, use_cross_encoder=use_cross_encoder)
        pred_ids = [_id_of(h, i) for i, h in enumerate(hits)]

        p5  = precision_at_k(pred_ids, rel, k=min(5, topk))
        p10 = precision_at_k(pred_ids, rel, k=min(10, topk))
        mrr = reciprocal_rank(pred_ids, rel)

        rows.append({
            "query_id": qid,
            "query": q,
            "relevant_ids": ";".join(rel),
            "pred_ids": ";".join(pred_ids),
            "P@5": p5,
            "P@10": p10,
            "MRR": mrr
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"[ok] wrote per-query results -> {OUT_PATH}")

    # summary
    print("\n=== SUMMARY ===")
    for metric in ["P@5","P@10","MRR"]:
        print(f"{metric}: {out[metric].mean():.3f}")

if __name__ == "__main__":
    run()
