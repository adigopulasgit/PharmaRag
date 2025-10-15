"""
RAGOps Logger — robust CSV logger for PharmaRAG.
Creates data/ragops_eval.csv automatically and appends one row per query.
Safe to call from Streamlit after each answer.
"""

import os, csv, datetime
from typing import Dict, Any

DATA_DIR = "data"
LOG_PATH = os.path.join(DATA_DIR, "ragops_eval.csv")

HEADER = [
    "timestamp",        # ISO8601
    "question",         # user prompt
    "grounded",         # True/False
    "datasets",         # "PubChem;BindingDB"
    "latency_ms",       # int
    "llm_used",         # model name
    "answer_length",    # len(answer)
    "tokens_used",      # optional (0 if unknown)
    "evidence_count",   # len(evidence)
]

def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def _ensure_file():
    _ensure_dirs()
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            writer.writeheader()

def log_query(question: str, result: Dict[str, Any], answer: str, llm_used: str, tokens_used: int = 0):
    """
    Append a single log row. Swallows exceptions so UI never breaks.
    """
    try:
        _ensure_file()
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "grounded": bool(result.get("grounded", False)),
            "datasets": ";".join(result.get("datasets_used", [])),
            "latency_ms": int(result.get("latency_ms", 0)),
            "llm_used": llm_used or "",
            "answer_length": len(answer or ""),
            "tokens_used": int(tokens_used or 0),
            "evidence_count": len(result.get("evidence", []) or []),
        }
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            writer.writerow(row)
    except Exception as e:
        print(f"⚠️ RAGOps logging error: {e}")
