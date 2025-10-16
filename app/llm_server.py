"""
LLM Server — Standalone Tester for PharmaRAG
--------------------------------------------
Use this file to test Ollama model responses directly.
"""

import requests
import time
from typing import Dict, Any
from app.retriever import retrieve

MODEL = "llama3"  # or "llama3:8b" if pulled

def _ollama_chat(messages):
    """Send chat messages to local Ollama server"""
    url = "http://localhost:11434/api/chat"
    resp = requests.post(url, json={"model": MODEL, "messages": messages}, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]

def rag_answer(question: str, k: int = 5, allow_fallback: bool = True) -> Dict[str, Any]:
    """RAG pipeline: retrieve evidence, then call LLM (with fallback)"""
    from app.llm_service import _build_messages, _build_fallback_messages, _flatten_topk

    t0 = time.time()
    preds = retrieve(question, k=k)
    topk = _flatten_topk(preds.get("evidence", []), k=k)
    used_fallback = not bool(topk) and allow_fallback

    messages = _build_fallback_messages(question) if used_fallback else _build_messages(question, topk)
    answer_text = _ollama_chat(messages)

    latency = int((time.time() - t0) * 1000)
    return {
        "query": question,
        "answer": answer_text,
        "grounded": not used_fallback,
        "latency_ms": latency,
        "sources": topk,
    }

if __name__ == "__main__":
    q = input("🔬 Ask a question: ")
    res = rag_answer(q)
    print("\nAnswer:\n", res["answer"])
    print("\nSources:", len(res["sources"]), " • Grounded:", res["grounded"])
    print("Latency:", res["latency_ms"], "ms")
