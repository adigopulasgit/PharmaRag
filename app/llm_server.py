# llm_service.py
import requests
import time
from typing import Dict

from app.retriever.retriever import retrieve  # make sure this exists
from app.llm.context_builders import format_admet, format_corpus_hits
from app.llm.prompt import SYSTEM_PROMPT, build_user_prompt


# Change if you use mistral or another Ollama model
MODEL = "llama3:8b"

def _ollama_chat(messages):
    """
    Send chat messages to Ollama running locally.
    Ollama must be installed & model pulled: ollama pull llama3:8b
    """
    url = "http://localhost:11434/api/chat"
    resp = requests.post(url, json={"model": MODEL, "messages": messages}, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def rag_answer(question: str, admet_preds: Dict[str, float], k: int = 5) -> Dict[str, str]:
    """
    Main entrypoint: given a question + ADMET predictions,
    retrieve docs + call Ollama LLM to generate an answer.
    """
    t0 = time.time()

    # 1. Retrieve top-k docs
    hits = retrieve(question, k=k)

    # 2. Format inputs
    admet_block = format_admet(admet_preds)
    ctx_block = format_corpus_hits(hits)
    user_prompt = build_user_prompt(question, admet_block, ctx_block)

    # 3. Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    # 4. Call Ollama
    answer_text = _ollama_chat(messages)

    # 5. Build response
    latency_ms = int((time.time() - t0) * 1000)
    return {
        "query": question,
        "answer": answer_text,
        "latency_ms": latency_ms,
        "docs_used": ",".join([h["doc_id"] for h in hits]),
        "admet": admet_preds,
        "evidence": hits
    }
