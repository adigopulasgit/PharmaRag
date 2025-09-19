# app/llm_service.py
import os
import csv
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple

import requests

# -------- Paths & Logging --------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_PATH = os.path.join(LOG_DIR, "ragops_eval.csv")
os.makedirs(LOG_DIR, exist_ok=True)

# -------- Simple token normalization for groundedness proxy --------
STOPWORDS = set("""
a an and are as at be by for from has have if in into is it its of on or that the to was were will with this those these your you we our us not found
""".split())

def _normalize_tokens(text: str) -> set:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s._-]", " ", text)
    toks = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return set(toks)

def groundedness_score(answer_text: str, sources_text: str) -> float:
    """
    Cheap proxy: % of answer tokens that also appear in concatenated sources.
    0.0 (low grounding) .. 1.0 (high grounding).
    """
    a = _normalize_tokens(answer_text)
    s = _normalize_tokens(sources_text)
    if not a:
        return 0.0
    return len(a & s) / max(1, len(a))

def _ensure_csv_header() -> None:
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp_iso",
                "question",
                "answer",
                "k",
                "latency_ms",
                "prompt_eval_count",
                "eval_count",
                "prompt_eval_duration_ms",
                "eval_duration_ms",
                "groundedness",
                "source_count",
                "source_preview",
                "used_fallback"
            ])

def _log_eval(row: List[Any]) -> None:
    _ensure_csv_header()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)

# -------- Retrieval flattening --------
def _flatten_topk(preds_dict: Any, k: int = 5) -> List[Dict[str, Any]]:
    """
    Accepts your retrieval output and returns a flat list of
    {'text': str, 'score': float, 'meta': dict}.
    Supported shapes:
      - list[dict{text, score?, meta?}]
      - dict[str, list[dict{text, score?, meta?}]]
    """
    flat: List[Dict[str, Any]] = []
    if isinstance(preds_dict, list):
        for d in preds_dict:
            if not isinstance(d, dict):
                continue
            flat.append({
                "text": d.get("text") or d.get("chunk") or str(d),
                "score": float(d.get("score", 0.0)),
                "meta": d.get("meta", {})
            })
    elif isinstance(preds_dict, dict):
        for _, lst in preds_dict.items():
            if not isinstance(lst, list):
                continue
            for d in lst:
                if not isinstance(d, dict):
                    continue
                flat.append({
                    "text": d.get("text") or d.get("chunk") or str(d),
                    "score": float(d.get("score", 0.0)),
                    "meta": d.get("meta", {})
                })
    # Take top-k by score if present; otherwise keep order
    flat.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return flat[:k]

# -------- Prompting --------
PROMPT_SYSTEM = """You are a drug discovery assistant.
Answer ONLY using the retrieved evidence. If the evidence is insufficient, say: "Not found in retrieved results."
Always produce two sections exactly:

Answer:
<your short answer here>

Supporting Evidence:
- Bullet 1 (include the [tag] of a source when it supports a claim)
- Bullet 2
"""

PROMPT_FALLBACK_SYSTEM = """You are a drug discovery assistant.
No retrieved evidence is available from our dataset for this question.
Provide a general background answer, but at the VERY TOP include:
**Disclaimer: This answer is NOT grounded in our dataset (LLM-only fallback).**

Produce two sections:

Answer:
<concise background answer; keep practical and cautious>

General Notes:
- Keep claims high-level; avoid unverifiable specifics.
- Encourage the user to refine the query or add data for grounding.
"""

def _build_messages(question: str, topk: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    bullets = []
    for i, d in enumerate(topk, 1):
        meta = d.get("meta") or {}
        tag = meta.get("id") or meta.get("dataset") or f"doc{i}"
        preview = (d["text"][:400] + "…") if len(d["text"]) > 400 else d["text"]
        bullets.append(f"- [{tag}] {preview}")

    context = "Retrieved Evidence:\n" + ("\n".join(bullets) if bullets else "- (none)")

    user_prompt = f"""{context}

Question:
{question}

Instructions:
- Use only the evidence above.
- If an item supports your answer, reference it by its [tag].
- Output must have the exact two sections specified.
"""

    return [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user",   "content": user_prompt},
    ]

def _build_fallback_messages(question: str) -> List[Dict[str, str]]:
    user_prompt = f"""No retrieved evidence was found for this question in our dataset.

Question:
{question}

Instructions:
- Provide a careful, high-level background answer.
- Include the disclaimer at the very top as instructed.
- Use the exact two sections specified (Answer, General Notes).
"""
    return [
        {"role": "system", "content": PROMPT_FALLBACK_SYSTEM},
        {"role": "user",   "content": user_prompt},
    ]

# -------- Ollama Chat --------
def _ollama_chat(messages: List[Dict[str, str]], model: str = "llama3", stream: bool = False
                 ) -> Tuple[str, Dict[str, Any], int]:
    """
    Returns (answer_text, meta_tail, latency_ms)
    meta_tail may include counters/durations if Ollama returns them.
    """
    url = "http://localhost:11434/api/chat"
    payload = {"model": model, "messages": messages}

    if stream:
        resp = requests.post(url, json=payload, stream=True, timeout=300)
        resp.raise_for_status()
        answer_text = ""
        meta_tail: Dict[str, Any] = {}
        start = time.perf_counter()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if "message" in data and "content" in data["message"]:
                answer_text += data["message"]["content"]
            for k in ("total_duration","load_duration","prompt_eval_count","prompt_eval_duration","eval_count","eval_duration"):
                if k in data:
                    meta_tail[k] = data[k]
        latency_ms = int((time.perf_counter() - start) * 1000)
        return answer_text, meta_tail, latency_ms

    # Non-streaming single JSON
    start = time.perf_counter()
    resp = requests.post(url, json={**payload, "stream": False}, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    latency_ms = int((time.perf_counter() - start) * 1000)
    answer_text = data["message"]["content"]
    meta_tail = {
        k: data.get(k) for k in (
            "total_duration","load_duration","prompt_eval_count","prompt_eval_duration","eval_count","eval_duration"
        )
    }
    return answer_text, meta_tail, latency_ms

# -------- Public RAG API --------
def rag_answer(
    question: str,
    preds_dict: Any,
    k: int = 5,
    model: str = "llama3",
    stream: bool = False,
    allow_fallback: bool = False
) -> Dict[str, Any]:
    """
    Returns dict:
      - answer: str
      - sources: list of dicts (text, score, meta)
      - latency_ms: int
      - metrics: dict (prompt/gen token counts & durations if available)
      - groundedness: float
      - sources_text: concatenated string for convenience
      - used_fallback: bool
    """
    topk = _flatten_topk(preds_dict, k=k)
    used_fallback = False

    # Build messages: normal vs fallback
    if not topk and allow_fallback:
        msgs = _build_fallback_messages(question)
        used_fallback = True
    else:
        msgs = _build_messages(question, topk)

    answer_text, meta_tail, latency_ms = _ollama_chat(msgs, model=model, stream=stream)

    sources_text = "\n".join([d["text"] for d in topk]) if topk else ""
    gscore = groundedness_score(answer_text, sources_text) if topk else 0.0
    source_preview = (topk[0]["text"][:160] + "…") if topk else ""

    _log_eval([
        datetime.utcnow().isoformat(),
        question,
        answer_text.replace("\n", " ")[:1000],
        k,
        latency_ms,
        (meta_tail.get("prompt_eval_count") or 0),
        (meta_tail.get("eval_count") or 0),
        int((meta_tail.get("prompt_eval_duration") or 0) / 1e6),
        int((meta_tail.get("eval_duration") or 0) / 1e6),
        round(gscore, 3),
        len(topk),
        source_preview.replace("\n", " "),
        int(used_fallback)
    ])

    return {
        "answer": answer_text,
        "sources": topk,
        "latency_ms": latency_ms,
        "metrics": meta_tail,
        "groundedness": gscore,
        "sources_text": sources_text,
        "used_fallback": used_fallback
    }

# -------- Optional: live streaming generator with fallback --------
def stream_answer_chunks(
    question: str,
    preds_dict: Any,
    k: int = 5,
    model: str = "llama3",
    allow_fallback: bool = False
):
    """
    Yields text chunks for a live-typing UI.
    If no retrieval and allow_fallback=True, streams the fallback message.
    NOTE: This generator does NOT log metrics; call rag_answer(..., stream=False)
    afterward to compute metrics and record the run.
    """
    topk = _flatten_topk(preds_dict, k=k)
    if not topk and allow_fallback:
        msgs = _build_fallback_messages(question)
    else:
        msgs = _build_messages(question, topk)

    url = "http://localhost:11434/api/chat"
    resp = requests.post(url, json={"model": model, "messages": msgs}, stream=True, timeout=300)
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line.decode("utf-8"))
        if "message" in data and "content" in data["message"]:
            yield data["message"]["content"]
