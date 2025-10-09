# app/api.py
"""
FastAPI backend for PharmaRAG (RAGOps)
Provides /chat endpoint for chat-like grounded QA.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from app.retriever import retrieve
from app.llm_service import rag_answer

app = FastAPI(title="PharmaRAG API", version="1.0")

class ChatIn(BaseModel):
    message: str
    model: str = "llama3"
    allow_fallback: bool = True

class ChatOut(BaseModel):
    answer: str
    grounded: bool
    latency_ms: int
    datasets_used: List[str]
    evidence_count: int
    evidence: List[Dict[str, Any]]

@app.post("/chat", response_model=ChatOut)
def chat(inp: ChatIn):
    """Handles a chat query and returns grounded answer."""
    retrieved = retrieve(
        inp.message,
        k=10,
        use_hybrid=True,
        use_cross_encoder=False,
        use_web=True,
        fallback_to_llm=inp.allow_fallback,
    )
    evidence = retrieved.get("evidence", [])
    answer_obj = rag_answer(inp.message, evidence, model=inp.model, stream=False, allow_fallback=inp.allow_fallback)
    datasets = sorted(set((e.get("meta") or {}).get("dataset","") for e in evidence if e.get("meta")))
    return ChatOut(
        answer=answer_obj["answer"],
        grounded=retrieved["grounded"],
        latency_ms=retrieved["latency_ms"],
        datasets_used=datasets,
        evidence_count=len(evidence),
        evidence=evidence[:20],
    )
