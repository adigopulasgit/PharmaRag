from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_service import rag_answer


app = FastAPI(title="RAGOps Drug Discovery Chatbot")

class QueryRequest(BaseModel):
    query: str
    k: int = 5

@app.post("/ask")
def ask(request: QueryRequest):
    result = rag_answer(request.query, k=request.k)
    return result
