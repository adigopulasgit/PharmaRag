# app/llm/prompt.py

# System prompt for Ollama / LLaMA
SYSTEM_PROMPT = """You are a drug discovery assistant.
- Summarize ADMET predictions clearly.
- Combine with retrieved evidence.
- Highlight risks (toxicity, solubility, permeability issues).
- Cite evidence with [doc_id] when possible.
- Be concise (max 10 sentences).
"""

def build_user_prompt(question: str, admet_block: str, ctx_block: str) -> str:
    """
    Construct the user message that is passed to the LLM.
    """
    return f"""Question:
{question}

{admet_block}

{ctx_block}
"""
 