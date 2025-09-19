# app/reporting.py
from datetime import datetime
from typing import List, Dict, Any

def _tag(meta: dict, idx: int) -> str:
    return (meta or {}).get("id") or (meta or {}).get("dataset") or f"doc{idx}"

def build_markdown_report(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    groundedness: float,
    used_fallback: bool
) -> bytes:
    ts = datetime.utcnow().isoformat()
    lines = []
    lines.append(f"# PharmaRAG Report\n")
    lines.append(f"**Timestamp (UTC):** {ts}")
    lines.append(f"**Question:** {question}\n")
    if used_fallback:
        lines.append("> **Disclaimer:** This answer is **NOT grounded** in your dataset (LLM-only fallback).\n")
    lines.append("## Answer")
    lines.append(answer.strip() or "_(empty)_")
    lines.append("\n## Supporting Evidence")
    if sources:
        for i, s in enumerate(sources, 1):
            tag = _tag(s.get("meta"), i)
            score = s.get("score", 0.0)
            txt = s.get("text", "")
            preview = (txt[:400] + "…") if len(txt) > 400 else txt
            lines.append(f"- **[{tag}]** (score: {score:.3f}) — {preview}")
    else:
        lines.append("_None_")
    lines.append("\n## Metrics")
    lines.append(f"- **Latency (ms):** {metrics.get('latency_ms', 'n/a')}")
    lines.append(f"- **Prompt tokens:** {metrics.get('prompt_eval_count', 0)}")
    lines.append(f"- **Gen tokens:** {metrics.get('eval_count', 0)}")
    lines.append(f"- **Groundedness:** {groundedness:.2f}")
    md = "\n".join(lines) + "\n"
    return md.encode("utf-8")
