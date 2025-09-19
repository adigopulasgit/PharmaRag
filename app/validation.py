# app/validation.py
import re
from typing import List, Dict, Any, Set

_TAG_RX = re.compile(r"\[([^\]]+)\]")  # matches [tag]

def extract_cited_tags(answer_text: str) -> Set[str]:
    return set(m.group(1).strip() for m in _TAG_RX.finditer(answer_text))

def validate_citations(answer_text: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    cited = extract_cited_tags(answer_text)
    available = set(
        (s.get("meta") or {}).get("id")
        or (s.get("meta") or {}).get("dataset")
        or f"doc{i+1}"
        for i, s in enumerate(sources)
    )
    missing = {t for t in cited if t not in available}
    ok = len(cited) > 0 and len(missing) == 0
    return {"ok": ok, "cited": cited, "available": available, "missing": missing}
