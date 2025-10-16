"""
Unified Retriever for PharmaRAG (RAGOps)
----------------------------------------
Dynamic universal retriever — integrates live web ingestion
and cached lookups for PubChem, BindingDB, ChEMBL, and TDC.
Auto-filters off-domain questions and gracefully falls back to LLM.
"""

import re, time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# Optional ChemDataExtractor for chemical entity recognition
try:
    from chemdataextractor import Document
    USE_CDE = True
    print("✅ ChemDataExtractor enabled.")
except Exception:
    USE_CDE = False
    print("⚠️ ChemDataExtractor not available — regex fallback only.")

from app.ingestion.web_ingestor import (
    find_uniprot_id,
    find_chembl_id,
    query_bindingdb_by_uniprot,
    query_chembl_target,
    query_pubchem_by_name,
    query_tdc_summary,
)

# Optional live ingestor
try:
    from app.ingestion.live_ingestor import get_live_results
    USE_LIVE = True
    print("✅ Live Ingestor enabled (async + cache).")
except Exception:
    USE_LIVE = False
    print("ℹ️ Live Ingestor not found — using threaded web mode.")

TDC_KEYS = ["herg", "tox21", "caco2", "solubility", "lipophilicity"]

# --------- domain detection ----------
_PHARMA_TERMS = {
    "drug","compound","ligand","assay","tox","admet","smiles","ic50","ki","kd","kd50",
    "binding","inhibitor","agonist","antagonist","target","dataset","chembl","pubchem",
    "bindingdb","uniprot","protein","enzyme","receptor","molecule","structure","inchi","inchikey"
}
_SMILES_CHARS = set(list("=#[]()@+-\\/123456789BrClNOPSF"))

def looks_like_pharma(q: str) -> bool:
    ql = q.lower()
    if any(t in ql for t in _PHARMA_TERMS):
        return True
    # has many SMILES-like characters?
    return sum(c in _SMILES_CHARS for c in q) >= 5

# --------- name extraction ----------
_STOPWORDS = {
    "who","is","are","what","where","when","why","how","the","a","an",
    "show","give","please","about","from","for","with","in","on","to",
    "smiles","bindingdb","chembl","pubchem","uniprot","target","protein",
    "dataset","explain","theory","of","and"
}

def pick_drug_names(q: str) -> List[str]:
    """Extract plausible compound names for PubChem."""
    names: List[str] = []
    if USE_CDE:
        try:
            doc = Document(q)
            names = [c.text for c in doc.cems if c.text]
        except Exception:
            names = []
    if not names:
        # very conservative regex: words with letters/spaces/hyphens (no short/common words)
        candidates = re.findall(r"[A-Za-z][A-Za-z \-\(\)\/]{3,}", q)
        names = [
            c.strip()
            for c in candidates
            if c.strip().lower() not in _STOPWORDS
        ]
    # dedupe, keep at most 5
    out: List[str] = []
    seen = set()
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            out.append(n)
        if len(out) >= 5:
            break
    return out

# --------- threaded web retrieval ----------
def _web_retrieve(question: str, max_items: int = 25) -> List[Dict[str, Any]]:
    q = question.strip()
    tasks = []
    with ThreadPoolExecutor() as ex:
        # TDC keywords
        ql = q.lower()
        for key in TDC_KEYS:
            if key in ql:
                tasks.append(ex.submit(query_tdc_summary, key))

        # Target lookups → BindingDB + ChEMBL
        uniprot_id = find_uniprot_id(q)
        if uniprot_id:
            tasks.append(ex.submit(query_bindingdb_by_uniprot, uniprot_id, 10))
        chembl_id = find_chembl_id(q)
        if chembl_id:
            tasks.append(ex.submit(query_chembl_target, chembl_id))

        # PubChem names
        for name in pick_drug_names(q):
            tasks.append(ex.submit(query_pubchem_by_name, name, 10))

        # Explicit “SMILES for X” fallthrough
        if "smiles" in ql and not pick_drug_names(q):
            tasks.append(ex.submit(query_pubchem_by_name, q, 5))

        ev: List[Dict[str, Any]] = []
        try:
            for fut in as_completed(tasks, timeout=60):
                try:
                    res = fut.result(timeout=30)
                    if res:
                        ev.extend(res)
                except Exception as e:
                    print(f"⚠️ Web retrieve error: {e}")
        except TimeoutError:
            print("⚠️ Some API calls timed out — returning partial results.")

    # Deduplicate by text
    seen, out = set(), []
    for e in ev:
        txt = e.get("text", "")
        if txt and txt not in seen:
            seen.add(txt)
            out.append(e)
        if len(out) >= max_items:
            break
    return out

# --------- public retriever ----------
def retrieve(question: str, k: int = 10, use_web: bool = True, fallback_to_llm: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    q = question.strip()

    # If not pharma-related, short-circuit: no evidence (lets UI mark LLM-only).
    if not looks_like_pharma(q):
        return {
            "grounded": False,
            "datasets_used": [],
            "evidence": [],
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "debug_msg": "off_domain",
        }

    if use_web:
        if USE_LIVE:
            try:
                evidence = get_live_results(q, max_items=25)
                dbg = "live_ingestor"
            except Exception as e:
                print(f"⚠️ Live retriever failed -> threaded: {e}")
                evidence = _web_retrieve(q)
                dbg = "threaded_fallback"
        else:
            evidence = _web_retrieve(q)
            dbg = "threaded_only"
    else:
        evidence, dbg = [], "disabled"

    datasets_used = sorted({(e.get("meta") or {}).get("dataset", "") for e in evidence if e.get("meta")})
    grounded = bool(evidence)
    return {
        "grounded": grounded,
        "datasets_used": datasets_used,
        "evidence": evidence[:k],
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "debug_msg": dbg,
    }
