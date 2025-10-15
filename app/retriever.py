# app/retriever.py
"""
Unified Retriever for PharmaRAG (RAGOps)
Pure web mode — dynamically routes to:
 - UniProt → BindingDB
 - ChEMBL target
 - PubChem compound
 - TDC summaries
Automatically deduplicates and merges results.
"""

import re, time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.ingestion.web_ingestor import (
    find_uniprot_id,
    find_chembl_id,
    query_bindingdb_by_uniprot,
    query_chembl_target,
    query_pubchem_by_name,
    query_tdc_summary,
)
from app.ingestion.live_ingestor import get_live_results
def _web_retrieve(question, max_items=25):
    return get_live_results(question, max_items)


TDC_KEYS = ["herg", "tox21", "caco2", "solubility", "lipophilicity"]
SMILES_CHARS = set(list("=#[]()@+-\\/123456789BrClNOPSF"))

def looks_like_pharma(q: str) -> bool:
    ql = q.lower()
    pharma_words = ["drug","compound","ligand","assay","tox","admet","smiles","ic50","ki","kd",
                    "binding","inhibitor","target","dataset","chembl","pubchem","bindingdb","uniprot"]
    return any(w in ql for w in pharma_words) or sum(c in SMILES_CHARS for c in q) > 5

def pick_drug_names(q: str) -> List[str]:
    base = ["aspirin","ibuprofen","paracetamol","gefitinib","erlotinib","dasatinib","sunitinib","nilotinib","acetaminophen"]
    found = [d for d in base if d in q.lower()]
    if not found:
        caps = re.findall(r"\b[A-Z][a-z]{3,}\b", q)
        found = [c for c in caps if c.lower() not in {"what","give","show","please","provide","explain","about","from","bindingdb"}]
    return list(dict.fromkeys(found))[:3]

def _web_retrieve(question: str, max_items: int = 25) -> List[Dict[str, Any]]:
    q = question.strip()
    tasks = []
    with ThreadPoolExecutor() as ex:
        # 1️⃣ Dataset references
        for key in TDC_KEYS:
            if key in q.lower():
                tasks.append(ex.submit(query_tdc_summary, key))

        # 2️⃣ Try dynamic target lookup
        if any(w in q.lower() for w in ["bindingdb","ligand","target","inhibitor","affinity"]):
            uniprot_id = find_uniprot_id(q)
            if uniprot_id:
                tasks.append(ex.submit(query_bindingdb_by_uniprot, uniprot_id, 10))
            chembl_id = find_chembl_id(q)
            if chembl_id:
                tasks.append(ex.submit(query_chembl_target, chembl_id))

        # 3️⃣ Compound names → PubChem
        for name in pick_drug_names(q):
            tasks.append(ex.submit(query_pubchem_by_name, name, 10))

        # 4️⃣ If explicitly asks for SMILES
        if "smiles" in q.lower() and not pick_drug_names(q):
            tasks.append(ex.submit(query_pubchem_by_name, q, 5))

        ev: List[Dict[str, Any]] = []
        for fut in as_completed(tasks, timeout=25):
            try:
                res = fut.result()
                if res: ev.extend(res)
            except Exception as e:
                print(f"⚠️ Web retrieve error: {e}")

    # Deduplicate by text
    seen, out = set(), []
    for e in ev:
        t = e.get("text","")
        if t and t not in seen:
            seen.add(t)
            out.append(e)
        if len(out) >= max_items:
            break
    return out

def retrieve(question: str, k: int = 10, use_web: bool = True, fallback_to_llm: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    pharma = looks_like_pharma(question)
    evidence = _web_retrieve(question) if (use_web and pharma) else []
    datasets_used = sorted(set((e.get("meta") or {}).get("dataset","") for e in evidence if e.get("meta")))
    grounded = bool(evidence)
    return {
        "grounded": grounded,
        "datasets_used": datasets_used,
        "evidence": evidence[:k],
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }
