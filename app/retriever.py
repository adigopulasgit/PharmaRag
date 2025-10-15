"""
Unified Retriever for PharmaRAG (RAGOps)
----------------------------------------
Dynamic universal retriever — integrates live web ingestion
and cached lookups for PubChem, BindingDB, ChEMBL, and TDC.
"""
import time, re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from app.ingestion.web_ingestor import (
    find_uniprot_id,
    find_chembl_id,
    query_bindingdb_by_uniprot,
    query_chembl_target,
    query_pubchem_by_name,
    query_tdc_summary,
)

try:
    from app.ingestion.live_ingestor import get_live_results
    USE_LIVE = True
    print("✅ Live Ingestor enabled (async + cache).")
except ImportError:
    USE_LIVE = False
    print("⚠️ Live Ingestor not found — using threaded web mode.")

TDC_KEYS = ["herg", "tox21", "caco2", "solubility", "lipophilicity"]

def pick_drug_names(q: str) -> List[str]:
    base = ["aspirin","ibuprofen","paracetamol","gefitinib","erlotinib",
            "dasatinib","sunitinib","nilotinib","acetaminophen","caffeine",
            "dopamine","serotonin","glucose","water","ethanol"]
    found = [d for d in base if d in q.lower()]
    if not found:
        caps = re.findall(r"\b[A-Z][a-z]{2,}\b", q)
        found = [c for c in caps if c.lower() not in {
            "what","give","show","please","provide","explain","about","from","bindingdb","smiles","pubchem"
        }]
    formulas = re.findall(r"\b[A-Z][a-z]?[0-9]{0,3}\b", q)
    all_names = list(dict.fromkeys(found + formulas))
    return all_names[:5]

def _web_retrieve(question: str, max_items: int = 25) -> List[Dict[str, Any]]:
    q = question.strip()
    tasks = []
    with ThreadPoolExecutor() as ex:
        for key in TDC_KEYS:
            if key in q.lower():
                tasks.append(ex.submit(query_tdc_summary, key))
        uniprot_id = find_uniprot_id(q)
        if uniprot_id:
            tasks.append(ex.submit(query_bindingdb_by_uniprot, uniprot_id, 10))
        chembl_id = find_chembl_id(q)
        if chembl_id:
            tasks.append(ex.submit(query_chembl_target, chembl_id))
        for name in pick_drug_names(q):
            tasks.append(ex.submit(query_pubchem_by_name, name, 10))
        if "smiles" in q.lower() and not pick_drug_names(q):
            tasks.append(ex.submit(query_pubchem_by_name, q, 5))

        ev: List[Dict[str, Any]] = []
        try:
            for fut in as_completed(tasks, timeout=60):
                try:
                    res = fut.result(timeout=30)
                    if res: ev.extend(res)
                except Exception as e:
                    print(f"⚠️ Web retrieve error: {e}")
        except TimeoutError:
            print("⚠️ Some API calls took too long — partial results returned.")

    seen, out = set(), []
    for e in ev:
        t = e.get("text", "")
        if t and t not in seen:
            seen.add(t); out.append(e)
        if len(out) >= max_items: break
    return out

def retrieve(question: str, k: int = 10, use_web: bool = True, fallback_to_llm: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    q = question.strip()

    if use_web:
        if USE_LIVE:
            try:
                evidence = get_live_results(q, max_items=25)
                debug_msg = "cache/live"
            except Exception as e:
                print(f"⚠️ Live retriever failed, fallback to threaded mode: {e}")
                evidence = _web_retrieve(q)
                debug_msg = "fallback_threaded"
        else:
            evidence = _web_retrieve(q)
            debug_msg = "threaded_only"
    else:
        evidence, debug_msg = [], "disabled"

    datasets_used = sorted(set((e.get("meta") or {}).get("dataset","") for e in evidence if e.get("meta")))
    grounded = bool(evidence)
    latency = int((time.perf_counter() - t0) * 1000)

    if not grounded and fallback_to_llm:
        print("⚠️ No grounded evidence found — using LLM fallback.")

    return {
        "grounded": grounded,
        "datasets_used": datasets_used,
        "evidence": evidence[:k],
        "latency_ms": latency,
        "debug_msg": debug_msg,
    }
