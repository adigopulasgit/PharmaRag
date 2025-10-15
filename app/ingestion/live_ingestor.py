"""
Live Ingestor (Universal, Async) for PharmaRAG
- PubChem (name + fastsearch/description)
- ChEMBL fallback for targets
- Optional TDC summaries by keyword
- Returns unified evidence schema; cached for 24h
"""
import asyncio, httpx, re
from typing import List, Dict
from app.ingestion.cache_manager import get_cache, set_cache
from app.ingestion import web_ingestor as wi

TIMEOUT = httpx.Timeout(15.0, connect=10.0)
LIMIT = 6  # concurrent connections

# ---------- PubChem flexible search ----------
async def _pubchem_search(client, query: str) -> List[Dict]:
    urls = [
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/property/CanonicalSMILES,IsomericSMILES,InChIKey,MolecularFormula,MolecularWeight/JSON",
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsearch/description/{query}/property/CanonicalSMILES,InChIKey,MolecularFormula,MolecularWeight/JSON",
    ]
    for url in urls:
        try:
            r = await client.get(url)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            out = []
            for i, p in enumerate(props[:5]):
                smi = p.get("CanonicalSMILES") or p.get("IsomericSMILES", "")
                txt = (f"{query}: MF={p.get('MolecularFormula','')} "
                       f"MW={p.get('MolecularWeight','')} "
                       f"SMILES={smi} InChIKey={p.get('InChIKey','')}")
                out.append({"text": txt, "score": 1.0,
                            "meta": {"dataset": "PubChem", "id": f"pubchem_{query}_{i}"}})
            if out: return out
        except Exception as e:
            print(f"⚠️ PubChem search error ({query}): {e}")
    return []

# ---------- ChEMBL fallback (target info) ----------
async def _chembl_fallback(client, query: str) -> List[Dict]:
    try:
        chembl_id = wi.find_chembl_id(query)
        if chembl_id:
            url = f"https://www.ebi.ac.uk/chembl/api/data/target/{chembl_id}.json"
            r = await client.get(url)
            r.raise_for_status()
            d = r.json()
            pref = d.get("pref_name", chembl_id); org = d.get("organism", "")
            return [{"text": f"ChEMBL target {chembl_id}: {pref} ({org})",
                     "score": 1.0, "meta": {"dataset": "ChEMBL", "id": chembl_id}}]
    except Exception as e:
        print(f"⚠️ ChEMBL fallback error: {e}")
    return []

# ---------- TDC summaries by keyword ----------
async def _tdc_if_requested(query: str) -> List[Dict]:
    items = []
    for key in ["herg","tox21","caco2","solubility","lipophilicity"]:
        if key in query.lower():
            items += wi.query_tdc_summary(key)
    return items

# ---------- Extract possible chemical tokens ----------
def _extract_possible_compounds(q: str) -> List[str]:
    words = re.findall(r"[A-Za-z0-9\-\+\(\)]+", q)
    toks = [w for w in words if w.lower() not in {"smiles","from","pubchem","give","me","of","for","please"}]
    toks = [w for w in toks if len(w) > 1]
    return list(dict.fromkeys(toks))[:5] or [q.strip()]

# ---------- Async gather ----------
async def _gather_results(query: str, max_items: int = 25) -> List[Dict]:
    compounds = _extract_possible_compounds(query)
    tasks = []
    async with httpx.AsyncClient(timeout=TIMEOUT, limits=httpx.Limits(max_connections=LIMIT)) as client:
        for c in compounds:
            tasks.append(_pubchem_search(client, c))
        tasks.append(_chembl_fallback(client, query))
        tasks.append(_tdc_if_requested(query))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    ev = []
    for r in results:
        if isinstance(r, list): ev.extend(r)

    seen, out = set(), []
    for e in ev:
        t = e.get("text","")
        if t and t not in seen:
            seen.add(t); out.append(e)
        if len(out) >= max_items: break
    return out

# ---------- Public entry ----------
def get_live_results(query: str, max_items: int = 25) -> List[Dict]:
    cached = get_cache(query)
    if cached:
        print(f"⚡ Cache hit for '{query}'")
        return cached
    print(f"🌐 Live fetch for '{query}' …")
    results = asyncio.run(_gather_results(query, max_items))
    set_cache(query, results)
    return results
