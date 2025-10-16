"""
Web Ingestor for PharmaRAG (RAGOps)
-----------------------------------
Unified retrieval from:
 - BindingDB (ligands by UniProt)
 - PubChem (compound properties by name)
 - ChEMBL (target summaries)
 - TDC (dataset summaries)
 - UniProt / ChEMBL ID lookup
"""

import requests
from requests.adapters import HTTPAdapter, Retry
from typing import List, Dict, Optional
from functools import lru_cache

# ---------- resilient session ----------
_session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=retries))
DEFAULT_TIMEOUT = 30


def _wrap(text: str, dataset: str, _id: str = "", score: float = 1.0) -> Dict:
    return {"text": text, "score": float(score), "meta": {"dataset": dataset, "id": _id or dataset}}


# ---------- UniProt lookup ----------
@lru_cache(maxsize=500)
def find_uniprot_id(query: str) -> Optional[str]:
    url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&fields=accession&size=1&format=json"
    try:
        r = _session.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        js = r.json()
        if js.get("results"):
            return js["results"][0]["primaryAccession"]
    except Exception as e:
        print(f"⚠️ UniProt lookup error ({query}): {e}")
    return None


# ---------- ChEMBL lookup ----------
@lru_cache(maxsize=500)
def find_chembl_id(query: str) -> Optional[str]:
    url = f"https://www.ebi.ac.uk/chembl/api/data/target/search.json?q={query}"
    try:
        r = _session.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        results = r.json().get("targets", [])
        if results:
            return results[0]["target_chembl_id"]
    except Exception as e:
        print(f"⚠️ ChEMBL lookup error ({query}): {e}")
    return None


# ---------- BindingDB ----------
def query_bindingdb_by_uniprot(uniprot: str, top_k: int = 10) -> List[Dict]:
    url = f"https://bindingdb.org/rest/getLigandsByUniprot?uniprot={uniprot};100000&response=application/json"
    try:
        r = _session.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json() if r.text.strip() else []
        out = []
        for i, d in enumerate(data[:top_k]):
            smi = d.get("smiles") or d.get("LigandSMILES") or ""
            ki = d.get("Ki") or d.get("IC50") or d.get("Kd") or "NA"
            out.append(_wrap(f"Ligand {i+1}: SMILES={smi or 'NA'} | Affinity={ki}", "BindingDB", f"bindingdb_{uniprot}_{i}"))
        return out
    except Exception as e:
        print(f"⚠️ BindingDB error ({uniprot}): {e}")
        return []
# ... keep the top of the file exactly as you have it ...

import re

# ---------- PubChem ----------
def query_pubchem_by_name(name: str, top_k: int = 10) -> List[Dict]:
    """Query PubChem compound by name → molecular props + SMILES.

    Hard guard so we don't hit PubChem with junk like 'who', 'is', etc.
    """
    clean = name.strip()
    # reject very short, non-chemical strings or common words
    STOP = {
        "who","is","are","what","where","when","why","how","show","give","please",
        "target","bindingdb","chembl","pubchem","smiles","protein","dataset"
    }
    if len(clean) < 4 or clean.lower() in STOP:
        return []
    # reject if it doesn't look like a chemical/common alias (letters, spaces, hyphens)
    if not re.fullmatch(r"[A-Za-z][A-Za-z \-\(\)\/]{2,}", clean):
        return []

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{clean}/property/CanonicalSMILES,IsomericSMILES,InChIKey,"
        "MolecularFormula,MolecularWeight/JSON"
    )
    try:
        r = _session.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        props = r.json().get("PropertyTable", {}).get("Properties", [])
        out = []
        for i, p in enumerate(props[:top_k]):
            smi = p.get("CanonicalSMILES") or p.get("IsomericSMILES", "")
            txt = (
                f"{clean}: MF={p.get('MolecularFormula','')} "
                f"MW={p.get('MolecularWeight','')} "
                f"SMILES={smi} InChIKey={p.get('InChIKey','')}"
            )
            out.append(_wrap(txt, "PubChem", f"pubchem_{clean}_{i}"))
        return out
    except Exception as e:
        print(f"⚠️ PubChem error ({clean}): {e}")
        return []


# ---------- ChEMBL Target ----------
def query_chembl_target(chembl_id: str) -> List[Dict]:
    url = f"https://www.ebi.ac.uk/chembl/api/data/target/{chembl_id}.json"
    try:
        r = _session.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        d = r.json()
        pref = d.get("pref_name", chembl_id)
        org = d.get("organism", "")
        return [_wrap(f"ChEMBL target {chembl_id}: {pref} ({org})", "ChEMBL", f"chembl_{chembl_id}")]
    except Exception as e:
        print(f"⚠️ ChEMBL target error ({chembl_id}): {e}")
        return []


# ---------- TDC Summaries ----------
def query_tdc_summary(dataset_key: str) -> List[Dict]:
    key = dataset_key.lower()
    db = {
        "herg": [
            "hERG blockers classification and regression datasets for cardiotoxicity.",
            "Predicts inhibition at 1µM/10µM concentrations."
        ],
        "tox21": [
            "Tox21: 12-task toxicity classification (NR and SR pathways).",
            "Inputs: SMILES; Outputs: active/inactive labels."
        ],
        "caco2": ["Caco2_Wang: permeability (Papp) regression dataset."],
        "solubility": ["AqSolDB: aqueous solubility (logS) regression benchmark."],
        "lipophilicity": ["AstraZeneca Lipophilicity: logD prediction dataset."]
    }
    lines = db.get(key, [f"TDC dataset '{dataset_key}' summary not found."])
    return [_wrap(txt, "TDC", f"tdc_{key}_{i}") for i, txt in enumerate(lines)]
