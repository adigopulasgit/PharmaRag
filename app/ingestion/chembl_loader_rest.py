"""
ChEMBL Bioactivity Loader via REST API

- Fetches activities directly from ChEMBL REST endpoint
- Collects fields like IC50/Ki/Kd values, SMILES, targets
- Normalizes SMILES + InChIKey using RDKit
- Saves to CSV + Parquet + metadata JSON

Usage:
  python -m app.ingestion.chembl_loader_rest --limit 1000
"""

from __future__ import annotations
import argparse, requests, time, json
from pathlib import Path
from typing import Dict, List
import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
def canonicalize_smiles(smiles: str|None) -> str|None:
    if not smiles: return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def to_inchikey(smiles: str|None) -> str|None:
    if not smiles: return None
    mol = Chem.MolFromSmiles(smiles)
    return MolToInchiKey(mol) if mol else None

def fetch_page(limit: int, offset: int) -> Dict:
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?limit={limit}&offset={offset}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------- Main fetch ----------
def fetch_activities(total_limit: int, page_size: int = 200) -> pd.DataFrame:
    rows: List[Dict] = []
    offset = 0
    while offset < total_limit:
        page = fetch_page(min(page_size, total_limit-offset), offset)
        acts = page.get("activities", [])
        if not acts:
            break
        for a in acts:
            rows.append({
                "activity_id": a.get("activity_id"),
                "assay_chembl_id": a.get("assay_chembl_id"),
                "molecule_chembl_id": a.get("molecule_chembl_id"),
                "target_chembl_id": a.get("target_chembl_id"),
                "target_pref_name": a.get("target_pref_name"),
                "smiles": a.get("canonical_smiles"),
                "standard_type": a.get("standard_type"),
                "standard_relation": a.get("standard_relation"),
                "standard_value": a.get("standard_value"),
                "standard_units": a.get("standard_units"),
                "pchembl_value": a.get("pchembl_value"),
                "assay_description": a.get("assay_description")
            })
        offset += page_size
        print(f"Fetched {offset}/{total_limit} records...", flush=True)
        time.sleep(0.2)  # be nice to API

    df = pd.DataFrame(rows)
    if not df.empty:
        df["smiles"] = df["smiles"].apply(canonicalize_smiles)
        df["inchikey"] = df["smiles"].apply(to_inchikey)
    return df

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000, help="Number of rows to fetch")
    args = ap.parse_args()

    df = fetch_activities(args.limit)

    stem = "chembl_bioactivity_rest"
    csv_path = DATA_DIR / f"{stem}.csv"
    pq_path  = DATA_DIR / f"{stem}.parquet"
    meta_path= DATA_DIR / f"{stem}.metadata.json"

    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(pq_path, index=False)
    except Exception:
        pass

    meta = {
        "source": "ChEMBL REST",
        "limit": args.limit,
        "rows": int(len(df)),
        "columns": df.columns.tolist()
    }
    with open(meta_path, "w") as f: json.dump(meta, f, indent=2)

    print(f"✅ Wrote {len(df)} rows -> {csv_path}")

if __name__ == "__main__":
    main()
