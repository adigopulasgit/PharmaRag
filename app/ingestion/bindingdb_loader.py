# app/ingestion/bindingdb_loader.py

import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def to_inchikey(smiles: str):
    """Convert SMILES → InChIKey safely."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MolToInchiKey(mol) if mol else None
    except Exception:
        return None

def main():
    print("📦 Loading BindingDB_All.tsv...")

    # load only first 500k rows for faster iteration
    df = pd.read_csv(
        RAW_DIR / "BindingDB_All.tsv",
        sep="\t",
        low_memory=False,
        dtype=str,
        nrows=500000
    )

    # pick only essential columns for RAG/ML
    keep_cols = [
        "BindingDB Reactant_set_id",
        "Ligand SMILES",
        "Ligand InChI",
        "Ligand Name",
        "Target Name Assigned by Curator or DataSource",
        "BindingDB Target Chain  Sequence",
        "Ki (nM)",
        "IC50 (nM)",
        "Kd (nM)"
    ]

    df = df[[c for c in keep_cols if c in df.columns]]

    # standardize SMILES → InChIKey
    if "Ligand SMILES" in df.columns:
        print("🧪 Generating InChIKeys...")
        df["inchikey"] = df["Ligand SMILES"].apply(to_inchikey)
    else:
        print("⚠️ No Ligand SMILES column found!")

    # save output
    out_path = PROCESSED_DIR / "bindingdb_lite.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Done: {len(df)} rows → {out_path}")

if __name__ == "__main__":
    main()
