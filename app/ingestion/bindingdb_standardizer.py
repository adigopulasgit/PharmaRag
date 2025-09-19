# app/ingestion/bindingdb_standardizer.py

import pandas as pd
import numpy as np
import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
CARDS_DIR = PROCESSED_DIR / "cards"
CARDS_DIR.mkdir(parents=True, exist_ok=True)

def nm_to_pX(value_nm):
    """Convert nM values → molar → pX scale (e.g. IC50 → pIC50)."""
    try:
        val = float(value_nm)
        molar = val * 1e-9  # nM → M
        if molar <= 0:
            return None
        return -np.log10(molar)
    except Exception:
        return None

def main():
    print("📦 Loading bindingdb_lite.parquet...")
    df = pd.read_parquet(PROCESSED_DIR / "bindingdb_lite.parquet")

    # Normalize activity columns
    for col in ["Ki (nM)", "IC50 (nM)", "Kd (nM)"]:
        if col in df.columns:
            pcol = "p" + col.split()[0]  # Ki (nM) → pKi
            print(f"🧪 Converting {col} → {pcol}")
            df[pcol] = df[col].apply(nm_to_pX)

    # Auto-detect name and smiles columns
    name_col_candidates = [c for c in df.columns if "ligand" in c.lower() and "name" in c.lower()]
    name_col = name_col_candidates[0] if name_col_candidates else None

    smiles_col_candidates = [c for c in df.columns if "smiles" in c.lower()]
    smiles_col = smiles_col_candidates[0] if smiles_col_candidates else None

    # Deduplicate by InChIKey
    print("🔗 Deduplicating ligands by InChIKey...")
    grouped = df.groupby("inchikey", dropna=True)

    for inchikey, subdf in grouped:
        if pd.isna(inchikey):
            continue

        ligand_name = subdf[name_col].dropna().unique().tolist() if name_col else []
        smiles = subdf[smiles_col].dropna().unique().tolist() if smiles_col else []

        targets = []
        for _, row in subdf.iterrows():
            target = row.get("Target Name Assigned by Curator or DataSource")
            seq = row.get("BindingDB Target Chain  Sequence")
            assays = []

            for col in ["Ki (nM)", "IC50 (nM)", "Kd (nM)"]:
                if col in subdf.columns and pd.notna(row.get(col)):
                    assays.append({
                        "type": col.replace(" (nM)", ""),
                        "value_nM": row[col],
                        "pValue": row.get("p" + col.split()[0])
                    })

            if target or assays:
                targets.append({
                    "name": target,
                    "sequence": seq,
                    "assays": assays
                })

        card = {
            "inchikey": inchikey,
            "ligand_name": ligand_name,
            "smiles": smiles,
            "targets": targets
        }

        # Save evidence card
        card_path = CARDS_DIR / f"{inchikey}.json"
        with open(card_path, "w") as f:
            json.dump(card, f, indent=2)

    # Save standardized parquet
    out_path = PROCESSED_DIR / "bindingdb_standardized.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Standardization done → {out_path}")
    print(f"✅ Evidence cards written to {CARDS_DIR}")

if __name__ == "__main__":
    main()
