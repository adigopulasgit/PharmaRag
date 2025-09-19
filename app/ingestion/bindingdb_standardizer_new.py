# app/ingestion/bindingdb_standardizer.py

import pandas as pd
import numpy as np
import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
CARDS_DIR = PROCESSED_DIR / "cards"
CARDS_DIR.mkdir(parents=True, exist_ok=True)

def nm_to_pX(value_nm):
    """Convert nM values → molar → pX (e.g. IC50 → pIC50)."""
    try:
        val = float(value_nm)
        molar = val * 1e-9  # nM → M
        if molar <= 0:
            return None
        return -np.log10(molar)
    except Exception:
        return None

def normalize_bindingdb(df):
    """Convert BindingDB lite dataframe → unified schema."""
    norm_rows = []

    # auto-detect ligand name + smiles
    name_col = next((c for c in df.columns if "ligand" in c.lower() and "name" in c.lower()), None)
    smiles_col = next((c for c in df.columns if "smiles" in c.lower()), None)

    for _, row in df.iterrows():
        smiles = row.get(smiles_col) if smiles_col else None
        inchikey = row.get("inchikey")
        ligand_name = row.get(name_col) if name_col else None
        target = row.get("Target Name Assigned by Curator or DataSource")
        target_seq = row.get("BindingDB Target Chain  Sequence")

        for col in ["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"]:
            if col in df.columns and pd.notna(row.get(col)):
                activity_type = col.replace(" (nM)", "")
                value = row[col]
                pval = nm_to_pX(value)
                norm_rows.append({
                    "smiles": smiles,
                    "inchikey": inchikey,
                    "ligand_name": ligand_name,
                    "target": target if target else target_seq,
                    "activity_type": activity_type,
                    "value": value,
                    "pValue": pval
                })

    return pd.DataFrame(norm_rows)

def main():
    print("📦 Loading bindingdb_lite.parquet...")
    df = pd.read_parquet(PROCESSED_DIR / "bindingdb_lite.parquet")

    print("🧪 Normalizing BindingDB data...")
    df_norm = normalize_bindingdb(df)

    # save unified parquet
    out_path = PROCESSED_DIR / "bindingdb_standardized.parquet"
    df_norm.to_parquet(out_path, index=False)
    print(f"✅ Saved standardized BindingDB → {out_path} ({len(df_norm)} rows)")

    # generate evidence cards per compound
    print("📝 Generating evidence cards...")
    grouped = df_norm.groupby("inchikey", dropna=True)

    for inchikey, subdf in grouped:
        card = {
            "inchikey": inchikey,
            "smiles": subdf["smiles"].dropna().unique().tolist(),
            "ligand_name": subdf["ligand_name"].dropna().unique().tolist(),
            "targets": []
        }

        for target, tdf in subdf.groupby("target"):
            assays = []
            for _, row in tdf.iterrows():
                assays.append({
                    "activity_type": row["activity_type"],
                    "value": row["value"],
                    "pValue": row["pValue"]
                })
            card["targets"].append({
                "target": target,
                "assays": assays
            })

        # save JSON card
        if inchikey:
            card_path = CARDS_DIR / f"{inchikey}.json"
            with open(card_path, "w") as f:
                json.dump(card, f, indent=2)

    print(f"✅ Evidence cards written to {CARDS_DIR}")

if __name__ == "__main__":
    main()
