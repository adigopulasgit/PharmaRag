# app/ingestion/make_cards.py

import pandas as pd
import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
CARDS_DIR = PROCESSED_DIR / "cards"
CARDS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_parquet(PROCESSED_DIR / "bindingdb_standardized.parquet")
    print("✅ Loaded standardized BindingDB:", df.shape)

    grouped = df.groupby("inchikey", dropna=True)

    count = 0
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

        # Save card
        if inchikey:
            card_path = CARDS_DIR / f"{inchikey}.json"
            with open(card_path, "w") as f:
                json.dump(card, f, indent=2)
            count += 1

    print(f"✅ Wrote {count} evidence cards → {CARDS_DIR}")

if __name__ == "__main__":
    main()
