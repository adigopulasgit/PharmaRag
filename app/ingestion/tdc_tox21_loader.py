import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def to_inchikey(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return MolToInchiKey(mol) if mol else None

def main():
    # Get all available Tox21 labels
    labels = retrieve_label_name_list("tox21")
    print(f"Found {len(labels)} Tox21 endpoints: {labels}")

    all_dfs = []

    for label in labels:
        print(f"⚡ Processing Tox21 endpoint: {label}")
        dataset = Tox(name="Tox21", label_name=label, path=str(DATA_DIR))
        split = dataset.get_split()

        # Merge splits
        df = pd.concat([split["train"], split["valid"], split["test"]])
        df = df.rename(columns={"Drug": "smiles", "Y": "label"})
        df["inchikey"] = df["smiles"].apply(to_inchikey)
        df["endpoint"] = label

        # Save per-label parquet
        out = DATA_DIR / f"tdc_tox21_{label}.parquet"
        df.to_parquet(out, index=False)
        print(f"✅ Saved {len(df)} rows for {label} -> {out}")

        all_dfs.append(df)

    # Optionally merge all into one big file
    merged = pd.concat(all_dfs)
    merged_out = DATA_DIR / "tdc_tox21_all.parquet"
    merged.to_parquet(merged_out, index=False)
    print(f"🎉 Finished. Total {len(merged)} rows across {len(labels)} endpoints -> {merged_out}")

if __name__ == "__main__":
    main()
