# app/ingestion/tdc_ingestor.py

import pandas as pd
from pathlib import Path
from tdc.single_pred import ADME, Tox
from tdc.multi_pred import DTI

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def save_dataset(df, name):
    """Save dataset to processed/ as parquet + preview."""
    out_path = PROCESSED_DIR / f"tdc_{name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved {name} → {out_path} ({len(df)} rows)")
    print(df.head(3))
    return out_path

def ingest_adme(name):
    print(f"📦 Loading ADME dataset: {name}...")
    data = ADME(name=name)
    split = data.get_split()
    df = pd.concat([split["train"], split["valid"], split["test"]], ignore_index=True)
    return save_dataset(df, name)

def ingest_tox(name, label_name):
    print(f"📦 Loading TOX dataset: {name} ({label_name})...")
    data = Tox(name=name, label_name=label_name)
    split = data.get_split()
    df = pd.concat([split["train"], split["valid"], split["test"]], ignore_index=True)
    return save_dataset(df, f"{name}_{label_name}")

def ingest_dti(name):
    print(f"📦 Loading DTI dataset: {name}...")
    data = DTI(name=name)
    split = data.get_split()
    df = pd.concat([split["train"], split["valid"], split["test"]], ignore_index=True)
    return save_dataset(df, name)

def main():
    # Choose datasets here 👇
    adme_datasets = ["cyp2d6_veith", "lipophilicity_astrazeneca", "solubility_aqsoldb"]

    # Tox21 endpoints (all 12 by default)
    tox21_labels = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]

    dti_datasets = ["DAVIS", "KIBA", "BindingDB_Ki"]

    # Ingest ADME datasets
    for name in adme_datasets:
        ingest_adme(name)

    # Ingest Tox21 endpoints
    for label in tox21_labels:
        ingest_tox("Tox21", label)

    # Ingest DTI datasets
    for name in dti_datasets:
        ingest_dti(name)

if __name__ == "__main__":
    main()
