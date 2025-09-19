# app/ingestion/tdc_ingestor.py

import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
from tdc.single_pred import ADME, Tox
from tdc.multi_pred import DTI

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def to_inchikey(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MolToInchiKey(mol) if mol else None
    except Exception:
        return None

def normalize_df(df, source, activity_type, target_col=None):
    """Unify schema: [smiles, inchikey, ligand_name, target, activity_type, value, pValue]"""
    norm = pd.DataFrame()
    norm["smiles"] = df["Drug"] if "Drug" in df.columns else df["Drug_ID"]
    norm["inchikey"] = norm["smiles"].apply(to_inchikey)
    norm["ligand_name"] = df["Drug_ID"].astype(str) if "Drug_ID" in df.columns else None
    norm["target"] = df[target_col] if target_col and target_col in df.columns else activity_type
    norm["activity_type"] = activity_type
    norm["value"] = df["Y"]

    # Compute pValue if numeric
    def safe_pvalue(v):
        try:
            val = float(v)
            molar = val * 1e-9  # assume nM → M
            return -pd.np.log10(molar) if molar > 0 else None
        except Exception:
            return None
    norm["pValue"] = norm["value"].apply(safe_pvalue)

    return norm

def save_dataset(df, name):
    out_path = PROCESSED_DIR / f"tdc_{name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved {name} → {out_path} ({len(df)} rows)")
    print(df.head(3))
    return out_path

def ingest_adme(name, target_name):
    print(f"📦 Loading ADME dataset: {name}...")
    data = ADME(name=name)
    split = data.get_split()
    df = pd.concat([split["train"], split["valid"], split["test"]], ignore_index=True)
    norm = normalize_df(df, "ADME", name, target_col=None)
    norm["target"] = target_name
    return save_dataset(norm, name)

def ingest_tox(label_name):
    print(f"📦 Loading TOX dataset: Tox21 ({label_name})...")
    data = Tox(name="Tox21", label_name=label_name)
    split = data.get_split()
    df = pd.concat([split["train"], split["valid"], split["test"]], ignore_index=True)
    norm = normalize_df(df, "Tox", label_name, target_col=None)
    return save_dataset(norm, f"Tox21_{label_name}")

def ingest_dti(name):
    print(f"📦 Loading DTI dataset: {name}...")
    data = DTI(name=name)
    split = data.get_split()
    df = pd.concat([split["train"], split["valid"], split["test"]], ignore_index=True)
    norm = normalize_df(df, "DTI", name, target_col="Target" if "Target" in df.columns else None)
    return save_dataset(norm, name)

def main():
    # ADME datasets with explicit targets
    adme_map = {
        "cyp2d6_veith": "CYP2D6",
        "lipophilicity_astrazeneca": "logP",
        "solubility_aqsoldb": "solubility"
    }

    tox21_labels = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]

    dti_datasets = ["DAVIS", "KIBA", "BindingDB_Ki"]

    for name, target in adme_map.items():
        ingest_adme(name, target)

    for label in tox21_labels:
        ingest_tox(label)

    for name in dti_datasets:
        ingest_dti(name)

if __name__ == "__main__":
    main()
