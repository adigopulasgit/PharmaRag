from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def safe_to_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path); ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False); tmp.replace(path); return path

def save_json(obj: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path); ensure_dir(path.parent)
    with open(path, "w") as f: json.dump(obj, f, indent=2); return path

def canonicalize_smiles(smiles: Optional[str]) -> Optional[str]:
    if not smiles or not isinstance(smiles, str): return None
    mol = Chem.MolFromSmiles(smiles); 
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def to_inchikey(smiles: Optional[str]) -> Optional[str]:
    if not smiles: return None
    mol = Chem.MolFromSmiles(smiles)
    return MolToInchiKey(mol) if mol else None

def now_metadata(source: str, extra: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    meta = {"source": source, "created_at_unix": int(time.time()), "schema_version": "v1"}
    if extra: meta.update(extra); return meta
