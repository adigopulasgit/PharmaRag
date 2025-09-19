import argparse
import math
import requests
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def to_inchikey(smiles: str | None) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return MolToInchiKey(mol) if mol else None

def fetch_cids_json(aid: int) -> list[int]:
    # JSON endpoint works where CSV may 400
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/cids/JSON"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()
    # path: PC_AssayContainer -> [0] -> assaY -> [0] -> cid -> list of ints
    try:
        return js["PC_AssayContainer"][0]["assay"][0]["cid"]
    except Exception:
        # some assays expose cids directly under InformationList
        if "InformationList" in js and "Information" in js["InformationList"]:
            return [i["CID"] for i in js["InformationList"]["Information"] if "CID" in i]
        raise

def fetch_smiles_for_cids(cids: list[int], batch_size: int = 100) -> pd.DataFrame:
    rows = []
    total = len(cids)
    for i in range(0, total, batch_size):
        batch = cids[i:i+batch_size]
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            + ",".join(map(str, batch))
            + "/property/CanonicalSMILES/JSON"
        )
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        js = r.json()
        props = js.get("PropertyTable", {}).get("Properties", [])
        for p in props:
            rows.append({"cid": p.get("CID"), "smiles": p.get("CanonicalSMILES")})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aid", type=int, required=True, help="PubChem Assay ID (AID)")
    ap.add_argument("--limit", type=int, default=200, help="Limit number of compounds (for demo)")
    args = ap.parse_args()

    print(f"📡 Fetching CIDs (JSON) for AID={args.aid} ...")
    cids = fetch_cids_json(args.aid)
    if not cids:
        raise SystemExit("No CIDs returned for this AID.")

    if args.limit:
        cids = cids[: args.limit]
    print(f"✅ Got {len(cids)} CIDs; resolving SMILES in batches...")

    df_props = fetch_smiles_for_cids(cids, batch_size=100)
    # ensure all requested CIDs appear (some may miss SMILES)
    df = pd.DataFrame({"cid": cids}).merge(df_props, on="cid", how="left")
    df["inchikey"] = df["smiles"].apply(to_inchikey)

    out = DATA_DIR / f"pubchem_assay_{args.aid}_cids.parquet"
    df.to_parquet(out, index=False)
    print(f"🎉 Wrote {len(df)} rows -> {out}")

if __name__ == "__main__":
    main()
