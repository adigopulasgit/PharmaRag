# app/ingestion/bindingdb_loader.py

import requests
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
from sentence_transformers import SentenceTransformer
import faiss
import time

INDEX_DIR = Path("faiss_indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dim)

def to_inchikey(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MolToInchiKey(mol) if mol else None
    except Exception:
        return None

def fetch_by_uniprot(uniprot_id: str, cutoff: int = 10000):
    """Fetch ligands for a protein target via UniProt ID."""
    url = f"http://bindingdb.org/rest/getLigandsByUniprots?uniprot={uniprot_id}&cutoff={cutoff}&response=application/json"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"⚠️ API failed for {uniprot_id}")
        return []
    try:
        return r.json()
    except Exception:
        print(f"⚠️ Invalid JSON for {uniprot_id}")
        return []

def process_and_add(data):
    """Convert BindingDB JSON response → embeddings → add to FAISS."""
    if not data:
        return
    df = pd.DataFrame(data)
    if "smiles" not in df.columns:
        return
    df["inchikey"] = df["smiles"].apply(to_inchikey)

    texts = df["smiles"].dropna().astype(str).tolist()
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=False)
    index.add(embeddings)
    print(f"✅ Added {len(texts)} ligands to FAISS")

def main():
    # Example UniProt IDs (replace with a larger list later)
    uniprot_ids = ["P35355", "P00533", "Q9Y243"]  # ADRB2, EGFR, etc.
    
    for uid in uniprot_ids:
        data = fetch_by_uniprot(uid)
        process_and_add(data)
        time.sleep(1)  # avoid spamming API

    out_path = INDEX_DIR / "bindingdb_full.index"
    faiss.write_index(index, str(out_path))
    print(f"🎉 Saved FAISS index → {out_path}, size={index.ntotal}")

if __name__ == "__main__":
    main()
