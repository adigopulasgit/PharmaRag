# build_index.py
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

docs = [
    {"doc_id": "1", "text": "EGFR inhibitors are drugs targeting the epidermal growth factor receptor.", "source": "PubChem"},
    {"doc_id": "2", "text": "hERG blockers can cause cardiotoxicity and QT prolongation.", "source": "BindingDB"},
    {"doc_id": "3", "text": "CYP2D6 is a key enzyme in drug metabolism affecting bioavailability.", "source": "TDC"}
]

df = pd.DataFrame(docs)
os.makedirs("artifacts", exist_ok=True)
df.to_parquet("artifacts/docs.parquet", index=False)

# embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embs = model.encode(df["text"].tolist(), normalize_embeddings=True).astype("float32")

# build FAISS index
index = faiss.IndexFlatIP(embs.shape[1])  # cosine similarity
index.add(embs)
faiss.write_index(index, "artifacts/faiss.index")

print("✅ FAISS index + docs.parquet built.")
