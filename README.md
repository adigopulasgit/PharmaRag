<div align="center">


<h3>⚗️ PharmaRAG: Retrieval-Augmented AI for Drug Discovery</h3>

<p>
  <a href="https://github.com/adigopulasgit/pharmarag" target="_blank">🌐 GitHub</a> •
  <a href="#-quick-start">⚡ Quick Start</a> •
  <a href="#-example-queries">🧬 Examples</a> •
  <a href="#-monitoring--logging">📊 Dashboard</a>
</p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python">
  <img alt="Framework" src="https://img.shields.io/badge/framework-Streamlit%20%7C%20FastAPI-orange?logo=fastapi">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="GitHub stars" src="https://img.shields.io/github/stars/adigopulasgit/pharmarag?style=social">
</p>

</div>

---

## 🎯 The Challenge in Modern Drug Discovery

Researchers rely on multiple biochemical databases — BindingDB, PubChem, ChEMBL, and TDC — each containing fragments of vital information.  
But manual searches are slow, disconnected, and require expertise across domains.

**PharmaRAG** solves this by combining **retrieval-augmented generation (RAG)** with **LLM reasoning**,  
enabling natural-language queries grounded in scientific evidence.

---

<div align="center">

## ⚗️ The PharmaRAG Solution

</div>

**PharmaRAG** automatically:
- Retrieves molecular and assay data from BindingDB, PubChem, ChEMBL, and TDC  
- Extracts structured evidence (SMILES, IC₅₀, Ki, Kd, assay summaries)  
- Generates contextual insights with a local LLM (LLaMA, Mistral, etc.)  
- Logs every query for monitoring, reproducibility, and auditing  

```bash




🚀 Quick Start
# 1️⃣ Clone the repository
git clone https://github.com/adigopulasgit/pharmarag.git
cd pharmarag

# 2️⃣ Create & activate conda environment
conda create -n ragops python=3.11 -y
conda activate ragops

# 3️⃣ Install dependencies
pip install -r requirements.txt
conda install -c conda-forge rdkit -y  # optional, for SMILES rendering
🧠 Start the FastAPI backend
uvicorn app.api:app --reload
📍 Visit http://127.0.0.1:8000/docs
💬 Launch the Streamlit Chat UI
streamlit run app/ui_streamlit.py
📍 Open http://localhost:8501
📊 Run the Monitoring Dashboard
streamlit run app/monitor/dashboard.py
<div align="center">
🧩 Project Structure
</div>

PharmaRAG/
│
├── app/
│   ├── api.py                 # FastAPI REST backend
│   ├── retriever.py           # Unified web + local retrieval
│   ├── llm_service.py         # LLM reasoning + grounding
│   ├── ui_streamlit.py        # Streamlit chat interface
│   ├── ingestion/             # Web data loaders
│   ├── monitor/
│   │   ├── logger.py          # Logs queries to CSV
│   │   └── dashboard.py       # Metrics visualization
│
├── data/
│   └── ragops_eval.csv        # Auto-logged query results
├── requirements.txt
├── environment.yml
└── Dockerfile (optional)
```

<div align="center">
🧠 Features
</div>

🔍 Retrieval-Augmented QA over biochemical sources

🧬 Automatic dataset routing (BindingDB, ChEMBL, PubChem, TDC)

🧠 LLM fallback reasoning for contextual interpretation

🧪 SMILES → molecule rendering (RDKit)

📑 Evidence provenance view for transparency

💾 CSV-based query logging with latency tracking

📊 Interactive dashboard for performance analytics

⚙️ Local-only operation — no paid API keys required


<div align="center">
🧾 Credits
</div> 
Developed by Guru Ganesh Adigopula  

🎓 M.S. Computer Science, Texas A&M University – Corpus Christi  

📘 Advanced Software Engineering (COSC 6370)  

Instructor: Dr. Carlos Rubio-Medrano  

<div align="center">
📜 License
</div>
MIT License © 2025 Guru Ganesh Adigopula  

<div align="center">
⭐ Star this repository to support PharmaRAG!

🧠 Retrieval-Augmented AI for the next generation of drug discovery.
</div>
