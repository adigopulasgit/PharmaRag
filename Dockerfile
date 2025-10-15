# ---------- BASE IMAGE ----------
    FROM continuumio/miniconda3

    # ---------- SETUP ----------
    WORKDIR /app
    COPY . /app
    
    # ---------- ENVIRONMENT ----------
    # Create a single environment named ragops
    RUN conda create -n ragops python=3.11 -y && \
        echo "conda activate ragops" >> ~/.bashrc
    
    SHELL ["conda", "run", "-n", "ragops", "/bin/bash", "-c"]
    
    # Install dependencies
    COPY environment.yml .
    RUN conda install pip -y && pip install -r requirements.txt
    
    # Optional: RDKit (for SMILES rendering)
    RUN conda install -c conda-forge rdkit -y
    
    # ---------- EXPOSE SERVICES ----------
    EXPOSE 8000
    EXPOSE 8501
    
    # ---------- ENTRY POINT ----------
    CMD ["bash", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run app/ui_streamlit.py --server.port=8501 --server.address=0.0.0.0"]
    