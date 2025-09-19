# app/pages/01_Dashboard.py
import os
import pandas as pd
import streamlit as st

CURRENT_DIR = os.path.dirname(__file__)
APP_DIR = os.path.dirname(CURRENT_DIR)
LOG_PATH = os.path.join(APP_DIR, "logs", "ragops_eval.csv")

st.set_page_config(page_title="PharmaRAG — Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("📊 PharmaRAG — Evaluation Dashboard")

# --- Robust reader that tolerates 12/13 column logs ---
def safe_read_logs(path: str) -> pd.DataFrame:
    cols13 = [
        "timestamp_iso","question","answer","k","latency_ms",
        "prompt_eval_count","eval_count","prompt_eval_duration_ms","eval_duration_ms",
        "groundedness","source_count","source_preview","used_fallback"
    ]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols13)

    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        n = len(df.columns)
        if n == 13:
            df.columns = cols13
            return df
        if n == 12:
            df.columns = cols13[:-1]
            df["used_fallback"] = 0
            return df
        # Fallback: force names
        df = pd.read_csv(path, engine="python", on_bad_lines="skip", header=None, names=cols13)
        if len(df) and df.iloc[0].tolist() == cols13:
            df = df.iloc[1:]
        return df
    except Exception:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip", header=None, names=cols13)
        if len(df) and list(df.iloc[0]) == cols13:
            df = df.iloc[1:]
        return df

# --- Optional one-click migration to 13 columns on disk ---
with st.sidebar:
    st.subheader("Maintenance")
    if st.button("Migrate logs to latest schema (adds used_fallback)"):
        df = safe_read_logs(LOG_PATH)
        df.to_csv(LOG_PATH, index=False)
        st.success("Logs migrated to 13-column schema.")

# --- Load logs ---
df = safe_read_logs(LOG_PATH)

if df.empty:
    st.warning("No logs found yet. Run a few queries in the main app.")
    st.stop()

# --- High-level metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total runs", len(df))
c2.metric("Avg latency (ms)", f"{df['latency_ms'].astype(float).mean():.0f}")
c3.metric("Avg groundedness", f"{df['groundedness'].astype(float).mean():.2f}")
c4.metric("Fallback uses", int(df['used_fallback'].astype(int).sum()))

st.markdown("### Recent Runs")
st.dataframe(df.tail(200), use_container_width=True)

# --- Charts ---
st.markdown("### Latency over time")
st.line_chart(df["latency_ms"].astype(float))

st.markdown("### Groundedness over time")
st.line_chart(df["groundedness"].astype(float))

# --- Downloads ---
st.download_button(
    "⬇️ Download logs (CSV)",
    df.to_csv(index=False),
    file_name="ragops_eval.csv",
    mime="text/csv"
)
