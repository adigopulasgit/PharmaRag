"""
PharmaRAG — RAGOps Monitoring Dashboard (stable for Streamlit ≥1.36)
Includes:
 - Auto-refresh every 30 s
 - Rolling averages (7-query window)
 - CSV download
 - Optional email alerts when grounded% < 60%
"""

import os
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------- Config ----------
DATA_DIR = Path("data")
LOG_PATH = DATA_DIR / "ragops_eval.csv"
AUTO_REFRESH_SEC = 30
ROLLING_N = 7

# Optional email alert — set these env vars to enable
SMTP_HOST = os.getenv("PRAG_SMTP_HOST", "")
SMTP_PORT = int(os.getenv("PRAG_SMTP_PORT", "587") or 587)
SMTP_USER = os.getenv("PRAG_SMTP_USER", "")
SMTP_PASS = os.getenv("PRAG_SMTP_PASS", "")
ALERT_TO = os.getenv("PRAG_ALERT_TO", "")

# ---------- UI setup ----------
st.set_page_config(page_title="PharmaRAG Dashboard", page_icon="📊", layout="wide")
st.title("📊 PharmaRAG — RAGOps Monitoring")
st.caption(f"Auto-refreshing every {AUTO_REFRESH_SEC}s")

# ---------- Auto-refresh ----------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="pharmarag_refresh")
except Exception:
    st.info("Install `streamlit-autorefresh` to enable live auto-refresh (`pip install streamlit-autorefresh`).")

# ---------- Load data ----------
if not LOG_PATH.exists():
    st.warning("⚠️ No log file yet. Run a few queries in the chat app to generate `data/ragops_eval.csv`.")
    st.stop()

df = pd.read_csv(LOG_PATH)
if df.empty:
    st.warning("Log file is empty. Run a few queries first.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

time_min = st.sidebar.date_input("From date", value=df["timestamp"].min().date())
time_max = st.sidebar.date_input("To date", value=df["timestamp"].max().date())

mask = (df["timestamp"].dt.date >= time_min) & (df["timestamp"].dt.date <= time_max)
df = df.loc[mask].copy()

dataset_vals = sorted(set(";".join(df["datasets"].dropna()).split(";")) - {""})
sel = st.sidebar.multiselect("Datasets", dataset_vals, default=dataset_vals)
if sel:
    df = df[df["datasets"].apply(lambda s: any(x in str(s) for x in sel))]

# ---------- KPIs ----------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Queries", len(df))
c2.metric("Grounded", int(df["grounded"].sum()))
c3.metric("Grounded %", round(df["grounded"].mean() * 100, 2))
c4.metric("Avg Latency (ms)", int(df["latency_ms"].mean()))
c5.metric("Avg Tokens Used", int(df.get("tokens_used", pd.Series([0]*len(df))).mean()))

# ---------- Rolling averages ----------
df_sorted = df.sort_values("timestamp")
df_sorted["grounded_num"] = df_sorted["grounded"].astype(int)
df_sorted["grounded_roll"] = df_sorted["grounded_num"].rolling(ROLLING_N, min_periods=1).mean()
df_sorted["latency_roll"] = df_sorted["latency_ms"].rolling(ROLLING_N, min_periods=1).mean()

st.markdown(f"### 📈 Rolling Averages (window={ROLLING_N} queries)")
lc1, lc2 = st.columns(2)
with lc1:
    st.line_chart(df_sorted.set_index("timestamp")["grounded_roll"], height=220)
with lc2:
    st.line_chart(df_sorted.set_index("timestamp")["latency_roll"], height=220)

# ---------- Latency histogram ----------
st.markdown("### ⏱️ Latency Distribution")
st.bar_chart(df["latency_ms"], height=200)

# ---------- Top slow queries ----------
st.markdown("### 🔝 Top Slow Queries")
slow = df.sort_values("latency_ms", ascending=False).head(10)
st.dataframe(
    slow[["timestamp", "question", "datasets", "latency_ms", "grounded", "evidence_count"]],
    use_container_width=True,
)

# ---------- Source distribution ----------
st.markdown("### 🧬 Source Distribution")
src = (
    df["datasets"].dropna()
    .str.split(";")
    .explode()
    .value_counts()
    .rename_axis("source")
    .reset_index(name="count")
)
if not src.empty:
    st.bar_chart(src.set_index("source")["count"], height=220)
else:
    st.info("No dataset tags found yet.")

# ---------- Download button ----------
st.markdown("### 💾 Export Logs")
st.download_button(
    "Download ragops_eval.csv",
    data=Path(LOG_PATH).read_bytes(),
    file_name="ragops_eval.csv",
    mime="text/csv",
)

# ---------- Optional email alert ----------
def _send_alert(subject: str, body: str) -> Optional[str]:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and ALERT_TO):
        return "Email not configured"
    try:
        import smtplib
        from email.mime.text import MIMEText
        to_list = [x.strip() for x in ALERT_TO.split(",") if x.strip()]
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = ", ".join(to_list)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, to_list, msg.as_string())
        return "sent"
    except Exception as e:
        return f"email error: {e}"

st.markdown("---")
low_grounded = df["grounded"].mean() < 0.60
if low_grounded:
    st.error("⚠️ Grounded rate below 60 %")
    status = _send_alert(
        "[PharmaRAG] Grounded % below 60 %",
        f"Grounded mean: {df['grounded'].mean():.2%}\nTotal queries: {len(df)}\nTime window: {time_min} → {time_max}",
    )
    st.caption(f"Alert status: {status}")
else:
    st.success("✅ Grounded rate healthy (≥ 60 %)")
