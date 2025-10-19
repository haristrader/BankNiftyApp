# pages/Data_Backfill.py
# -------------------------------------------------------------
# One-click historic data loader to Supabase
# - Daily: from 2020 (full)
# - 60m: ~2 years
# - 15m / 5m: last 60 days (yfinance limit)
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
from utils import sb_save_candles

st.set_page_config(page_title="Data Backfill", layout="wide")
st.title("📦 Data Backfill → Supabase")

symbol = st.text_input("Symbol", value="^NSEBANK")
st.caption("Tip: For stocks use TICKER.NS e.g., HDFCBANK.NS")

tf_opts = st.multiselect(
    "Select timeframes to backfill",
    ["1d", "60m", "15m", "5m"],
    default=["1d","60m","15m","5m"]
)

if st.button("Start Backfill"):
    total_rows = 0
    with st.spinner("Downloading & uploading…"):
        # 1D — full history (max)
        if "1d" in tf_opts:
            df = yf.download(symbol, period="max", interval="1d", auto_adjust=True, progress=False)
            if not df.empty:
                total_rows += sb_save_candles(df, symbol, "1d")
        # 60m — up to ~730d
        if "60m" in tf_opts:
            df = yf.download(symbol, period="730d", interval="60m", auto_adjust=True, progress=False)
            if not df.empty:
                total_rows += sb_save_candles(df, symbol, "60m")
        # 15m — 60 days
        if "15m" in tf_opts:
            df = yf.download(symbol, period="60d", interval="15m", auto_adjust=True, progress=False)
            if not df.empty:
                total_rows += sb_save_candles(df, symbol, "15m")
        # 5m — 60 days
        if "5m" in tf_opts:
            df = yf.download(symbol, period="60d", interval="5m", auto_adjust=True, progress=False)
            if not df.empty:
                total_rows += sb_save_candles(df, symbol, "5m")

    st.success(f"✅ Backfill complete. Rows upserted: {total_rows}")
    st.caption("Note: 5m/15m are limited by Yahoo (last ~60 days). For deeper intraday history, broker/NSE API required.")
