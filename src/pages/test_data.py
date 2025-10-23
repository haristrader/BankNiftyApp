import streamlit as st
import yfinance as yf
import pandas as pd

st.title("🔍 Live Data Check - BankNifty")

# --- Symbol & UI ---
sym = "NSEBANK.NS"
st.write("Fetching from Yahoo Finance:", sym)

# --- Try multiple fallbacks ---
intervals = ["5m", "15m", "1h", "1d"]
periods = ["5d", "1mo", "3mo"]

df = None
used_interval = None

for interval in intervals:
    for period in periods:
        try:
            temp = yf.download(sym, period=period, interval=interval, progress=False)
            if not temp.empty:
                df = temp
                used_interval = f"{interval} / {period}"
                break
        except Exception:
            continue
    if df is not None:
        break

# --- Display Results ---
if df is None or df.empty:
    st.error("⚠ NO DATA RETURNED — Check internet or Yahoo blocking.")
else:
    st.success(f"✅ Data found using interval: {used_interval}")
    st.dataframe(df.tail(10))

    st.caption("Tip: Market बंद हो तो 1h या 1d intervals में try करें।")
