import streamlit as st
import yfinance as yf

st.title("🔍 Live Data Check - BankNifty")
sym = "NSEBANK.NS"

st.write("Fetching from Yahoo Finance:", sym)
try:
    df = yf.download(sym, period="1d", interval="5m")
    st.write(df.tail())
    if df.empty:
        st.error("⚠ NO DATA RETURNED")
    else:
        st.success("✅ DATA FOUND")
except Exception as e:
    st.error(f"❌ ERROR — {e}")
