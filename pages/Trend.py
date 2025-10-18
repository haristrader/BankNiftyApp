# Trend.py (FINAL INTEGRATED VERSION)
import streamlit as st
import numpy as np
import pandas as pd
import ta   # Technical Analysis Library (pip install ta)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import fetch, DEFAULT_SYMBOL  # Fetch from utils.py

st.title("📈 Trend Strength (EMA + RSI)")

# --- User Inputs ---
symbol = st.text_input("Symbol (Index/Stock)", value=DEFAULT_SYMBOL)
period = st.selectbox("Period", ["7d", "14d", "30d", "60d"], index=1)
interval = st.selectbox("Interval", ["5m", "15m", "60m", "1d"], index=0)

# --- Fetch Data ---
df = fetch(symbol, period, interval, auto_adjust=True)
if df.empty:
    st.error("⚠ No data received! Try different period or symbol.")
    st.stop()

# --- Indicators ---
df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

latest = df.iloc[-1]

# --- Trend Scoring (0–100) ---
score = 50  # Base score

# EMA Impact
score += 20 if latest["close"] > latest["ema20"] else -20
score += 20 if latest["close"] > latest["ema50"] else -20

# RSI Bias
if latest["rsi"] > 60:
    score += 10
elif latest["rsi"] < 40:
    score -= 10

# Clamp score to [0, 100]
score = int(np.clip(score, 0, 100))

# --- Display Score ---
st.metric("📊 Trend Strength Score", f"{score}/100")

# --- Chart View ---
st.line_chart(df[["close", "ema20", "ema50"]].tail(200))

# --- Save to Dashboard Session ---
st.session_state.setdefault("scores", {})
st.session_state["scores"]["trend"] = score

st.success("✅ Trend Score saved to Dashboard")
