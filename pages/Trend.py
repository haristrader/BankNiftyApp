# pages/Trend.py
# -------------------------------------------------------------
# Trend Strength (Multi-TF EMA & RSI)
# • Uses utils.fetch() to get OHLCV
# • Computes strength score 0–100 + BUY/SELL/NEUTRAL bias
# • SMART MODE Supabase save:
#     - silently saves current TF candles
#     - also saves Daily(1D) candles for AI/history
# • Stores to session_state["performance"]["trend"]
# • Logs module score to Supabase (module_scores)
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import ta

from utils import (
    fetch, DEFAULT_SYMBOL,          # your existing helpers
    sb_save_candles, sb_record_module_score  # Supabase helpers
)

st.set_page_config(page_title="Trend Strength", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Trend Strength (EMA & RSI)")

# ---------------- Sidebar Controls ----------------
symbol   = st.sidebar.text_input("Symbol", value=DEFAULT_SYMBOL)  # "^NSEBANK" from utils
period   = st.sidebar.selectbox("Period", ["2d","5d","7d","14d","1mo","3mo","6mo"], index=2)
interval = st.sidebar.selectbox("Interval (TF)", ["5m","15m","60m","1d"], index=1)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Market बंद हो तो 1d चुनें (safe). Intraday TFs live hours में best हैं.")

# ---------------- Fetch Data ----------------
with st.spinner("Fetching OHLCV…"):
    df = fetch(symbol, period=period, interval=interval, auto_adjust=True)

if df is None or df.empty or len(df) < 30:
    st.error("No/insufficient data. Try different Period/Interval or market hours.")
    st.stop()

# -------------- INDICATORS & SCORE --------------
idf = df.copy()
idf["ema20"] = ta.trend.EMAIndicator(idf["close"], 20).ema_indicator()
idf["ema50"] = ta.trend.EMAIndicator(idf["close"], 50).ema_indicator()
idf["rsi"]   = ta.momentum.RSIIndicator(idf["close"], 14).rsi()
idf = idf.dropna()
if idf.empty:
    st.error("Not enough candles after indicators. Try longer period.")
    st.stop()

last = idf.iloc[-1]
score_raw = 0
# close vs EMA20/EMA50
score_raw += 2 if last["close"] > last["ema20"] else -2
score_raw += 3 if last["close"] > last["ema50"] else -3
# RSI zones
if last["rsi"] > 60:
    score_raw += 3
elif last["rsi"] < 40:
    score_raw -= 3

# normalize ~ (-8..+8) → (0..100)
normalized = ((score_raw + 8.0) / 16.0) * 100.0
trend_score = float(max(0.0, min(100.0, normalized)))

# bias label
if trend_score >= 60:
    bias = "BUY"
elif trend_score <= 40:
    bias = "SELL"
else:
    bias = "NEUTRAL"

# -------------- UI: Meter + Stats --------------
st.subheader("Trend Strength Meter (0–100)")
st.progress(int(trend_score))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Score", f"{trend_score:.0f}")
c2.metric("RSI(14)", f"{last['rsi']:.1f}")
c3.metric("EMA20", f"{last['ema20']:.1f}")
c4.metric("EMA50", f"{last['ema50']:.1f}")

if bias == "BUY":
    st.success("Bias: BUY — strength on the long side.")
elif bias == "SELL":
    st.error("Bias: SELL — strength on the short side.")
else:
    st.info("Bias: NEUTRAL — wait for clarity or other modules.")

with st.expander("Recent (last 60)"):
    st.dataframe(idf.tail(60)[["open","high","low","close","volume","ema20","ema50","rsi"]])

# -------------- SILENT SUPABASE SAVES (SMART MODE) --------------
# Save current TF candles
try:
    sb_save_candles(df, symbol, interval)
except Exception:
    pass  # silent

# Also save Daily candles (Smart Mode) for AI/history
try:
    daily_df = fetch(symbol, period="6mo", interval="1d", auto_adjust=True)
    if daily_df is not None and not daily_df.empty:
        sb_save_candles(daily_df, symbol, "1d")
except Exception:
    pass  # silent

# Log module score to Supabase (for AI learning)
try:
    sb_record_module_score(
        module="trend",
        score=float(trend_score),
        bias=bias,
        symbol=symbol,
        tf=interval,
        meta={"period": period, "note": "Trend page"}
    )
except Exception:
    pass  # silent

# -------------- Save for Dashboard Fusion --------------
st.session_state.setdefault("performance", {})
st.session_state["performance"]["trend"] = {
    "tf": interval,
    "mode": "daily" if interval == "1d" else "intraday",
    "final_score": float(trend_score),  # Dashboard expects final_score
    "bias": bias,
    "symbol": symbol,
}

st.caption("✅ Trend score saved for Dashboard fusion.")
