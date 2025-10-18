import streamlit as st
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import fetch, DEFAULT_SYMBOL, fib_confidence

st.title("🌀 Fibonacci Retracement Analysis")

# --- User Inputs ---
symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
period = st.selectbox("Period", ["7d", "14d", "30d", "60d"], index=1)
interval = st.selectbox("Interval", ["5m", "15m", "60m", "1d"], index=0)
lookback = st.slider("Swing Lookback (Bars)", 50, 800, 200, 10)

# --- Get Data ---
df = fetch(symbol, period, interval, auto_adjust=True)
if df.empty:
    st.error("⚠ No data found for Fibonacci analysis. Try different settings.")
    st.stop()

# --- Compute Fibonacci Confidence ---
score, levels = fib_confidence(df, lookback)

# --- Score Display ---
st.metric("📊 Fibonacci Confidence Score", f"{score}/100")

# --- Display Levels ---
fib_df = pd.DataFrame({"Level": list(levels.keys()), "Price": list(levels.values())}).set_index("Level")
st.dataframe(fib_df)

# --- Closing Chart ---
st.line_chart(df["close"].tail(200))

# --- Send Score to Dashboard ---
st.session_state.setdefault("scores", {})
st.session_state["scores"]["fibonacci"] = score

st.success("✅ Fibonacci Score saved to Dashboard")
