# pages/04_Price_Action.py
import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import fetch, DEFAULT_SYMBOL

st.title("🎯 Price Action (50% Rule)")

symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
period = st.selectbox("Period", ["7d","14d","30d"], index=0)
interval = st.selectbox("Interval", ["5m","15m"], index=0)
mid_factor = st.slider("Previous candle mid factor", 0.1, 0.9, 0.5, 0.05)

df = fetch(symbol, period, interval, auto_adjust=True)
if df.empty:
    st.error("No data."); st.stop()

sig = ["HOLD"]
for i in range(1, len(df)):
    ph, pl = df["high"].iloc[i-1], df["low"].iloc[i-1]
    prev_mid = pl + (ph - pl)*mid_factor
    c = df.iloc[i]
    if (c["low"] >= prev_mid) and (c["close"] > ph):
        sig.append("BUY")
    elif (c["high"] <= prev_mid) and (c["close"] < pl):
        sig.append("SELL")
    else:
        sig.append("HOLD")

df2 = df.copy()
df2["signal"] = sig
last = df2.tail(1)["signal"].item()

score = 50.0
if last == "BUY": score = 75.0
elif last == "SELL": score = 25.0
st.metric("Price Action Score", score)
st.write("Last signal:", last)
st.line_chart(df2[["close"]].tail(300))

st.session_state.setdefault("scores", {})["priceaction"] = score
st.success("Price Action score saved to Dashboard.")
