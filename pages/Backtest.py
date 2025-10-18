import streamlit as st
import numpy as np
import pandas as pd
from utils import fetch, DEFAULT_SYMBOL

st.title("🧪 Backtest (Trailing SL Sim) — Index Proxy")

# User Inputs
symbol  = st.text_input("Symbol", value=DEFAULT_SYMBOL)
period  = st.selectbox("Period",  ["5d","14d","30d"], index=1)
interval = st.selectbox("Interval", ["5m", "15m"], index=0)
init_sl = st.number_input("Initial SL (points)", 5, 50, 10, 1)

# Safe Fetch Logic
df = fetch(symbol, period="5d", interval="5m", auto_adjust=True)
if df is None or df.empty:
    df = fetch(symbol, period="3mo", interval="1d", auto_adjust=True)

if df is None or df.empty:
    st.error("⚠ No data available even after fallbacks.")
    st.stop()

# Signal Generation (50% Rule)
signals = ["HOLD"]
for i in range(1, len(df)):
    ph, pl = df["high"].iloc[i-1], df["low"].iloc[i-1]
    prev_mid = (ph + pl) / 2.0
    c = df.iloc[i]
    if (c["low"] >= prev_mid) and (c["close"] > ph):
        signals.append("BUY")
    elif (c["high"] <= prev_mid) and (c["close"] < pl):
        signals.append("SELL")
    else:
        signals.append("HOLD")

df2 = df.copy()
df2["signal"] = signals

# Trailing SL Simulation
trades = []
pos=None; entry=None; sl=None
closes = df2["close"].values
times  = list(df2.index)

for i in range(len(df2)-1):
    s   = df2["signal"].iloc[i]
    nxt = float(closes[i+1]); t = times[i+1]

    if pos is None:
        if s == "BUY":
            pos="LONG"; entry=nxt; sl=entry - init_sl
        elif s == "SELL":
            pos="SHORT"; entry=nxt; sl=entry + init_sl
        continue

    if pos == "LONG":
        profit = nxt - entry
        if profit >= 10 and sl < entry:        sl = entry
        if profit >= 20 and sl < entry + 10:   sl = entry + 10
        if profit >= 30 and sl < entry + 15:   sl = entry + 15
        if profit >= 50:
            new_sl = nxt - (profit * 0.5)
            if new_sl > sl: sl = new_sl
        if nxt <= sl:
            trades.append(("LONG", entry, sl, t))
            pos=None; entry=None; sl=None

    elif pos == "SHORT":
        profit = entry - nxt
        if profit >= 10 and sl > entry:        sl = entry
        if profit >= 20 and sl > entry - 10:   sl = entry - 10
        if profit >= 30 and sl > entry - 15:   sl = entry - 15
        if profit >= 50:
            new_sl = nxt + (profit * 0.5)
            if new_sl < sl: sl = new_sl
        if nxt >= sl:
            trades.append(("SHORT", entry, sl, t))
            pos=None; entry=None; sl=None

tr = pd.DataFrame(trades, columns=["side","entry","exit","exit_time"])
if tr.empty:
    st.info("No closed trades in this range yet.")
    score = 50.0
else:
    tr["pnl"] = tr.apply(lambda r: (r["exit"]-r["entry"]) if r["side"]=="LONG" else (r["entry"]-r["exit"]), axis=1)
    winrate = 100.0 * (tr["pnl"] > 0).mean()
    st.metric("Win Rate (%)", f"{winrate:.1f}%")
    st.line_chart(tr["pnl"].cumsum())
    score = float(np.clip(winrate, 0, 100))

st.session_state.setdefault("scores", {})
st.session_state["scores"]["backtest"] = score
st.success("✅ Backtest score saved to Dashboard")
