# pages/Trend.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import ta
from src.data_engine_final import fetch_master as fetch_smart
from src.data_engine_v3 import fetch_master as fetch_smart

st.set_page_config(page_title="Trend Strength (EMA & RSI)", layout="wide")
st.title("📈 Trend Strength — EMA + RSI Engine")

# ---------------- Sidebar Controls ----------------
c1, c2, c3 = st.sidebar.columns([1.3, 1.2, 1.0])
with c1:
    symbol = st.text_input("Symbol", value="NSEBANK.NS")
with c2:
    period = st.selectbox("Period", ["2d", "5d", "7d", "14d", "1mo"], index=1)
with c3:
    interval = st.selectbox("Interval", ["5m", "15m", "1h", "1d"], index=1)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Market बंद हो तो 1D चुनें. Intraday TF live hours में best हैं.")

# ---------------- Data Fetch ----------------
with st.spinner("Fetching OHLCV data…"):
    df, used, msg = fetch_smart(symbol, prefer=(period, interval))

if msg: st.info(msg)
if df is None or df.empty:
    st.error("⚠️ No data fetched. Try another timeframe or wait for market hours.")
    st.stop()

# ensure lowercase columns
df = df.rename(columns={c: c.lower() for c in df.columns})
# ---------------- Indicators ----------------
df["ema20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
df["ema50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
idf = df.dropna()

if idf.empty:
    st.warning("Not enough candles after indicators.")
    st.stop()

# ---------------- Scoring ----------------
def trend_score(row):
    s = 0
    s += 2 if row["close"] > row["ema20"] else -2
    s += 3 if row["close"] > row["ema50"] else -3
    if row["rsi"] > 60: s += 3
    elif row["rsi"] < 40: s -= 3
    norm = ((s + 8) / 16) * 100
    return np.clip(norm, 0, 100)

idf["score"] = idf.apply(trend_score, axis=1)
final_score = float(idf["score"].iloc[-1])

bias = "BUY" if final_score >= 60 else "SELL" if final_score <= 40 else "NEUTRAL"

# ---------------- Charts ----------------
st.markdown("---")
left, right = st.columns([3, 1.2])

def draw_candles(ax, d, max_bars=150):
    d = d.tail(max_bars)
    x = np.arange(len(d))
    for i, (o, h, l, c) in enumerate(zip(d["open"], d["high"], d["low"], d["close"])):
        color = "#26a69a" if c >= o else "#ef5350"
        ax.plot([i, i], [l, h], color=color, linewidth=0.7)
        ax.add_patch(plt.Rectangle((i - 0.3, min(o, c)), 0.6, abs(c - o), color=color, alpha=0.9))
    ax.plot(x, d["ema20"], label="EMA20", linewidth=1.0)
    ax.plot(x, d["ema50"], label="EMA50", linewidth=1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(-1, len(d))

with left:
    st.subheader("Candlestick + EMA20/EMA50")
    fig, ax = plt.subplots(figsize=(11, 4))
    draw_candles(ax, idf)
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("RSI(14)")
    fig2, ax2 = plt.subplots(figsize=(5, 2.6))
    tail = idf.tail(200)
    ax2.plot(tail.index, tail["rsi"], linewidth=1.2, color="#64b5f6")
    ax2.axhline(70, color="#ef5350", linestyle="--")
    ax2.axhline(30, color="#26a69a", linestyle="--")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.25)
    st.pyplot(fig2, clear_figure=True)

# ---------------- Results ----------------
st.subheader("Trend Strength Meter (0–100)")
st.progress(int(final_score))
c1, c2, c3 = st.columns(3)
c1.metric("Trend Score", f"{final_score:.1f}")
c2.metric("RSI(14)", f"{idf['rsi'].iloc[-1]:.1f}")
c3.metric("Bias", bias)

if bias == "BUY": st.success("Uptrend continuation likely.")
elif bias == "SELL": st.error("Downtrend pressure dominant.")
else: st.info("Neutral / Range — wait for other confirmations.")

# ---------------- Save for Dashboard ----------------
st.session_state.setdefault("performance", {})
st.session_state["performance"]["trend"] = {
    "symbol": symbol,
    "tf": used[1] if used else interval,
    "final_score": float(final_score),
    "bias": bias,
    "used": used
}
st.success("✅ Trend module synced with Dashboard.")
