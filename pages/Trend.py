# pages/Trend.py
# -------------------------------------------------------------
# Trend Strength (EMA & RSI) with Candlestick (Matplotlib)
# Symbol: NSEBANK.NS (default)
# Features:
#  - Intraday fetch with smart fallback to Daily (1D)
#  - EMA20/EMA50 + RSI(14)
#  - Candlestick chart without mplfinance (so no extra dependency)
#  - Silent Supabase saves (candles + module score)
#  - Exposes session_state["performance"]["trend"]["final_score"]
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

from utils import (
    fetch, DEFAULT_SYMBOL,
    sb_save_candles, sb_record_module_score
)
df, used, msg = fetch_smart("NSEBANK.NS", ("5d","5m"))
st.write("DEBUG:", msg)
st.dataframe(df.head())
st.set_page_config(page_title="Trend Strength (EMA & RSI)", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Trend Strength (EMA & RSI)")

# ---------------- Sidebar Controls ----------------
symbol = st.sidebar.text_input("Symbol", value="NSEBANK.NS")
period = st.sidebar.selectbox("Period", ["2d", "5d", "7d", "14d", "1mo", "3mo", "6mo"], index=2)
interval = st.sidebar.selectbox("Interval (TF)", ["5m", "15m", "60m", "1d"], index=1)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Market बंद हो तो 1D चुनें (safe). Intraday TFs live hours में best हैं.")

# ---------------- Helpers ----------------
def compute_trend_score(ind_row: pd.Series) -> float:
    score_raw = 0
    score_raw += 2 if ind_row["close"] > ind_row["ema20"] else -2
    score_raw += 3 if ind_row["close"] > ind_row["ema50"] else -3
    if ind_row["rsi"] > 60:
        score_raw += 3
    elif ind_row["rsi"] < 40:
        score_raw -= 3
    # normalize (-8..+8) -> 0..100
    norm = ((score_raw + 8.0) / 16.0) * 100.0
    return float(max(0.0, min(100.0, norm)))

def bias_from_score(score: float) -> str:
    if score >= 60: return "BUY"
    if score <= 40: return "SELL"
    return "NEUTRAL"

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ta.trend.EMAIndicator(out["close"], 20).ema_indicator()
    out["ema50"] = ta.trend.EMAIndicator(out["close"], 50).ema_indicator()
    out["rsi"]   = ta.momentum.RSIIndicator(out["close"], 14).rsi()
    return out.dropna()

def draw_candles(ax, cdf: pd.DataFrame, max_bars: int = 150):
    """
    Simple candlestick using matplotlib (no mplfinance).
    cdf must have columns: open, high, low, close and a DateTimeIndex.
    """
    d = cdf.tail(max_bars).copy()
    x = np.arange(len(d))
    ax.grid(True, alpha=0.25)

    # Wicks
    for i, (hi, lo) in enumerate(zip(d["high"].values, d["low"].values)):
        ax.vlines(i, lo, hi, linewidth=1, color="#888888")

    # Bodies
    width = 0.6
    for i, (o, c) in enumerate(zip(d["open"].values, d["close"].values)):
        color = "#26a69a" if c >= o else "#ef5350"  # green/red
        lower = min(o, c)
        height = abs(c - o)
        ax.add_patch(plt.Rectangle((i - width/2, lower), width, max(height, 1e-6), color=color, alpha=0.9))

    # X ticks as dates (sparse)
    ax.set_xlim(-1, len(d))
    xticks_idx = np.linspace(0, len(d)-1, num=min(6, len(d))).astype(int)
    ax.set_xticks(xticks_idx)
    ax.set_xticklabels([d.index[i].strftime("%d-%b %H:%M") if hasattr(d.index[i], "hour") else d.index[i].strftime("%d-%b")
                        for i in xticks_idx], rotation=0, ha="center", fontsize=8)

def smart_fetch(symbol: str, period: str, interval: str):
    """
    Try requested TF first. If empty -> fallback to 1D (daily, 6mo).
    Returns (df, used_tf, used_period, is_fallback)
    """
    df = fetch(symbol, period=period, interval=interval, auto_adjust=True)
    if df is not None and not df.empty and len(df) >= 30:
        return df, interval, period, False
    # fallback to daily
    df_d = fetch(symbol, period="6mo", interval="1d", auto_adjust=True)
    if df_d is not None and not df_d.empty and len(df_d) >= 60:
        return df_d, "1d", "6mo", True
    # final return
    return (pd.DataFrame(), interval, period, False)

# ---------------- Fetch Data (with fallback) ----------------
with st.spinner("Fetching OHLCV…"):
    df, used_tf, used_period, fell_back = smart_fetch(symbol, period, interval)

if df is None or df.empty:
    st.error("No/insufficient data. Try different Period/Interval or market hours.")
    st.stop()

# Indicators
idf = add_indicators(df)
if idf is None or idf.empty:
    st.error("Not enough candles after indicators. Try longer period.")
    st.stop()

last = idf.iloc[-1]
trend_score = compute_trend_score(last)
bias = bias_from_score(trend_score)

# ---------------- Supabase: silent saves ----------------
# Save used TF candles
try:
    sb_save_candles(df, symbol, used_tf)
except Exception:
    pass
# Also save daily for AI/history
try:
    daily_df = fetch(symbol, period="6mo", interval="1d", auto_adjust=True)
    if daily_df is not None and not daily_df.empty:
        sb_save_candles(daily_df, symbol, "1d")
except Exception:
    pass
# Record module score
try:
    sb_record_module_score(
        module="trend",
        score=float(trend_score),
        bias=bias,
        symbol=symbol,
        tf=used_tf,
        meta={"period": used_period, "note": "Trend page"}
    )
except Exception:
    pass

# ---------------- Header Badges ----------------
info_cols = st.columns(3)
with info_cols[0]:
    st.caption(f"Using: **period={used_period}**, **interval={used_tf}**")
with info_cols[1]:
    if fell_back:
        st.warning("Fallback: Intraday unavailable → showing **Daily (1D)**.")
    else:
        st.info("Live TF data in use.")
with info_cols[2]:
    st.caption("Candles include EMA20/EMA50 & RSI(14)")

# ---------------- Score & Stats ----------------
st.subheader("Trend Strength Meter (0–100)")
st.progress(int(trend_score))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Score", f"{trend_score:.0f}")
m2.metric("RSI(14)", f"{last['rsi']:.1f}")
m3.metric("EMA20", f"{last['ema20']:.1f}")
m4.metric("EMA50", f"{last['ema50']:.1f}")

if bias == "BUY":
    st.success("Bias: BUY — strength on the long side.")
elif bias == "SELL":
    st.error("Bias: SELL — strength on the short side.")
else:
    st.info("Bias: NEUTRAL — wait for clarity or other modules.")

# ---------------- Charts ----------------
st.markdown("---")
left, right = st.columns([3, 1.2])

with left:
    st.subheader("Candlesticks + EMA20/EMA50")
    fig, ax = plt.subplots(figsize=(11, 4))
    draw_candles(ax, idf, max_bars=150)
    # overlay EMAs
    tail = idf.tail(150)
    x = np.arange(len(tail))
    ax.plot(x, tail["ema20"].values, linewidth=1.2, label="EMA20")
    ax.plot(x, tail["ema50"].values, linewidth=1.2, label="EMA50")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=8)
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("RSI(14)")
    fig2, ax2 = plt.subplots(figsize=(5.2, 2.6))
    tail = idf.tail(200)
    ax2.plot(tail.index, tail["rsi"], linewidth=1.2)
    ax2.axhline(70, color="#ef5350", linewidth=1, linestyle="--")
    ax2.axhline(30, color="#26a69a", linewidth=1, linestyle="--")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.25)
    st.pyplot(fig2, clear_figure=True)

with st.expander("Recent data (last 80)"):
    st.dataframe(idf.tail(80)[["open","high","low","close","volume","ema20","ema50","rsi"]])

# ---------------- Save for Dashboard Fusion ----------------
st.session_state.setdefault("performance", {})
st.session_state["performance"]["trend"] = {
    "tf": used_tf,
    "mode": "daily" if used_tf == "1d" else "intraday",
    "final_score": float(trend_score),
    "bias": bias,
    "symbol": symbol,
}
st.caption("✅ Trend score saved for Dashboard fusion.")
