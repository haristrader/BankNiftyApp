# pages/Bank_Impact.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import streamlit as st
import pandas as pd, numpy as np, ta, mplfinance as mpf, datetime
from src.utils import *

st.set_page_config(page_title="Bank Impact", layout="wide", initial_sidebar_state="expanded")
st.title("🏦 Bank Leaders Sentiment — Impact on BankNifty")

c1, c2, c3 = st.columns([1.6, 1.2, 1.1])
with c1:
    st.caption("Tracking heavyweights: HDFC, ICICI, KOTAK, AXIS, SBI")
with c2:
    data_mode = st.radio("Mode", ["🔴 Live Intraday", "🟦 Safe / Daily"], index=0, horizontal=True)
with c3:
    tf = st.selectbox("Timeframe", ["5m", "15m", "60m", "1d"], index=0)

TF_PREF = {"5m": ("5d", "5m"), "15m": ("14d", "15m"), "60m": ("60d", "60m"), "1d": ("6mo", "1d")}
prefer = TF_PREF.get(tf, ("5d", "5m"))
if data_mode.startswith("🟦") or tf == "1d":
    prefer = ("6mo", "1d")

BANKS = [
    ("HDFC Bank", "HDFCBANK.NS"),
    ("ICICI Bank", "ICICIBANK.NS"),
    ("Kotak Bank", "KOTAKBANK.NS"),
    ("Axis Bank", "AXISBANK.NS"),
    ("SBI", "SBIN.NS"),
]

@st.cache_data(ttl=180)
def _cached_fetch(ticker, prefer_tuple):
    try:
        return fetch_smart(ticker, prefer=prefer_tuple)
    except Exception as e:
        return pd.DataFrame(), prefer_tuple, f"Error fetching {ticker}: {e}"

def _get_with_fallback(ticker, prefer_tuple):
    df, used, msg = _cached_fetch(ticker, prefer_tuple)
    if df is None or df.empty:
        df2, used2, msg2 = _cached_fetch(ticker, ("6mo", "1d"))
        if df2 is not None and not df2.empty:
            return df2, used2, f"Using daily fallback ({ticker})", True
    return df, used, msg, False

def _add_indicators(df):
    out = df.copy()
    out = out.rename(columns={c: c.lower() for c in out.columns})
    out["ema20"] = ta.trend.EMAIndicator(out["close"], 20).ema_indicator()
    out["ema50"] = ta.trend.EMAIndicator(out["close"], 50).ema_indicator()
    out["rsi"] = ta.momentum.RSIIndicator(out["close"], 14).rsi()
    return out.dropna()

def _bank_score(last_row, frame):
    ema_pts = 0
    ema_pts += 18 if last_row["close"] > last_row["ema20"] else -8
    ema_pts += 22 if last_row["close"] > last_row["ema50"] else -10
    ema_pts = np.clip(ema_pts, 0, 40)
    rsi_scaled = (last_row["rsi"] - 30) / 40
    rsi_pts = np.clip(rsi_scaled, 0, 1) * 30
    tail = frame["close"].tail(20)
    if len(tail) >= 5:
        slope = np.polyfit(range(len(tail)), tail, 1)[0]
        norm = slope / (np.std(tail) if np.std(tail) != 0 else 1)
        mom_pts = np.clip(norm * 12 + 15, 0, 30)
    else:
        mom_pts = 10
    return round(float(np.clip(ema_pts + rsi_pts + mom_pts, 0, 100)), 2)

def _mini_chart(df, title):
    df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"})
    df = df.tail(120)
    style = mpf.make_mpf_style(base_mpf_style="yahoo", rc={"axes.facecolor":"#0E1117"})
    fig = mpf.figure(style=style, figsize=(4.8, 2.2))
    ax = fig.add_subplot(1,1,1)
    mpf.plot(df, type="candle", ax=ax, volume=False)
    ax.set_title(title, fontsize=9, color="white", pad=3)
    st.pyplot(fig, clear_figure=True)

rows, scores, any_fallback = [], [], False
with st.spinner("Fetching bank data…"):
    for name, ticker in BANKS:
        df, used, msg, fallback = _get_with_fallback(ticker, prefer)
        any_fallback |= fallback
        if df is None or df.empty:
            st.info(f"{name} — Data unavailable.")
            continue

        idf = _add_indicators(df)
        if idf.empty:
            st.warning(f"{name} — Insufficient candles.")
            continue

        last = idf.iloc[-1]
        score = _bank_score(last, idf)
        bias = "Bullish" if score >= 60 else "Bearish" if score <= 40 else "Neutral"

        with st.expander(f"{name} ({ticker}) — {bias} | {score:.0f}/100"):
            col1, col2 = st.columns([2, 1])
            with col1: _mini_chart(idf, name)
            with col2:
                st.metric("Close", f"{last['close']:.2f}")
                st.metric("RSI(14)", f"{last['rsi']:.1f}")
                st.metric("EMA20/EMA50", f"{last['ema20']:.1f} / {last['ema50']:.1f}")
                st.metric("Bias", bias)

        rows.append({"Bank": name, "Ticker": ticker, "Score": score, "Bias": bias})
        scores.append(score)

if any_fallback:
    st.warning("⚠️ Using Daily fallback data (market closed).")

if len(scores) == 0:
    st.error("No valid data to compute final Bank Impact.")
    st.stop()

bank_score = float(np.mean(scores))
st.subheader(f"🏦 Final Bank Impact Score: **{bank_score:.0f}/100**")

if bank_score >= 65: st.success("Strong Bullish alignment by leaders.")
elif bank_score <= 35: st.error("Bearish tone across leaders.")
else: st.info("Mixed or neutral sentiment.")

st.session_state.setdefault("performance", {})
st.session_state["performance"]["bankimpact"] = {
    "timestamp": datetime.datetime.now().isoformat(),
    "final_score": float(bank_score),
    "mode": "live" if data_mode.startswith("🔴") else "daily",
    "banks": rows,
    "used": used
}

try:
    from src.data_engine import sb_record_module_score
    sb_record_module_score("bankimpact", float(bank_score), bias="auto", symbol="NSEBANK.NS", tf=tf)
    st.success("✅ Synced with Dashboard + AI Console.")
except Exception:
    st.info("⚙️ Local mode: Supabase sync skipped.")
