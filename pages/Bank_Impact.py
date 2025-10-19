# pages/Bank_Impact.py
# -------------------------------------------------------------
# Bank Leaders Impact (ADV) — Candlestick Micro Charts + Scores
# Auto Daily Fallback (W1) with visible note (F1)
# Saves final score to st.session_state["performance"]["bankimpact"]
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import ta
import mplfinance as mpf

from utils import fetch_smart  # returns: (df, used_tuple, message)

st.set_page_config(page_title="Bank Impact", layout="wide", initial_sidebar_state="expanded")
st.title("🏦 Bank Leaders Sentiment — Impact on BankNifty")

# -------------------- Controls --------------------
c1, c2, c3 = st.columns([1.6, 1.2, 1.1])
with c1:
    st.caption("Analyzing heavyweights: HDFCBANK, ICICIBANK, KOTAKBANK, AXISBANK, SBIN")
with c2:
    data_mode = st.radio("Data Mode", ["🔴 Live Intraday", "🟦 Offline / Daily (Safe)"], index=0, horizontal=True)
with c3:
    tf = st.selectbox("Timeframe", ["5m", "15m", "60m", "1d"], index=0)

TF_PREF = {
    "5m":  ("5d",  "5m"),
    "15m": ("14d", "15m"),
    "60m": ("60d", "60m"),
    "1d":  ("6mo", "1d"),
}
prefer = TF_PREF.get(tf, ("5d", "5m"))
# Safe mode or 1D -> force daily
if data_mode.startswith("🟦") or tf == "1d":
    prefer = ("6mo", "1d")

BANKS = [
    ("HDFC Bank",  "HDFCBANK.NS"),
    ("ICICI Bank", "ICICIBANK.NS"),
    ("Kotak Bank", "KOTAKBANK.NS"),
    ("Axis Bank",  "AXISBANK.NS"),
    ("SBI",        "SBIN.NS"),
]

# -------------------- Helpers --------------------
@st.cache_data(ttl=180)
def _cached_fetch(ticker: str, prefer_tuple: tuple[str, str]):
    """Cache wrapper around utils.fetch_smart()"""
    return fetch_smart(ticker, prefer=prefer_tuple)

def _get_with_fallback(ticker: str, prefer_tuple: tuple[str, str]):
    """
    Try preferred (intraday). If no data -> auto fallback to daily.
    Returns df, used, msg, used_fallback(bool)
    """
    df, used, msg = _cached_fetch(ticker, prefer_tuple)
    used_fallback = False
    if (df is None or df.empty) or (used and used[1] != "1d" and len(df) < 60):
        # fallback to daily (6mo, 1d)
        df2, used2, msg2 = _cached_fetch(ticker, ("6mo", "1d"))
        if df2 is not None and not df2.empty:
            return df2, used2, "Using Daily fallback (market likely closed).", True
        # if fallback also failed, return original
        return df, used, msg or msg2, used_fallback
    return df, used, msg, used_fallback

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ta.trend.EMAIndicator(close=out["close"], window=20).ema_indicator()
    out["ema50"] = ta.trend.EMAIndicator(close=out["close"], window=50).ema_indicator()
    out["rsi"]   = ta.momentum.RSIIndicator(close=out["close"], window=14).rsi()
    return out.dropna()

def _bank_score(last_row: pd.Series, frame: pd.DataFrame) -> float:
    # A) EMA structure (0..40)
    ema_pts = 0.0
    ema_pts += 18.0 if last_row["close"] > last_row["ema20"] else -8.0
    ema_pts += 22.0 if last_row["close"] > last_row["ema50"] else -10.0
    ema_pts = float(np.clip(ema_pts, 0.0, 40.0))

    # B) RSI (0..30) — scale 30..70 -> 0..30
    rsi = float(last_row["rsi"])
    rsi_scaled = (rsi - 30.0) / 40.0
    rsi_pts = float(np.clip(rsi_scaled, 0.0, 1.0) * 30.0)

    # C) Momentum last 20 closes (0..30)
    tail = frame["close"].tail(20).reset_index(drop=True)
    x = np.arange(len(tail), dtype=float)
    if len(tail) >= 5:
        xm, ym = x.mean(), tail.mean()
        num = float(((x - xm) * (tail - ym)).sum())
        den = float(((x - xm) ** 2).sum()) + 1e-9
        slope = num / den
        norm = slope / max(1.0, tail.iloc[-1])
        mom_pts = float(np.clip((norm * 2000.0) + 15.0, 0.0, 30.0))
    else:
        mom_pts = 10.0

    return round(float(np.clip(ema_pts + rsi_pts + mom_pts, 0.0, 100.0)), 2)

def _mini_candles(df_plot: pd.DataFrame, title: str):
    d = df_plot.rename(columns={
        "open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"
    }).copy()
    d.index.name = "Date"
    d = d.tail(150)

    addplots = [
        mpf.make_addplot(d["Close"].ewm(span=20, adjust=False).mean(), width=0.8),
        mpf.make_addplot(d["Close"].ewm(span=50, adjust=False).mean(), width=0.8),
    ]
    style = mpf.make_mpf_style(base_mpf_style="yahoo")
    fig = mpf.figure(style=style, figsize=(5.2, 2.6))
    ax = fig.add_subplot(1,1,1)
    mpf.plot(d, type="candle", volume=False, addplot=addplots, ax=ax, axisoff=True, xrotation=0)
    ax.set_title(title, fontsize=10, pad=2)
    st.pyplot(fig, clear_figure=True)

# -------------------- Fetch + Render --------------------
rows, final_scores = [], []
any_fallback_used = False

with st.spinner("Fetching bank data & computing scores…"):
    for bank_name, ticker in BANKS:
        df, used, note, used_fallback = _get_with_fallback(ticker, prefer)
        any_fallback_used = any_fallback_used or used_fallback

        if df is None or df.empty or len(df) < 60:
            st.info(f"**{bank_name}** — Market data unavailable. Please try again later.")
            rows.append({"Bank": bank_name, "Ticker": ticker, "RSI": "-", "EMA20": "-", "EMA50": "-", "Close": "-",
                         "Score": 0.0, "status": "No data"})
            continue

        idf = _add_indicators(df)
        if idf.empty:
            st.info(f"**{bank_name}** — Not enough candles for indicators.")
            rows.append({"Bank": bank_name, "Ticker": ticker, "RSI": "-", "EMA20": "-", "EMA50": "-", "Close": "-",
                         "Score": 0.0, "status": "No indicators"})
            continue

        last = idf.iloc[-1]
        score = _bank_score(last, idf)

        chart_col, info_col = st.columns([2, 1])
        with chart_col:
            label = f"{bank_name}  ({ticker})"
            # Add tag to title if daily fallback was used for this bank
            if used_fallback:
                label += "  • 1D"
            _mini_candles(idf, label)
        with info_col:
            st.metric("Close", f"{last['close']:.2f}")
            st.metric("RSI(14)", f"{last['rsi']:.1f}")
            st.metric("EMA20 / EMA50", f"{last['ema20']:.1f} / {last['ema50']:.1f}")
            if (last['close'] > last['ema20'] and last['close'] > last['ema50'] and last['rsi'] >= 55):
                bias = "Bullish"
            elif (last['close'] < last['ema20'] and last['close'] < last['ema50'] and last['rsi'] <= 45):
                bias = "Bearish"
            else:
                bias = "Neutral"
            st.metric("Bias", f"{bias}  |  Score: {score:.0f}/100")

        rows.append({
            "Bank": bank_name,
            "Ticker": ticker,
            "RSI": round(float(last["rsi"]), 1),
            "EMA20": round(float(last["ema20"]), 2),
            "EMA50": round(float(last["ema50"]), 2),
            "Close": round(float(last["close"]), 2),
            "Score": score,
            "status": "OK" if not used_fallback else "OK (1D)"
        })
        final_scores.append(score)

# F1 — visible fallback note (one banner, top of summary)
if any_fallback_used:
    st.warning("⚠️ Intraday data unavailable for one or more banks — **using Daily OHLC fallback** (auto).")

st.markdown("---")

# -------------------- Summary table + Final score --------------------
df_table = pd.DataFrame(rows)
st.subheader("Summary Table")
st.dataframe(df_table, use_container_width=True)

if len(final_scores) == 0:
    bank_impact = 0.0
    st.warning("No valid bank data to compute a final impact score.")
else:
    bank_impact = float(np.mean(final_scores))
    st.subheader(f"🧮 Final Bank Impact Score: **{bank_impact:.0f} / 100**")
    if bank_impact >= 65:
        st.success("Leaders indicate **Bullish** bias on BankNifty.")
    elif bank_impact <= 35:
        st.error("Leaders indicate **Bearish** bias on BankNifty.")
    else:
        st.info("Leaders indicate **Neutral / Mixed** conditions.")

# -------------------- Save for Dashboard Fusion --------------------
st.session_state.setdefault("performance", {})
st.session_state["performance"]["bankimpact"] = {
    "tf": tf,
    "mode": "intraday" if data_mode.startswith("🔴") else "daily",
    "prefer_period": prefer[0],
    "prefer_interval": prefer[1],
    "banks": rows,
    "final_score": float(bank_impact),
}
st.success("✅ Bank Impact score saved for Dashboard fusion.")
