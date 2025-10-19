# pages/Bank_Impact.py
# -------------------------------------------------------------
# Bank Leaders Impact (ADV) — Candlestick Micro Charts + Scores
# • Weekend-safe via utils.fetch_smart()
# • Analyzes top 5 banks: HDFC, ICICI, KOTAK, AXIS, SBI
# • For each bank:
#     - Mini candlestick chart (last 150 bars)
#     - EMA20 / EMA50, RSI(14)
#     - Score (0–100) from EMA structure, RSI strength, short-term momentum
# • Final "Bank Impact Score" (avg of banks) saved to Dashboard fusion:
#     st.session_state["performance"]["bankimpact"]
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt

from utils import fetch_smart  # weekend-safe data fetch

st.set_page_config(page_title="Bank Impact", layout="wide", initial_sidebar_state="expanded")
st.title("🏦 Bank Leaders Sentiment — Impact on BankNifty")

# -------------------- Controls --------------------
c1, c2, c3 = st.columns([1.6, 1.1, 1.1])
with c1:
    st.caption("Analyzing heavyweights: HDFCBANK, ICICIBANK, KOTAKBANK, AXISBANK, SBIN")
with c2:
    data_mode = st.radio("Data Mode", ["🔴 Live Intraday", "🟦 Offline / Daily (Safe)"], index=0, horizontal=True)
with c3:
    tf = st.selectbox("Timeframe", ["5m", "15m", "60m", "1d"], index=0)

TF_PREF = {
    "5m": ("5d", "5m"),
    "15m": ("14d", "15m"),
    "60m": ("60d", "60m"),
    "1d": ("6mo", "1d"),
}
prefer = TF_PREF.get(tf, ("5d", "5m"))
if data_mode.startswith("🟦") or tf == "1d":
    # Force reliable daily on safe mode
    prefer = ("6mo", "1d")

BANKS = [
    ("HDFC Bank",  "HDFCBANK.NS"),
    ("ICICI Bank", "ICICIBANK.NS"),
    ("Kotak Bank", "KOTAKBANK.NS"),
    ("Axis Bank",  "AXISBANK.NS"),
    ("SBI",        "SBIN.NS"),
]

# -------------------- Helpers --------------------
def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ta.trend.EMAIndicator(close=out["close"], window=20).ema_indicator()
    out["ema50"] = ta.trend.EMAIndicator(close=out["close"], window=50).ema_indicator()
    out["rsi"]   = ta.momentum.RSIIndicator(close=out["close"], window=14).rsi()
    return out.dropna()

def _bank_score(last_row: pd.Series, frame: pd.DataFrame) -> float:
    """
    Score components:
      A) EMA structure (40 pts): close>ema20 (+18) & close>ema50 (+22), else negatives
      B) RSI strength (30 pts): scale 30..70 → 0..30 (clipped)
      C) Short momentum (30 pts): slope of last ~20 closes, scaled to 0..30
    """
    # A) EMA structure
    ema_pts = 0.0
    if last_row["close"] > last_row["ema20"]:
        ema_pts += 18.0
    else:
        ema_pts -= 8.0
    if last_row["close"] > last_row["ema50"]:
        ema_pts += 22.0
    else:
        ema_pts -= 10.0
    ema_pts = np.clip(ema_pts, 0, 40)

    # B) RSI strength
    rsi = float(last_row["rsi"])
    rsi_scaled = (rsi - 30.0) / 40.0  # 30→0, 70→1
    rsi_pts = float(np.clip(rsi_scaled, 0.0, 1.0) * 30.0)

    # C) Momentum (linear slope on last 20 closes normalized by price)
    tail = frame["close"].tail(20).reset_index(drop=True)
    x = np.arange(len(tail), dtype=float)
    if len(tail) >= 5:
        # simple linear regression slope
        x_mean = x.mean()
        y_mean = tail.mean()
        num = float(((x - x_mean) * (tail - y_mean)).sum())
        den = float(((x - x_mean) ** 2).sum()) + 1e-9
        slope = num / den
        norm_slope = slope / max(1.0, tail.iloc[-1])  # normalize by price
        # scale to 0..30 with gentle sensitivity
        mom_pts = float(np.clip((norm_slope * 2000.0) + 15.0, 0.0, 30.0))
    else:
        mom_pts = 10.0

    total = float(np.clip(ema_pts + rsi_pts + mom_pts, 0.0, 100.0))
    return round(total, 2)

def _plot_mini_candles(df_plot: pd.DataFrame, title: str):
    """
    Render a tiny candlestick with volume and EMA overlays.
    Uses last ~150 bars for speed/clarity.
    """
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
    mpf.plot(
        d, type="candle", volume=False, addplot=addplots,
        ax=ax, axisoff=True, xrotation=0, datetime_format="%d-%b" if len(d)>0 else "%H:%M"
    )
    ax.set_title(title, fontsize=10, pad=2)
    st.pyplot(fig, clear_figure=True)

# -------------------- Fetch, analyze, render --------------------
rows = []
final_scores = []

with st.spinner("Fetching bank data & computing scores…"):
    for bank_name, ticker in BANKS:
        # Weekend-safe fetch
        df, used, msg = fetch_smart(ticker, prefer=prefer)
        if msg:
            st.info(f"{bank_name}: {msg}")
        if df is None or df.empty or len(df) < 60:
            rows.append({
                "Bank": bank_name, "Ticker": ticker,
                "RSI": "-", "EMA20": "-", "EMA50": "-", "Close": "-",
                "Score": 0.0, "status": "No data"
            })
            # Even if no data, draw an empty placeholder container for consistent UI
            with st.container():
                st.write(f"**{bank_name}** — no data")
            continue

        idf = _add_indicators(df)
        if idf.empty:
            rows.append({
                "Bank": bank_name, "Ticker": ticker,
                "RSI": "-", "EMA20": "-", "EMA50": "-", "Close": "-",
                "Score": 0.0, "status": "No indicators"
            })
            with st.container():
                st.write(f"**{bank_name}** — not enough candles for indicators")
            continue

        last = idf.iloc[-1]
        score = _bank_score(last, idf)

        # Layout: mini chart + metrics side-by-side
        chart_col, info_col = st.columns([2, 1])
        with chart_col:
            _plot_mini_candles(idf, f"{bank_name}  ({ticker})")
        with info_col:
            st.metric("Close", f"{last['close']:.2f}")
            st.metric("RSI(14)", f"{last['rsi']:.1f}")
            st.metric("EMA20 / EMA50", f"{last['ema20']:.1f} / {last['ema50']:.1f}")
            bias = "Bullish" if (last['close'] > last['ema20'] and last['close'] > last['ema50'] and last['rsi'] >= 55) else ("Bearish" if (last['close'] < last['ema20'] and last['close'] < last['ema50'] and last['rsi'] <= 45) else "Neutral")
            st.metric("Bias", f"{bias}  |  Score: {score:.0f}/100")

        rows.append({
            "Bank": bank_name,
            "Ticker": ticker,
            "RSI": round(float(last["rsi"]), 1),
            "EMA20": round(float(last["ema20"]), 2),
            "EMA50": round(float(last["ema50"]), 2),
            "Close": round(float(last["close"]), 2),
            "Score": score,
            "status": "OK"
        })
        final_scores.append(score)

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
    "banks": rows,                      # per-bank metrics for AI learning later
    "final_score": float(bank_impact),  # 0..100
}

st.success("✅ Bank Impact score saved for Dashboard fusion.")
