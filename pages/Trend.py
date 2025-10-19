# pages/Trend.py
# -------------------------------------------------------------
# Trend Analysis (Multi-TF) — BankNifty
# • Weekend-safe: uses last daily data when market is closed.
# • Pro view: Candles + Volume + EMA20/EMA50 + RSI.
# • Score (0–100): EMA/RSI blend per timeframe + optional multi-TF.
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import ta  # pip install ta
import mplfinance as mpf
import matplotlib.pyplot as plt

from utils import DEFAULT_SYMBOL, fetch_smart

st.set_page_config(page_title="Trend", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Trend Analysis — BankNifty")

# -------------------- Controls --------------------
left, mid, right = st.columns([1.6, 1.1, 1.1])
with left:
    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
with mid:
    # Timeframe choice drives period/interval suggestion
    tf = st.selectbox(
        "Timeframe",
        options=["Daily (1D)", "1H", "15m", "5m"],
        index=0,
        help="Select the timeframe to analyze.",
    )
with right:
    data_mode = st.radio(
        "Data Mode",
        options=["🔴 Live Intraday", "🟦 Offline / Daily (Safe)"],
        index=1 if "Daily" in tf else 0,
        horizontal=True,
    )

# Map timeframe → (period, interval) preferences
TF_PREF = {
    "Daily (1D)": ("3mo", "1d"),
    "1H": ("60d", "60m"),
    "15m": ("14d", "15m"),
    "5m": ("5d", "5m"),
}
prefer = TF_PREF.get(tf, ("3mo", "1d"))

# If user picked Daily Safe OR weekend: fetch_smart will message accordingly.
with st.spinner("Fetching price data…"):
    df, used, msg = fetch_smart(
        symbol,
        prefer=("3mo", "1d") if data_mode.startswith("🟦") or "Daily" in tf else prefer
    )
if msg:
    st.info(msg)
if df is None or df.empty:
    st.warning("No data available for this selection right now.")
    st.stop()

st.caption(f"Using data: period={used[0]} • interval={used[1]}")

# -------------------- Indicators & Score --------------------
def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ema20"] = ta.trend.EMAIndicator(out["close"], window=20).ema_indicator()
    out["ema50"] = ta.trend.EMAIndicator(out["close"], window=50).ema_indicator()
    out["rsi"] = ta.momentum.RSIIndicator(out["close"], window=14).rsi()
    return out.dropna()

def trend_score(row: pd.Series) -> int:
    score = 0
    # Structure & momentum mix
    score += 2 if row["close"] > row["ema20"] else -2
    score += 3 if row["close"] > row["ema50"] else -3
    if row["rsi"] >= 60:
        score += 3
    elif row["rsi"] <= 40:
        score -= 3
    # Clamp roughly to [-8, +8] → normalize 0..100
    return score

idf = add_indicators(df)
if idf.empty:
    st.warning("Not enough candles for indicators. Try a longer period.")
    st.stop()

latest = idf.iloc[-1]
raw = trend_score(latest)
normalized = int(np.clip(((raw + 8) / 16) * 100, 0, 100))

colA, colB, colC, colD = st.columns(4)
colA.metric("Close", f"{latest['close']:.2f}")
colB.metric("EMA20 / EMA50", f"{latest['ema20']:.1f} / {latest['ema50']:.1f}")
colC.metric("RSI (14)", f"{latest['rsi']:.1f}")
bias = "Bullish" if normalized >= 60 else ("Bearish" if normalized <= 40 else "Neutral")
colD.metric("Trend Bias", f"{bias} ({normalized}/100)")

st.progress(normalized)

# -------------------- Pro Candlestick (EMA + Volume + RSI) --------------------
def plot_candles(frame: pd.DataFrame):
    d = frame.copy()
    d = d.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    d.index.name = "Date"

    add_plots = [
        mpf.make_addplot(d["Close"].ewm(span=20, adjust=False).mean(), color="tab:blue", width=1.0),
        mpf.make_addplot(d["Close"].ewm(span=50, adjust=False).mean(), color="tab:orange", width=1.0),
    ]
    # RSI on separate panel
    rsi = ta.momentum.RSIIndicator(d["Close"], 14).rsi()
    add_plots.append(mpf.make_addplot(rsi, panel=2, ylabel="RSI", width=1.0))

    style = mpf.make_mpf_style(base_mpf_style="yahoo")
    mpf.plot(
        d.tail(300),                      # limit for speed
        type="candle",
        volume=True,
        addplot=add_plots,
        figratio=(18, 9),
        figscale=1.2,
        style=style,
        panel_ratios=(5, 1, 2),          # price, volume, rsi
        ylabel="Price",
        ylabel_lower="Volume",
        datetime_format="%d-%b %H:%M" if used[1] != "1d" else "%d-%b",
        xrotation=0,
    )

st.subheader("Candles + Volume + EMA20/50 + RSI")
fig = mpf.figure(style="yahoo", figsize=(14, 7))
plt.close(fig)  # avoid duplicate render in Streamlit
plot_candles(idf)
st.pyplot(plt.gcf(), clear_figure=True)

# -------------------- Notes --------------------
with st.expander("How is the score computed?", expanded=False):
    st.write(
        """
- **EMA structure**: Close above EMA20 (+2), above EMA50 (+3). Below them: negative points.
- **RSI**: RSI ≥ 60 (+3), RSI ≤ 40 (–3).
- Raw score is normalized to **0–100** for the progress bar.
- This is a quick bias meter for the chosen timeframe.
        """
    )

# Save a tiny summary for Dashboard fusion (not heavy)
st.session_state.setdefault("performance", {})
st.session_state["performance"]["trend"] = {
    "tf": tf,
    "score": int(normalized),
    "bias": bias,
}
st.success("✅ Trend score saved for Dashboard fusion.")
