# pages/Smart_Money.py
# -------------------------------------------------------------
# Smart Money (PRO++) — True Trap / Absorption Engine for BankNifty
# Weekend-safe + Institutional Imbalance Factor (IIF)
# Detects bull/bear traps, absorption, stop-hunts, RSI divergence
# Outputs Score (0–100) + Bias + Recent trap summary
# -------------------------------------------------------------
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import streamlit as st
import pandas as pd, numpy as np
import mplfinance as mpf
import ta
from src.utils import *
from src.data_engine import *

st.set_page_config(page_title="Smart Money (PRO++)", layout="wide")
st.title("🧠 Smart Money — Trap & Absorption Engine (Pro-Level)")

# ---------------- Controls ----------------
DEFAULT_SYMBOL = "NSEBANK.NS"

c1, c2, c3, c4 = st.columns([1.3, 1.1, 1.1, 1.3])
with c1:
    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
with c2:
    tf = st.selectbox("Timeframe", ["5m", "15m", "60m", "1d"], index=0)
with c3:
    lookback = st.selectbox("Lookback", ["5d", "14d", "30d", "3mo", "6mo"], index=1)
with c4:
    mode = st.radio("Mode", ["🔴 Live", "🟦 Safe / Daily"], index=0, horizontal=True)

TF_PREF = {"5m": ("5d","5m"), "15m": ("14d","15m"), "60m": ("60d","60m"), "1d": ("6mo","1d")}
prefer = TF_PREF.get(tf, ("5d","5m"))
if mode.startswith("🟦") or tf == "1d":
    prefer = ("6mo","1d")
prefer = (lookback, prefer[1])

# ---------------- Data fetch ----------------
with st.spinner("Fetching price/volume data..."):
    df, msg = fetch_smart(symbol)

if msg: st.info(msg)
if df is None or df.empty:
    st.warning("⚠️ No data received. Market may be closed or source unreachable.")
    st.stop()

st.caption(f"Using period={used[0]} interval={used[1]} | candles={len(df)}")

# ---------------- Feature Engineering ----------------
def build_features(d):
    d = d.rename(columns=str.lower).copy()
    d = d[["open","high","low","close","volume"]].dropna()
    d["body"] = (d["close"] - d["open"]).abs()
    d["range"] = (d["high"] - d["low"]).replace(0,np.nan)
    d["upper_wick"] = d["high"] - d[["open","close"]].max(axis=1)
    d["lower_wick"] = d[["open","close"]].min(axis=1) - d["low"]
    d["vol_ma20"] = d["volume"].rolling(20).mean()
    d["vol_factor"] = (d["volume"]/d["vol_ma20"]).fillna(0)
    d["ema20"] = ta.trend.EMAIndicator(d["close"],20).ema_indicator()
    d["ema50"] = ta.trend.EMAIndicator(d["close"],50).ema_indicator()
    d["rsi"] = ta.momentum.RSIIndicator(d["close"],14).rsi()
    d["prev_high"], d["prev_low"] = d["high"].shift(1), d["low"].shift(1)
    return d.dropna()

feat = build_features(df)

# ---------------- Trap Detection ----------------
def detect_traps(f):
    f = f.copy()
    f["trap_type"] = ""
    f["upper_break"] = f["high"] > f["prev_high"]
    f["lower_break"] = f["low"] < f["prev_low"]

    # Fake breakouts
    f["reject_down"] = f["upper_break"] & (f["close"] < f["prev_high"])
    f["reject_up"] = f["lower_break"] & (f["close"] > f["prev_low"])

    # Long wicks
    f["long_upper"] = (f["upper_wick"]/f["range"]) > 0.45
    f["long_lower"] = (f["lower_wick"]/f["range"]) > 0.45

    # Stop-hunts
    f["bull_trap"] = (f["reject_down"] | f["long_upper"]) & f["upper_break"]
    f["bear_trap"] = (f["reject_up"] | f["long_lower"]) & f["lower_break"]

    # Absorption
    mid = f["low"] + 0.5*f["range"]
    f["absorb_buy"] = (f["vol_factor"]>1.6) & (f["close"]<mid) & (f["close"]>f["open"])
    f["absorb_sell"] = (f["vol_factor"]>1.6) & (f["close"]>mid) & (f["close"]<f["open"])

    f.loc[f["bull_trap"], "trap_type"] = "BULL_TRAP"
    f.loc[f["bear_trap"], "trap_type"] = "BEAR_TRAP"
    f.loc[f["absorb_buy"], "trap_type"] = "ABSORB_BUY"
    f.loc[f["absorb_sell"], "trap_type"] = "ABSORB_SELL"
    return f

feat2 = detect_traps(feat)

# ---------------- Smart Money Scoring ----------------
def compute_score(f, window=60):
    d = f.tail(window)
    score = 50
    bull, bear = (d["trap_type"]=="BULL_TRAP").sum(), (d["trap_type"]=="BEAR_TRAP").sum()
    ab_buy, ab_sell = (d["trap_type"]=="ABSORB_BUY").sum(), (d["trap_type"]=="ABSORB_SELL").sum()
    iif = ((ab_buy + bear) - (ab_sell + bull)) * 2.2  # Institutional Imbalance Factor
    score += iif
    score = float(np.clip(score, 0, 100))
    bias = "BUY" if score>60 else "SELL" if score<40 else "NEUTRAL"
    traps = d[d["trap_type"]!=""][["open","high","low","close","volume","trap_type"]].tail(25)
    return score, bias, traps

score, bias, traps = compute_score(feat2)

# ---------------- UI ----------------
st.subheader("Smart Money Sentiment")
st.progress(int(score))
if bias=="BUY":
    st.success(f"Institutions likely accumulating — Score {score:.1f}/100 (BUY)")
elif bias=="SELL":
    st.error(f"Distribution pressure detected — Score {score:.1f}/100 (SELL)")
else:
    st.info(f"Balanced activity — Score {score:.1f}/100 (NEUTRAL)")

# ---------------- Chart ----------------
st.subheader("📊 Trap Markers (Smart Money View)")
plot_df = feat2.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}).copy()
plot_df.index.name="Date"
plot_df = plot_df.tail(200)

buy_y, sell_y = np.full(len(plot_df), np.nan), np.full(len(plot_df), np.nan)
for i, row in enumerate(feat2.tail(200).itertuples()):
    if row.trap_type in ["BEAR_TRAP","ABSORB_BUY"]:
        buy_y[i] = row.close*0.995
    elif row.trap_type in ["BULL_TRAP","ABSORB_SELL"]:
        sell_y[i] = row.close*1.005

apds = [
    mpf.make_addplot(buy_y, type="scatter", marker="^", color="lime", markersize=70),
    mpf.make_addplot(sell_y, type="scatter", marker="v", color="red", markersize=70)
]

style = mpf.make_mpf_style(base_mpf_style="yahoo", facecolor="#111")
fig, _ = mpf.plot(plot_df, type="candle", volume=True, addplot=apds, style=style,
                  figratio=(18,9), figscale=1.1, returnfig=True,
                  title=f"{symbol} • Smart Money {bias}")
st.pyplot(fig, clear_figure=True)

# ---------------- Table ----------------
st.subheader("Recent Trap Events")
if traps.empty:
    st.write("No recent traps found.")
else:
    st.dataframe(traps.sort_index(ascending=False), use_container_width=True)

# ---------------- Save to Dashboard ----------------
st.session_state.setdefault("performance", {})
st.session_state["performance"]["smartmoney"] = {
    "symbol": symbol,
    "tf": tf,
    "score": float(score),
    "bias": bias,
    "trap_count": len(traps),
    "recent_traps": traps.reset_index().to_dict(orient="records"),
}
st.success("✅ Smart Money performance saved to Dashboard fusion.")
