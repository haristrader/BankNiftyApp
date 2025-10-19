# pages/Smart_Money.py
# -------------------------------------------------------------
# Smart Money (PRO) — Trap Engine for BankNifty
# - Weekend-safe: auto daily fallback if intraday missing
# - Detects Bull/Bear Traps, Stop-hunts, Absorption
# - Volume anomaly + Wick structure + Breakout validity
# - Optional RSI divergence signal
# - Returns Score (0–100) + Bias + recent trap table
# - Saves to st.session_state["performance"]["smartmoney"]
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import ta
import mplfinance as mpf

from utils import fetch_smart  # (df, used, msg)

st.set_page_config(page_title="Smart Money (Trap Engine)", layout="wide", initial_sidebar_state="expanded")
st.title("🧠 Smart Money — Trap / Absorption Engine (PRO)")

# --------------------- Controls ---------------------
DEFAULT_SYMBOL = "NSEBANK.NS"  # stay consistent with your other pages

c1, c2, c3, c4 = st.columns([1.3, 1.1, 1.1, 1.3])
with c1:
    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
with c2:
    tf = st.selectbox("Timeframe", ["5m", "15m", "60m", "1d"], index=0)
with c3:
    lookback_days = st.selectbox("Lookback (approx.)", ["5d","14d","30d","3mo","6mo"], index=1)
with c4:
    data_mode = st.radio("Data Mode", ["🔴 Live Intraday", "🟦 Offline / Daily (Safe)"], index=0, horizontal=True)

TF_PREF = {
    "5m":  ("5d",  "5m"),
    "15m": ("14d", "15m"),
    "60m": ("60d", "60m"),
    "1d":  ("6mo", "1d"),
}
prefer = TF_PREF.get(tf, ("5d", "5m"))

# force safe daily when chosen or timeframe is daily
if data_mode.startswith("🟦") or tf == "1d":
    prefer = ("6mo", "1d")

# tune by lookback selector
period_override = {
    "5d": "5d", "14d": "14d", "30d": "30d", "3mo": "3mo", "6mo": "6mo"
}[lookback_days]
prefer = (period_override, prefer[1])

# ------------------ Fetch (with fallback) ------------------
@st.cache_data(ttl=180)
def _cached_fetch(t, p):
    return fetch_smart(t, prefer=p)

def _get_with_fallback(ticker, prefer_tuple):
    df, used, msg = _cached_fetch(ticker, prefer_tuple)
    used_fallback = False
    # fallback to daily if intraday is thin/absent
    if (df is None or df.empty) or (used and used[1] != "1d" and len(df) < 60):
        df2, used2, msg2 = _cached_fetch(ticker, ("6mo", "1d"))
        if df2 is not None and not df2.empty:
            return df2, used2, "Using Daily fallback (market likely closed).", True
        return df, used, msg or msg2, used_fallback
    return df, used, msg, used_fallback

with st.spinner("Fetching price/volume…"):
    df, used, info_msg, used_fallback = _get_with_fallback(symbol, prefer)

if info_msg:
    st.caption(f"ℹ️ {info_msg}")
if used_fallback:
    st.warning("⚠️ Intraday not available — switched to **Daily OHLC** auto fallback.")

if df is None or df.empty:
    st.error("No data received. Try different timeframe/period or market hours.")
    st.stop()

# ------------------ Feature Engineering ------------------
def build_features(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    # Ensure required columns lowercase
    out = out.rename(columns=str.lower)[["open","high","low","close","volume"]].dropna()

    # Price structure
    out["body"] = (out["close"] - out["open"]).abs()
    out["range"] = (out["high"] - out["low"]).replace(0, np.nan)
    out["upper_wick"] = out["high"] - out[["close","open"]].max(axis=1)
    out["lower_wick"] = out[["close","open"]].min(axis=1) - out["low"]
    out["wick_ratio"] = ((out["upper_wick"] + out["lower_wick"]) / out["range"]).fillna(0)

    # Averages
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    out["vol_factor"] = (out["volume"] / out["vol_ma20"]).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Indicators
    out["ema20"] = ta.trend.EMAIndicator(close=out["close"], window=20).ema_indicator()
    out["ema50"] = ta.trend.EMAIndicator(close=out["close"], window=50).ema_indicator()
    out["rsi"]    = ta.momentum.RSIIndicator(close=out["close"], window=14).rsi()

    # HH/LL refs for breakout checks (prev bar high/low)
    out["prev_high"] = out["high"].shift(1)
    out["prev_low"]  = out["low"].shift(1)

    # Simple ATR for scaling
    tr = ta.volatility.AverageTrueRange(
        high=out["high"], low=out["low"], close=out["close"], window=14
    ).average_true_range()
    out["atr"] = tr

    return out.dropna()

feat = build_features(df)
if feat.empty or len(feat) < 50:
    st.warning("Not enough candles to compute Smart Money metrics.")
    st.stop()

# ------------------ Trap / Absorption Logic ------------------
def detect_traps(f: pd.DataFrame) -> pd.DataFrame:
    t = f.copy()

    # Breakout conditions vs previous high/low
    t["upper_break"] = t["close"] > t["prev_high"]
    t["lower_break"] = t["close"] < t["prev_low"]

    # Rejection back inside range (fake breakout)
    t["reject_down"] = (t["upper_break"]) & (t["close"] < t["prev_high"])
    t["reject_up"]   = (t["lower_break"]) & (t["close"] > t["prev_low"])

    # Long wick definitions
    t["long_upper"] = (t["upper_wick"] / t["range"]) > 0.45
    t["long_lower"] = (t["lower_wick"] / t["range"]) > 0.45

    # Stop-hunt style: big wick (> ATR fraction) + close opposite third
    t["big_up_wick"]   = (t["upper_wick"] > 0.6 * t["atr"]) & (t["close"] < (t["low"] + 0.33 * t["range"]))
    t["big_down_wick"] = (t["lower_wick"] > 0.6 * t["atr"]) & (t["close"] > (t["high"] - 0.33 * t["range"]))

    # Volume anomaly
    t["hi_vol"] = t["vol_factor"] > 1.4
    t["lo_vol"] = t["vol_factor"] < 0.8

    # Trap tags
    t["trap_type"] = ""
    # Bull Trap: pokes above then closes back with long upper or stop-hunt, often on hi/lo vol
    bull_trap = (t["reject_down"] | t["big_up_wick"]) & (t["long_upper"]) & (t["lo_vol"] | t["hi_vol"])
    # Bear Trap: pokes below then closes back with long lower or stop-hunt
    bear_trap = (t["reject_up"] | t["big_down_wick"]) & (t["long_lower"]) & (t["lo_vol"] | t["hi_vol"])
    t.loc[bull_trap, "trap_type"] = "BULL_TRAP"   # bearish implication
    t.loc[bear_trap, "trap_type"] = "BEAR_TRAP"   # bullish implication

    # Absorption bars (institutional soaking): high volume + small body + closes into mid
    mid = t["low"] + 0.5 * t["range"]
    t["absorb"] = (t["vol_factor"] > 1.6) & (t["body"] < 0.35 * t["range"]) & (
        ((t["close"] < mid) & (t["close"] > t["open"])) | ((t["close"] > mid) & (t["close"] < t["open"]))
    )
    t.loc[t["absorb"] & (t["close"] > t["open"]), "trap_type"] = t["trap_type"].replace("", "ABSORB_BUY")
    t.loc[t["absorb"] & (t["close"] < t["open"]), "trap_type"] = t["trap_type"].replace("", "ABSORB_SELL")

    # Optional: RSI divergence signals (very light)
    # Bearish div: price higher high but RSI lower high
    t["price_hh"] = t["high"] > t["high"].shift(2)
    t["rsi_lh"]   = t["rsi"] < t["rsi"].shift(2)
    t["bear_div"] = t["price_hh"] & t["rsi_lh"]

    # Bullish div: price lower low but RSI higher low
    t["price_ll"] = t["low"] < t["low"].shift(2)
    t["rsi_hl"]   = t["rsi"] > t["rsi"].shift(2)
    t["bull_div"] = t["price_ll"] & t["rsi_hl"]

    return t

feat2 = detect_traps(feat)

# ------------------ Scoring Model (0–100) ------------------
def smart_money_score(f: pd.DataFrame, window: int = 40) -> tuple[float, str, pd.DataFrame]:
    recent = f.tail(window).copy()

    bull_traps = (recent["trap_type"] == "BULL_TRAP").sum()
    bear_traps = (recent["trap_type"] == "BEAR_TRAP").sum()
    absorb_buy = (recent["trap_type"] == "ABSORB_BUY").sum()
    absorb_sell= (recent["trap_type"] == "ABSORB_SELL").sum()

    # Divergences (we keep weight light)
    bull_div   = recent["bull_div"].sum()
    bear_div   = recent["bear_div"].sum()

    # Volume balance
    hv = (recent["hi_vol"]).sum()
    lv = (recent["lo_vol"]).sum()

    # Start at neutral 50; add/deduct signals
    score = 50.0
    # Bearish signals (downward pressure)
    score -= bull_traps * 6.5
    score -= absorb_sell * 3.5
    score -= bear_div * 2.0
    # Bullish signals
    score += bear_traps * 6.5
    score += absorb_buy * 3.5
    score += bull_div * 2.0

    # Volume tilt (hi_vol more weight than lo_vol)
    score += (hv - lv) * 0.5

    score = float(np.clip(score, 0.0, 100.0))
    if score >= 60:
        bias = "BUY"
    elif score <= 40:
        bias = "SELL"
    else:
        bias = "NEUTRAL"

    # traps table (latest 20 with non-empty types)
    traps_tbl = recent[recent["trap_type"] != ""].tail(20).copy()
    traps_tbl = traps_tbl[["open","high","low","close","volume","vol_factor","upper_wick","lower_wick","trap_type"]]
    traps_tbl = traps_tbl.rename(columns={
        "vol_factor":"volX","upper_wick":"up_wick","lower_wick":"dn_wick"
    })
    traps_tbl["volX"] = traps_tbl["volX"].round(2)
    traps_tbl["up_wick"] = traps_tbl["up_wick"].round(2)
    traps_tbl["dn_wick"] = traps_tbl["dn_wick"].round(2)

    return score, bias, traps_tbl

score, bias, traps = smart_money_score(feat2, window=60)

# ------------------ UI: Score + Bias ------------------
st.subheader("Smart Money Score (0–100)")
st.progress(int(score))
if bias == "BUY":
    st.success(f"Accumulation / Bear Traps dominant — **{score:.0f}/100 (BUY)**")
elif bias == "SELL":
    st.error(f"Distribution / Bull Traps dominant — **{score:.0f}/100 (SELL)**")
else:
    st.info(f"Mixed / Neutral flows — **{score:.0f}/100 (NEUTRAL)**")

# ------------------ Mini Candlestick with markers ------------------
def _plot_mini(idf: pd.DataFrame, ttl: str):
    d = idf.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}).copy()
    d.index.name = "Date"
    d = d.tail(140)

    # Markers for traps
    marker_buy  = idf["trap_type"].isin(["BEAR_TRAP","ABSORB_BUY"])
    marker_sell = idf["trap_type"].isin(["BULL_TRAP","ABSORB_SELL"])
    add = []

    if marker_buy.any():
        buy_series = pd.Series(np.where(marker_buy.tail(140), d["Close"], np.nan), index=d.index)
        add.append(mpf.make_addplot(buy_series, type='scatter', markersize=40, marker='^'))
    if marker_sell.any():
        sell_series = pd.Series(np.where(marker_sell.tail(140), d["Close"], np.nan), index=d.index)
        add.append(mpf.make_addplot(sell_series, type='scatter', markersize=40, marker='v'))

    # EMAs
    add.append(mpf.make_addplot(d["Close"].ewm(span=20, adjust=False).mean(), width=0.8))
    add.append(mpf.make_addplot(d["Close"].ewm(span=50, adjust=False).mean(), width=0.8))

    style = mpf.make_mpf_style(base_mpf_style="yahoo")
    fig = mpf.figure(style=style, figsize=(11, 4))
    ax  = fig.add_subplot(1,1,1)
    mpf.plot(d, type="candle", addplot=add, volume=False, ax=ax, xrotation=0)
    ax.set_title(ttl, fontsize=11, pad=6)
    st.pyplot(fig, clear_figure=True)

_plot_mini(feat2, f"{symbol}  — Smart Money Traps (markers)  {'• 1D' if used_fallback else ''}")

# ------------------ Trap Table ------------------
st.subheader("Recent Trap / Absorption Events")
if traps.empty:
    st.write("No recent traps detected in the selected window.")
else:
    st.dataframe(traps.sort_index(ascending=False), use_container_width=True)

# ------------------ Save for Dashboard Fusion ------------------
st.session_state.setdefault("performance", {})
st.session_state["performance"]["smartmoney"] = {
    "symbol": symbol,
    "tf": tf,
    "mode": "intraday" if not used_fallback and used and used[1] != "1d" else "daily",
    "used": used,
    "final_score": float(score),
    "bias": bias,
    "traps": traps.reset_index().to_dict(orient="records"),
}
st.success("✅ Smart Money score saved for Dashboard fusion.")
