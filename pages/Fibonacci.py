# pages/Fibonacci.py
# -------------------------------------------------------------
# Fibonacci Retracement (Pivot-based) — BankNifty (Pro Upgrade)
# • Weekend-safe via utils.fetch_smart()
# • True swing detection (pivot highs/lows)
# • S2 sensitivity: 2-candle pivots
# • Candle + Volume + Fib lines + Confidence score
# • Safe mplfinance render
# -------------------------------------------------------------
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import streamlit as st
import pandas as pd, numpy as np
import mplfinance as mpf
from utils import DEFAULT_SYMBOL, fetch_smart

st.set_page_config(page_title="🧭 Fibonacci Retracement", layout="wide")
st.title("🧭 Fibonacci Retracement — BankNifty (Advanced)")

# -------------------- Controls --------------------
c1, c2, c3 = st.columns([1.6, 1.1, 1.1])
with c1:
    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
with c2:
    tf = st.selectbox("Timeframe", ["5m", "15m", "1h", "1d"], index=3)
with c3:
    data_mode = st.radio("Mode", ["🔴 Live", "🟦 Offline / Daily"], index=1 if tf == "1d" else 0, horizontal=True)

TF_PREF = {"5m": ("5d", "5m"), "15m": ("14d", "15m"), "1h": ("60d", "60m"), "1d": ("6mo", "1d")}
prefer = TF_PREF.get(tf, ("6mo", "1d"))

if data_mode.startswith("🟦") or tf == "1d":
    prefer = ("6mo", "1d")

with st.spinner("Fetching data..."):
    df, msg = fetch_smart(symbol)
if msg:
    st.info(msg)
if df is None or df.empty:
    st.warning("⚠️ No data available right now.")
    st.stop()

st.caption(f"Using data period={used[0]} interval={used[1]} | Candles={len(df)}")

if len(df) < 60:
    st.warning("Not enough candles for reliable Fibonacci structure.")
    st.stop()

# -------------------- Pivot detection --------------------
def detect_pivots(frame: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    """Detect pivot highs/lows with sensitivity window."""
    h, l = frame["high"].values, frame["low"].values
    n = len(frame)
    piv_high = np.full(n, False)
    piv_low = np.full(n, False)
    for i in range(window, n - window):
        if h[i] > max(h[i-window:i]) and h[i] > max(h[i+1:i+1+window]):
            piv_high[i] = True
        if l[i] < min(l[i-window:i]) and l[i] < min(l[i+1:i+1+window]):
            piv_low[i] = True
    out = frame.copy()
    out["piv_high"], out["piv_low"] = piv_high, piv_low
    return out

work = detect_pivots(df.copy(), window=2)

ph_idx = work.index[work["piv_high"]]
pl_idx = work.index[work["piv_low"]]
if len(ph_idx) == 0 or len(pl_idx) == 0:
    st.warning("Could not detect swings — try longer period.")
    st.stop()

last_ph, last_pl = ph_idx[-1], pl_idx[-1]
direction, swing_high, swing_low = None, None, None

if last_ph > last_pl:
    lows_after = [t for t in pl_idx if t > last_ph]
    if lows_after:
        swing_high = float(work.loc[last_ph, "high"])
        swing_low = float(work.loc[lows_after[-1], "low"])
        direction = "down"
    else:
        swing_high = float(work.loc[last_ph, "high"])
        prev_lows = [t for t in pl_idx if t < last_ph]
        swing_low = float(work.loc[prev_lows[-1], "low"]) if prev_lows else float(df["low"].min())
        direction = "down"
else:
    highs_after = [t for t in ph_idx if t > last_pl]
    if highs_after:
        swing_low = float(work.loc[last_pl, "low"])
        swing_high = float(work.loc[highs_after[-1], "high"])
        direction = "up"
    else:
        swing_low = float(work.loc[last_pl, "low"])
        prev_highs = [t for t in ph_idx if t < last_pl]
        swing_high = float(work.loc[prev_highs[-1], "high"]) if prev_highs else float(df["high"].max())
        direction = "up"

if abs(swing_high - swing_low) < 1e-6:
    st.warning("Swing range too small to compute Fibonacci levels.")
    st.stop()

# -------------------- Build Fib levels --------------------
def build_fibs(high, low, dirn):
    d = high - low
    if dirn == "up":
        return {
            "0%": high, "23.6%": high - 0.236*d, "38.2%": high - 0.382*d,
            "50%": high - 0.5*d, "61.8%": high - 0.618*d, "78.6%": high - 0.786*d, "100%": low
        }
    else:
        return {
            "0%": low, "23.6%": low + 0.236*d, "38.2%": low + 0.382*d,
            "50%": low + 0.5*d, "61.8%": low + 0.618*d, "78.6%": low + 0.786*d, "100%": high
        }

levels = build_fibs(swing_high, swing_low, direction)
close = float(df["close"].iloc[-1])
targets = [levels["38.2%"], levels["50%"], levels["61.8%"]]
dist = min(abs(close - t) for t in targets)
rng = max(1.0, abs(swing_high - swing_low) / 6.0)
score = max(0.0, 100.0 - (dist / rng) * 100.0)
score = round(score, 2)

if direction == "up":
    zone_text = "Golden Buy Zone (38.2–61.8%)" if levels["61.8%"] <= close <= levels["38.2%"] else "Outside Golden Zone"
else:
    zone_low, zone_high = min(levels["38.2%"], levels["61.8%"]), max(levels["38.2%"], levels["61.8%"])
    zone_text = "Golden Sell Zone (38.2–61.8%)" if zone_low <= close <= zone_high else "Outside Golden Zone"

# -------------------- Display metrics --------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Direction", direction.upper())
m2.metric("Swing High", f"{swing_high:.2f}")
m3.metric("Swing Low", f"{swing_low:.2f}")
m4.metric("Confidence", f"{score:.1f}/100")
st.progress(int(score))
st.info(zone_text)

# -------------------- Plot safely --------------------
st.subheader("📉 Candlestick + Volume + Fibonacci Levels")

plot_df = df.tail(300).copy()
plot_df = plot_df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
plot_df.index.name = "Date"

fib_plots = [mpf.make_addplot(pd.Series(v, index=plot_df.index), linestyle="--", width=0.8) for v in levels.values()]
style = mpf.make_mpf_style(base_mpf_style="yahoo", facecolor="#111", gridstyle="--")

fig, _ = mpf.plot(
    plot_df,
    type="candle",
    volume=True,
    addplot=fib_plots,
    style=style,
    figratio=(18,9),
    figscale=1.2,
    returnfig=True,
    title=f"{symbol} • {direction.upper()} | Fib Score {score:.1f}"
)
st.pyplot(fig, clear_figure=True)

# -------------------- Save score --------------------
st.session_state.setdefault("performance", {})
st.session_state["performance"]["fibonacci"] = {
    "interval": tf,
    "direction": direction,
    "score": float(score),
    "swing_high": float(swing_high),
    "swing_low": float(swing_low),
    "in_golden_zone": zone_text.startswith("Golden"),
}
st.success("✅ Fibonacci score saved for Dashboard fusion.")
