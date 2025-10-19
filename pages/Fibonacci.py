# pages/Fibonacci.py
# -------------------------------------------------------------
# Fibonacci Retracement (Pivot-based) — BankNifty
# • Weekend-safe via utils.fetch_smart()
# • True swing detection (pivot highs/lows), not hard min/max
# • S2 sensitivity: 2 candles on each side for pivots
# • Candles + Volume + Fib lines + Confidence score
# • Saves score to st.session_state["performance"]["fibonacci"]
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf

from utils import DEFAULT_SYMBOL, fetch_smart

st.set_page_config(page_title="Fibonacci", layout="wide", initial_sidebar_state="expanded")
st.title("🧭 Fibonacci Retracement — BankNifty")

# -------------------- Controls --------------------
c1, c2, c3 = st.columns([1.6, 1.1, 1.1])
with c1:
    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
with c2:
    tf = st.selectbox(
        "Timeframe",
        ["5m", "15m", "1h", "1d"],
        index=3,
        help="Select the interval for analysis.",
    )
with c3:
    data_mode = st.radio("Data Mode", ["🔴 Live Intraday", "🟦 Offline / Daily (Safe)"], index=1 if tf == "1d" else 0, horizontal=True)

# Preferred period by TF (enough candles for pivots)
TF_PREF = {"5m": ("5d", "5m"), "15m": ("14d", "15m"), "1h": ("60d", "60m"), "1d": ("6mo", "1d")}
prefer = TF_PREF.get(tf, ("6mo", "1d"))

# Force daily safe mode if chosen
if data_mode.startswith("🟦") or tf == "1d":
    prefer = ("6mo", "1d")

with st.spinner("Fetching price data…"):
    df, used, msg = fetch_smart(symbol, prefer=prefer)

if msg:
    st.info(msg)
if df is None or df.empty:
    st.warning("No data available for this selection right now.")
    st.stop()

st.caption(f"Using data: period={used[0]} • interval={used[1]}")

# Ensure we have enough rows
if len(df) < 60:
    st.warning("Not enough candles to detect reliable swings. Try a longer period.")
    st.stop()

# -------------------- Pivot-based swing detection (S2) --------------------
def detect_pivots(frame: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    """Return DataFrame with boolean columns 'piv_high' and 'piv_low' using S2 sensitivity."""
    h = frame["high"].values
    l = frame["low"].values
    n = len(frame)
    piv_high = np.full(n, False)
    piv_low = np.full(n, False)

    for i in range(window, n - window):
        left_h = h[i - window : i]
        right_h = h[i + 1 : i + 1 + window]
        left_l = l[i - window : i]
        right_l = l[i + 1 : i + 1 + window]

        if h[i] > np.max(left_h) and h[i] > np.max(right_h):
            piv_high[i] = True
        if l[i] < np.min(left_l) and l[i] < np.min(right_l):
            piv_low[i] = True

    out = frame.copy()
    out["piv_high"] = piv_high
    out["piv_low"] = piv_low
    return out

work = detect_pivots(df.copy(), window=2)

# Choose most recent meaningful swing pair:
# If last pivot is a HIGH (top then drop) → trend down → Fib from swing high to next swing low after it.
# If last pivot is a LOW (bottom then bounce) → trend up → Fib from swing low to next swing high after it.
ph_idx = work.index[work["piv_high"]]
pl_idx = work.index[work["piv_low"]]

if len(ph_idx) == 0 or len(pl_idx) == 0:
    st.warning("Could not detect clear swings. Try a longer period or different timeframe.")
    st.stop()

last_ph = ph_idx[-1]
last_pl = pl_idx[-1]

# Determine direction and pick the latest valid pair in proper order
direction = None
swing_high = None
swing_low = None

if last_ph > last_pl:
    # Last major event was a pivot HIGH → looking for swing LOW after that high
    # Find the latest pivot low after last_ph; if none, take last low
    lows_after = [t for t in pl_idx if t > last_ph]
    if len(lows_after) > 0:
        swing_high = float(work.loc[last_ph, "high"])
        swing_low = float(work.loc[lows_after[-1], "low"])
        direction = "down"  # using high→low fib
    else:
        # fallback: use nearest previous pivot low
        swing_high = float(work.loc[last_ph, "high"])
        # pick the nearest pivot low before last_ph to maintain structure
        prev_lows = [t for t in pl_idx if t < last_ph]
        swing_low = float(work.loc[prev_lows[-1], "low"]) if len(prev_lows) > 0 else float(df["low"].min())
        direction = "down"
else:
    # Last major event was a pivot LOW → looking for swing HIGH after that low
    highs_after = [t for t in ph_idx if t > last_pl]
    if len(highs_after) > 0:
        swing_low = float(work.loc[last_pl, "low"])
        swing_high = float(work.loc[highs_after[-1], "high"])
        direction = "up"   # using low→high fib
    else:
        swing_low = float(work.loc[last_pl, "low"])
        prev_highs = [t for t in ph_idx if t < last_pl]
        swing_high = float(work.loc[prev_highs[-1], "high"]) if len(prev_highs) > 0 else float(df["high"].max())
        direction = "up"

# Avoid degenerate range
if abs(swing_high - swing_low) < 1e-6:
    st.warning("Swing range too small to compute Fibonacci levels.")
    st.stop()

# -------------------- Build Fibonacci levels --------------------
def build_fibs(high: float, low: float, dirn: str):
    d = high - low
    if dirn == "up":
        levels = {
            "0%": high,
            "23.6%": high - 0.236 * d,
            "38.2%": high - 0.382 * d,
            "50%": high - 0.5 * d,
            "61.8%": high - 0.618 * d,
            "78.6%": high - 0.786 * d,
            "100%": low,
        }
    else:
        levels = {
            "0%": low,
            "23.6%": low + 0.236 * d,
            "38.2%": low + 0.382 * d,
            "50%": low + 0.5 * d,
            "61.8%": low + 0.618 * d,
            "78.6%": low + 0.786 * d,
            "100%": high,
        }
    return {k: float(v) for k, v in levels.items()}

levels = build_fibs(swing_high, swing_low, direction)

# -------------------- Confidence score (Golden zone proximity) --------------------
close = float(df["close"].iloc[-1])
targets = [levels["38.2%"], levels["50%"], levels["61.8%"]]
dist = min(abs(close - t) for t in targets)
rng = max(1.0, abs(swing_high - swing_low) / 6.0)
score = max(0.0, 100.0 - (dist / rng) * 100.0)
score = round(float(score), 2)

# Zone label
if direction == "up":
    zone_text = "Golden Buy Zone (38.2–61.8%)" if levels["61.8%"] <= close <= levels["38.2%"] else "Outside Golden Zone"
else:
    # downtrend: golden sell zone is between 38.2 and 61.8 on the way down
    zone_low = min(levels["38.2%"], levels["61.8%"])
    zone_high = max(levels["38.2%"], levels["61.8%"])
    zone_text = "Golden Sell Zone (38.2–61.8%)" if zone_low <= close <= zone_high else "Outside Golden Zone"

# -------------------- Display metrics --------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Direction", "UP" if direction == "up" else "DOWN")
m2.metric("Swing High", f"{swing_high:.2f}")
m3.metric("Swing Low", f"{swing_low:.2f}")
m4.metric("Fib Confidence", f"{score:.1f}/100")
st.progress(int(score))
st.info(zone_text)

# -------------------- Plot with Fib lines --------------------
def plot_with_fibs(frame: pd.DataFrame, lv: dict, dirn: str):
    d = frame.copy()
    d = d.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    d.index.name = "Date"

    # Create a list of horizontal line addplots for fibs
    add_plots = []
    for name, y in lv.items():
        add_plots.append(mpf.make_addplot(pd.Series(y, index=d.index), width=0.8, linestyle="--"))

    style = mpf.make_mpf_style(base_mpf_style="yahoo")
    mpf.plot(
        d.tail(300),
        type="candle",
        volume=True,
        addplot=add_plots,
        figratio=(18, 9),
        figscale=1.2,
        style=style,
        panel_ratios=(5, 1),
        ylabel="Price",
        ylabel_lower="Volume",
        datetime_format="%d-%b %H:%M" if used[1] != "1d" else "%d-%b",
    )

st.subheader("Candles + Volume + Fibonacci Levels")
fig = mpf.figure(style="yahoo", figsize=(14, 7))
plot_with_fibs(df, levels, direction)
st.pyplot(fig, clear_figure=True)

# -------------------- Save for Dashboard Fusion --------------------
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
