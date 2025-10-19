# pages/Price_Action.py
# -------------------------------------------------------------
# Price Action (50–60% Mid-Zone Rule) — Entry/Exit on 5m
# • Weekend-safe via utils.fetch_smart()
# • Signals: BUY / SELL / HOLD
# • Pro chart: Candles + Volume + BUY/SELL markers
# • Table of signals + Confidence score (0–100)
# • Saves to st.session_state["performance"]["priceaction"]
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf

from utils import DEFAULT_SYMBOL, fetch_smart

st.set_page_config(page_title="Price Action (50–60%)", layout="wide", initial_sidebar_state="expanded")
st.title("🕹️ Price Action — 50–60% Mid-Zone (5m Execution)")

# -------------------- Controls --------------------
c1, c2, c3, c4 = st.columns([1.6, 1.0, 1.0, 1.0])
with c1:
    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL, help="Use NSEBANK.NS or ^NSEBANK (index).")
with c2:
    # Entry/Exit always 5m, but we allow user to pick a longer lookback
    period = st.selectbox("Lookback Period", ["2d", "5d", "7d", "14d"], index=1)
with c3:
    # Interval is fixed to 5m for execution (locked)
    interval = st.selectbox("Interval (fixed to 5m for entries)", ["5m"], index=0, disabled=True)
with c4:
    data_mode = st.radio("Data Mode", ["🔴 Live Intraday", "🟦 Offline / Daily (Safe)"], index=0, horizontal=True)

st.caption("Entry & exit calculations are always on **5-minute candles**. Multi-TF context is used elsewhere (Trend/Fib).")

# -------------------- Data Fetch (Weekend-safe) --------------------
prefer = (period, "5m")
if data_mode.startswith("🟦"):
    # If user forces offline daily, we still compute on last available 5m (if any),
    # otherwise show info and stop gracefully.
    prefer = (period, "5m")

with st.spinner("Fetching intraday candles…"):
    df, used, msg = fetch_smart(symbol, prefer=prefer)

if msg:
    st.info(msg)

if df is None or df.empty:
    st.warning("No intraday data available right now. Try during market hours.")
    st.stop()

st.caption(f"Using data: period={used[0]} • interval={used[1]}")

if len(df) < 30:
    st.warning("Not enough 5-minute candles to evaluate the rule. Try a longer period.")
    st.stop()

# -------------------- 50–60% Mid-Zone Rule --------------------
def generate_signals_50_60(frame: pd.DataFrame) -> pd.DataFrame:
    """
    BUY:
      - current LOW >= prev_mid_50
      - current LOW <= prev_mid_60
      - current CLOSE > prev HIGH

    SELL:
      - current HIGH <= prev_mid_50
      - current HIGH >= prev_mid_40
      - current CLOSE < prev LOW
    Else HOLD.
    """
    d = frame.copy().reset_index()
    # Ensure needed columns exist
    for c in ["open", "high", "low", "close"]:
        if c not in d.columns:
            raise ValueError(f"Missing column: {c}")

    sig = ["HOLD"]
    conf = [0.0]

    for i in range(1, len(d)):
        ph = float(d.loc[i - 1, "high"])
        pl = float(d.loc[i - 1, "low"])
        pr = max(0.01, ph - pl)  # previous candle range (guard)

        mid_50 = pl + 0.50 * pr
        mid_60 = pl + 0.60 * pr
        mid_40 = pl + 0.40 * pr

        c_hi = float(d.loc[i, "high"])
        c_lo = float(d.loc[i, "low"])
        c_cl = float(d.loc[i, "close"])

        # BUY check
        if (c_lo >= mid_50) and (c_lo <= mid_60) and (c_cl > ph):
            # Confidence: how centrally it tapped 55%, breakout strength & range quality
            center = pl + 0.55 * pr
            # closeness to center (smaller better)
            zone_penalty = min(1.0, abs(c_lo - center) / (0.10 * pr))  # within ±10% of range around center
            zone_score = 100.0 * (1.0 - zone_penalty)

            breakout = (c_cl - ph) / pr  # how much beyond previous high
            breakout_score = max(0.0, min(100.0, breakout * 400.0))  # scale

            body = abs(c_cl - float(d.loc[i, "open"])) / pr
            body_score = max(0.0, min(100.0, body * 100.0))

            score = 0.45 * zone_score + 0.35 * breakout_score + 0.20 * body_score
            sig.append("BUY")
            conf.append(round(score, 2))
            continue

        # SELL check
        if (c_hi <= mid_50) and (c_hi >= mid_40) and (c_cl < pl):
            center = pl + 0.45 * pr  # inverse center
            zone_penalty = min(1.0, abs(c_hi - center) / (0.10 * pr))
            zone_score = 100.0 * (1.0 - zone_penalty)

            breakout = (pl - c_cl) / pr
            breakout_score = max(0.0, min(100.0, breakout * 400.0))

            body = abs(float(d.loc[i, "open"]) - c_cl) / pr
            body_score = max(0.0, min(100.0, body * 100.0))

            score = 0.45 * zone_score + 0.35 * breakout_score + 0.20 * body_score
            sig.append("SELL")
            conf.append(round(score, 2))
            continue

        sig.append("HOLD")
        conf.append(0.0)

    d["signal"] = sig
    d["confidence"] = conf
    d = d.set_index(d.columns[0])  # restore datetime index
    return d

sig_df = generate_signals_50_60(df)

# -------------------- Chart with BUY/SELL markers --------------------
st.subheader("Candles + Volume + Entry Markers (5m)")

plot_df = df.rename(
    columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
)
plot_df.index.name = "Date"

# build marker series
buy_y = np.full(len(plot_df), np.nan)
sell_y = np.full(len(plot_df), np.nan)
idx_map = {ts: i for i, ts in enumerate(plot_df.index)}

for ts, row in sig_df.iterrows():
    if ts in idx_map:
        j = idx_map[ts]
        if row["signal"] == "BUY":
            # put marker slightly below the close
            buy_y[j] = float(plot_df["Close"].iloc[j]) * 0.998
        elif row["signal"] == "SELL":
            sell_y[j] = float(plot_df["Close"].iloc[j]) * 1.002

apds = [
    mpf.make_addplot(buy_y, type="scatter", markersize=60, marker="^"),
    mpf.make_addplot(sell_y, type="scatter", markersize=60, marker="v"),
]

style = mpf.make_mpf_style(base_mpf_style="yahoo")
mpf.plot(
    plot_df.tail(300),
    type="candle",
    volume=True,
    addplot=apds,
    figratio=(18, 9),
    figscale=1.2,
    style=style,
    panel_ratios=(5, 1),
    ylabel="Price",
    ylabel_lower="Volume",
    datetime_format="%d-%b %H:%M",
)

st.pyplot(use_container_width=True, clear_figure=True)

# -------------------- Signals table --------------------
st.subheader("Signals (last 100 rows)")
table = sig_df[["signal", "confidence", "close"]].tail(100).copy()
table.rename(columns={"close": "price"}, inplace=True)
st.dataframe(table, use_container_width=True)

# -------------------- Latest signal widget --------------------
last = sig_df.tail(1).iloc[0]
latest_sig = str(last["signal"])
latest_price = float(last["close"])
latest_conf = float(last["confidence"])

if latest_sig == "BUY":
    st.success(f"Latest: **BUY** @ {latest_price:.2f}  |  Confidence: {latest_conf:.1f}/100")
elif latest_sig == "SELL":
    st.error(f"Latest: **SELL** @ {latest_price:.2f}  |  Confidence: {latest_conf:.1f}/100")
else:
    st.info(f"Latest: **HOLD**  |  Last price: {latest_price:.2f}")

# -------------------- Save to Dashboard Fusion --------------------
total_signals = int((sig_df["signal"] != "HOLD").sum())
avg_conf = float(sig_df.loc[sig_df["signal"] != "HOLD", "confidence"].mean()) if total_signals > 0 else 0.0

st.session_state.setdefault("performance", {})
st.session_state["performance"]["priceaction"] = {
    "period": used[0],
    "interval": used[1],
    "signals": total_signals,
    "avg_confidence": round(avg_conf, 2),
    "last_signal": latest_sig,
    "last_price": latest_price,
    "last_confidence": latest_conf,
}

st.success("✅ Price-Action stats saved for Dashboard fusion.")
st.caption("Note: Execution/backtest uses these signals with your trailing SL engine.")
