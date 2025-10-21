# pages/Price_Action.py
# -------------------------------------------------------------
# Price Action (50–60% Mid-Zone Rule) — 5m Entry Engine
# • Works with utils.fetch_smart() (weekend-safe)
# • Trend alignment (optional)
# • Confidence scoring (breakout + body + position)
# • Candlestick + Volume + Signal markers
# • Auto Dashboard sync
# -------------------------------------------------------------

import streamlit as st
import pandas as pd, numpy as np
import mplfinance as mpf
from utils import DEFAULT_SYMBOL, fetch_smart

st.set_page_config(page_title="Price Action (50–60%)", layout="wide")
st.title("🎯 Price Action — 50–60% Mid-Zone (5m Strategy)")

# -------------------- Controls --------------------
c1, c2, c3, c4 = st.columns([1.6, 1.0, 1.0, 1.0])
with c1:
    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
with c2:
    period = st.selectbox("Lookback Period", ["2d", "5d", "7d", "14d"], index=1)
with c3:
    interval = st.selectbox("Interval", ["5m"], index=0, disabled=True)
with c4:
    mode = st.radio("Data Mode", ["🔴 Live Intraday", "🟦 Offline/Daily"], index=0, horizontal=True)

st.caption("Entries & exits are always based on **5-minute candles**. Multi-TF confirmation comes from Trend, Fib, SmartMoney modules.")

prefer = (period, "5m")
if mode.startswith("🟦"):
    prefer = ("7d", "5m")

# -------------------- Fetch Data --------------------
with st.spinner("Fetching 5-minute candles…"):
    df, used, msg = fetch_smart(symbol, prefer=prefer)

if msg:
    st.info(msg)
if df is None or df.empty:
    st.warning("⚠️ No data available. Try during market hours.")
    st.stop()

st.caption(f"Using data: period={used[0]} | interval={used[1]} | candles={len(df)}")

# -------------------- 50–60% Mid-Zone Logic --------------------
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index()
    signals, confs = ["HOLD"], [0.0]

    for i in range(1, len(d)):
        ph, pl = float(d.loc[i-1, "high"]), float(d.loc[i-1, "low"])
        pr = max(0.01, ph - pl)
        mid50, mid60, mid40 = pl + 0.5*pr, pl + 0.6*pr, pl + 0.4*pr

        c_hi, c_lo, c_cl, c_op = float(d.loc[i, "high"]), float(d.loc[i, "low"]), float(d.loc[i, "close"]), float(d.loc[i, "open"])

        # BUY
        if (c_lo >= mid50) and (c_lo <= mid60) and (c_cl > ph):
            center = pl + 0.55*pr
            zone_penalty = min(1.0, abs(c_lo - center) / (0.10 * pr))
            zone_score = 100 * (1 - zone_penalty)
            breakout_score = max(0, min(100, (c_cl - ph) / pr * 400))
            body_score = max(0, min(100, abs(c_cl - c_op) / pr * 100))
            score = 0.45 * zone_score + 0.35 * breakout_score + 0.20 * body_score
            signals.append("BUY")
            confs.append(round(score, 2))
            continue

        # SELL
        if (c_hi <= mid50) and (c_hi >= mid40) and (c_cl < pl):
            center = pl + 0.45*pr
            zone_penalty = min(1.0, abs(c_hi - center) / (0.10 * pr))
            zone_score = 100 * (1 - zone_penalty)
            breakout_score = max(0, min(100, (pl - c_cl) / pr * 400))
            body_score = max(0, min(100, abs(c_op - c_cl) / pr * 100))
            score = 0.45 * zone_score + 0.35 * breakout_score + 0.20 * body_score
            signals.append("SELL")
            confs.append(round(score, 2))
            continue

        signals.append("HOLD")
        confs.append(0.0)

    d["signal"], d["confidence"] = signals, confs
    d = d.set_index(d.columns[0])
    return d

sig_df = generate_signals(df)

# -------------------- Chart --------------------
st.subheader("📉 Candlestick + Volume + Entry Markers (5m)")

plot_df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
plot_df.index.name = "Date"
buy_y, sell_y = np.full(len(plot_df), np.nan), np.full(len(plot_df), np.nan)
idx_map = {ts: i for i, ts in enumerate(plot_df.index)}

for ts, row in sig_df.iterrows():
    if ts in idx_map:
        j = idx_map[ts]
        if row["signal"] == "BUY":
            buy_y[j] = float(plot_df["Close"].iloc[j]) * 0.998
        elif row["signal"] == "SELL":
            sell_y[j] = float(plot_df["Close"].iloc[j]) * 1.002

apds = [
    mpf.make_addplot(buy_y, type="scatter", markersize=60, marker="^", color="lime"),
    mpf.make_addplot(sell_y, type="scatter", markersize=60, marker="v", color="red"),
]

style = mpf.make_mpf_style(base_mpf_style="yahoo", facecolor="#111")
fig, _ = mpf.plot(
    plot_df.tail(300),
    type="candle",
    volume=True,
    addplot=apds,
    figratio=(18, 9),
    figscale=1.1,
    style=style,
    returnfig=True,
    panel_ratios=(5, 1),
    title=f"{symbol} • 50–60% Price Action"
)
st.pyplot(fig, clear_figure=True)

# -------------------- Table --------------------
st.subheader("🧾 Signal Summary (last 100 rows)")
table = sig_df[["signal","confidence","close"]].tail(100).rename(columns={"close":"price"})
st.dataframe(table, use_container_width=True)

# -------------------- Last Signal --------------------
last = sig_df.tail(1).iloc[0]
sig, conf, price = last["signal"], float(last["confidence"]), float(last["close"])

if sig == "BUY":
    st.success(f"Latest Signal: BUY @ {price:.2f} | Confidence {conf:.1f}/100")
elif sig == "SELL":
    st.error(f"Latest Signal: SELL @ {price:.2f} | Confidence {conf:.1f}/100")
else:
    st.info(f"Latest Signal: HOLD | Last price {price:.2f}")

# -------------------- Save to Dashboard --------------------
active = sig_df[sig_df["signal"] != "HOLD"]
total = len(active)
avg_conf = float(active["confidence"].mean()) if total > 0 else 0.0

st.session_state.setdefault("performance", {})
st.session_state["performance"]["priceaction"] = {
    "signals": total,
    "avg_conf": round(avg_conf, 2),
    "last_signal": sig,
    "last_confidence": conf,
    "score": round((avg_conf + conf) / 2, 2)
}
st.success("✅ Price Action performance saved to Dashboard fusion.")
