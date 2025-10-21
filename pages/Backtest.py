# pages/Backtest.py
# -------------------------------------------------------------
# Full Backtest v2 — BankNifty ATM Options (Intraday / Daily)
# Virtual Capital + Equity Curve + Chart Overlay + AI Sync
# -------------------------------------------------------------

import streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from utils import fetch_smart, DEFAULT_SYMBOL, generate_signals_50pct, simulate_atm_option_trades

# -------------------- Page Config --------------------
st.set_page_config(page_title="Backtest - ATM Options", layout="wide")
st.title("🧪 BankNifty ATM Backtest — Paper Trade Simulator")

# -------------------- Sidebar Controls --------------------
st.sidebar.header("⚙️ Settings")
data_mode = st.sidebar.radio("Data Mode", ["🔴 Intraday (5m)", "🟦 Safe / Daily"], index=0)
symbol = st.sidebar.text_input("Symbol", value=DEFAULT_SYMBOL)
prefer_period = st.sidebar.selectbox("Preferred Period", ["5d", "14d", "1mo"], index=0)
prefer_interval = st.sidebar.selectbox("Preferred Interval", ["5m", "15m", "1h", "1d"], index=0)
init_sl = st.sidebar.number_input("Initial SL (pts)", 5, 100, 10)
lot = st.sidebar.number_input("Lot Size", 1, 50, 15)
theta = st.sidebar.number_input("Theta per Candle (Sim)", 0.0, 1.0, 0.02, 0.01)
pricing_mode = st.sidebar.radio("Pricing Mode", ["ATM Delta (Sim)", "Index Proxy"], index=0)

# -------------------- Virtual Capital --------------------
st.sidebar.markdown("---")
if "virtual_capital" not in st.session_state:
    st.session_state.virtual_capital = 10000
st.sidebar.metric("💰 Virtual Capital", f"₹{st.session_state.virtual_capital:,.0f}")
if st.sidebar.button("Add ₹10,000 Capital"):
    st.session_state.virtual_capital += 10000

# -------------------- Data Fetch --------------------
with st.spinner("Fetching data..."):
    if data_mode.startswith("🔴"):
        df, used, msg = fetch_smart(symbol, prefer=(prefer_period, prefer_interval))
    else:
        df, used, msg = fetch_smart(symbol, prefer=("3mo", "1d"))
    if msg:
        st.info(msg)
if df.empty:
    st.error("No data found. Try changing timeframe or wait for market hours.")
    st.stop()
st.caption(f"Using {used[0]} / {used[1]} timeframe")

# -------------------- Signal Generation --------------------
sig_df = generate_signals_50pct(df, mid_factor=0.55)
if sig_df.empty:
    st.warning("No signals generated.")
    st.stop()
st.dataframe(sig_df.tail(10), use_container_width=True)

# -------------------- Run Simulation --------------------
st.subheader("Backtest Simulation")
tr, equity, backtest_score = simulate_atm_option_trades(
    sig_df,
    signals_col="signal",
    init_sl_pts=init_sl,
    lot_size=lot,
    mode="delta" if pricing_mode.startswith("ATM") else "index",
    theta_per_candle=theta,
)

if tr is None or tr.empty:
    st.warning("No trades executed during this range.")
    st.stop()

# -------------------- Performance Metrics --------------------
wins = (tr["pnl_points"] > 0).sum()
losses = (tr["pnl_points"] <= 0).sum()
total = len(tr)
winrate = (wins / total * 100) if total > 0 else 0
pnl_total = tr["pnl_points"].sum()
st.session_state.virtual_capital += pnl_total

st.markdown("### 📈 Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trades", total)
c2.metric("Win Rate", f"{winrate:.1f}%")
c3.metric("PnL (₹)", f"{pnl_total:,.0f}")
c4.metric("Balance (₹)", f"{st.session_state.virtual_capital:,.0f}")

# -------------------- Equity Curve --------------------
st.markdown("---")
equity_series = pd.Series(equity if equity is not None else tr["pnl_points"].cumsum())
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(equity_series.index, equity_series.values, linewidth=1.3)
ax.set_title("Equity Curve")
ax.grid(alpha=0.3)
st.pyplot(fig, clear_figure=True)

# -------------------- Trade Entry/Exit Chart --------------------
st.markdown("### 📊 Candle Chart with Trade Points")
fig2, ax2 = plt.subplots(figsize=(11, 4))
ax2.plot(df.index, df["close"], label="Close", linewidth=0.8)
ax2.scatter(tr["entry_time"], tr["entry"], marker="^", color="lime", label="Entry", s=60)
ax2.scatter(tr["exit_time"], tr["exit"], marker="v", color="red", label="Exit", s=60)
ax2.legend()
ax2.grid(alpha=0.2)
st.pyplot(fig2, clear_figure=True)

# -------------------- Store for AI Console --------------------
def _serialize_equity(eq: pd.Series):
    if eq is None or len(eq) == 0:
        return []
    if isinstance(eq, pd.Series):
        return [{"t": str(t), "v": float(v)} for t, v in eq.items()]
    return [{"t": str(i), "v": float(v)} for i, v in enumerate(eq)]

perf_payload = {
    "symbol": symbol,
    "mode": "intraday" if data_mode.startswith("🔴") else "daily",
    "init_sl": float(init_sl),
    "lot": int(lot),
    "theta": float(theta),
    "trades": total,
    "wins": int(wins),
    "losses": int(losses),
    "winrate": float(winrate),
    "pnl_total": float(pnl_total),
    "balance": float(st.session_state.virtual_capital),
    "backtest_score": float(backtest_score),
    "equity_curve": _serialize_equity(equity_series),
}

st.session_state.setdefault("performance", {})
st.session_state["performance"]["backtest"] = perf_payload
st.success("✅ Backtest result stored for AI Console learning.")
