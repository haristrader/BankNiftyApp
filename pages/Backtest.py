# pages/Backtest.py
import streamlit as st
import pandas as pd
import numpy as np

# utils.py must provide these (we already added them earlier):
from utils import (
    DEFAULT_SYMBOL,         # recommend "NSEBANK.NS"
    fetch_smart,            # smart fetch with fallbacks
    generate_signals_50pct, # 50% rule signals
    simulate_atm_option_trades,  # ATM options simulator (CE/PE) with trailing ladder
)

st.title("🧪 Backtest (Trailing SL) — BankNifty ATM Options")

# =========================
# 1) Data Mode (Trader Style)
# =========================
st.markdown("### 📊 Data Source Mode")
data_mode = st.radio(
    label="Pick data mode",
    options=["🔴 Live Intraday (5m)", "🟦 Offline / Daily (Safe)"],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
)

# =========================
# 2) Controls
# =========================
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
with c2:
    prefer_period = st.selectbox("Preferred Period", ["5d", "14d", "30d"], index=0)
with c3:
    prefer_interval = st.selectbox("Preferred Interval", ["5m", "15m", "30m", "60m", "1d"], index=0)

st.markdown("---")

c4, c5, c6, c7 = st.columns([1, 1, 1, 1])
with c4:
    pricing_mode = st.radio("Pricing Mode", ["ATM Delta (Sim)", "Index Proxy"], index=0)
with c5:
    init_sl = st.number_input("Initial SL (option points)", min_value=5, max_value=100, value=10, step=1)
with c6:
    lot = st.number_input("Lot Size", min_value=1, max_value=100, value=15, step=1)
with c7:
    theta = st.number_input("Theta per candle (sim)", min_value=0.0, max_value=1.0, value=0.00, step=0.01)

# =========================
# 3) Data Fetch
# =========================
with st.spinner("Fetching market data..."):
    if data_mode.startswith("🔴"):
        # Intraday preferred; fetch_smart will cascade to safe combos if needed.
        df, used = fetch_smart(symbol, prefer=(prefer_period, prefer_interval))
    else:
        # Force daily mode directly (cloud-safe; avoids intraday throttling)
        import yfinance as yf
        df = yf.download(symbol, period="3mo", interval="1d", progress=False, auto_adjust=True)
        used = ("3mo", "1d")

st.caption(f"Using: period={used[0]}  interval={used[1]}")

if df is None or df.empty:
    st.error("⚠ No data available even after fallbacks.")
    st.stop()

# =========================
# 4) Generate Signals (50% rule)
# =========================
sig_df = generate_signals_50pct(df, mid_factor=0.5)
with st.expander("Recent Signals (last 12)", expanded=False):
    st.dataframe(sig_df[["close", "signal"]].tail(12), use_container_width=True)

# =========================
# 5) Run Simulation
# =========================
st.subheader("Simulation Results")

if pricing_mode.startswith("ATM"):
    tr, equity, backtest_score = simulate_atm_option_trades(
        sig_df,
        signals_col="signal",
        init_sl_pts=init_sl,
        lot_size=lot,
        mode="delta",               # delta-based ATM model (CE/PE)
        theta_per_candle=theta,
    )
else:
    # Index proxy (delta ≈ 1, no decay) using same engine for consistency
    tr, equity, backtest_score = simulate_atm_option_trades(
        sig_df,
        signals_col="signal",
        init_sl_pts=init_sl,
        lot_size=lot,
        mode="index",
        theta_per_candle=0.0,
    )

# =========================
# 6) Metrics + Displays
# =========================
def _max_drawdown(series: pd.Series) -> float:
    """Max drawdown of an equity curve (in same units as series)."""
    if series is None or series.empty:
        return 0.0
    roll_max = series.cummax()
    dd = series - roll_max
    return float(dd.min())  # negative value

if tr is None or tr.empty:
    st.info("No closed trades in this range yet. Try Daily mode or a longer period.")
    wins = losses = total = 0
    winrate = 0.0
    pnl_total = 0.0
    max_dd = 0.0
    equity_series = pd.Series(dtype=float)
else:
    # Compute robust metrics
    total = int(len(tr))
    wins = int((tr["pnl_points"] > 0).sum())
    losses = int((tr["pnl_points"] <= 0).sum())
    winrate = (wins / total) * 100.0 if total > 0 else 0.0
    pnl_total = float(tr["pnl_points"].sum())

    if equity is None or (isinstance(equity, pd.Series) and equity.empty) or ("cum_pnl" not in tr.columns):
        equity_series = tr["pnl_points"].cumsum()
    else:
        equity_series = equity

    max_dd = _max_drawdown(equity_series)

    # KPI cards
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Trades", total)
    cB.metric("Win Rate", f"{winrate:.1f}%")
    cC.metric("Total PnL (₹)", f"{pnl_total:,.0f}")
    cD.metric("Max Drawdown (₹)", f"{max_dd:,.0f}")

    st.dataframe(tr.tail(50), use_container_width=True)
    st.line_chart(equity_series, height=220)

# =========================
# 7) Save for AI Console (NOT for Dashboard score)
# =========================
perf_payload = {
    "symbol": symbol,
    "data_mode": "intraday" if data_mode.startswith("🔴") else "daily",
    "used_period": used[0],
    "used_interval": used[1],
    "pricing_mode": "ATM_Delta" if pricing_mode.startswith("ATM") else "IndexProxy",
    "init_sl": float(init_sl),
    "lot_size": int(lot),
    "theta": float(theta),
    "trades": int(total),
    "wins": int(wins),
    "losses": int(losses),
    "winrate": float(winrate),
    "pnl_total": float(pnl_total),
    "max_drawdown": float(max_dd),
    "backtest_score": float(backtest_score),  # learning signal only
}

# Place into session_state for AI Console to learn from
st.session_state.setdefault("performance", {})
st.session_state["performance"]["backtest"] = perf_payload

st.success("✅ Backtest performance saved for AI Console learning (no impact on Dashboard score).")
