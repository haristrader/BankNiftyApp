import streamlit as st
import pandas as pd
import numpy as np

from utils import (
    DEFAULT_SYMBOL,           # should be "NSEBANK.NS"
    fetch_smart,              # NSE-first data fetcher with fallbacks
    generate_signals_50pct,   # 50% rule signals
    simulate_atm_option_trades,  # ATM options simulator with trailing ladder
)

# ------------------------------------------------------
# Backtest – BankNifty ATM Options (Intraday/Daily Toggle)
# Performance is saved for AI Console (not for Dashboard score)
# ------------------------------------------------------

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
# 3) Data Fetch (NSE-first via utils.fetch_smart)
# =========================
with st.spinner("Fetching market data..."):
    if data_mode.startswith("🔴"):
        # Intraday preference; utils will try NSE 1st then fallback combos
        df, used = fetch_smart(symbol, prefer=(prefer_period, prefer_interval))
    else:
        # Force Daily Safe mode (NSE 1D first). Reliable on cloud/weekends.
        df, used = fetch_smart(symbol, prefer=("3mo", "1d"))

st.caption(f"Using: period={used[0]}  interval={used[1]}")

if df is None or df.empty:
    st.error("⚠ No data available even after NSE + fallback attempts.")
    st.stop()

# =========================
# 4) Generate Signals (50% rule)
# =========================
sig_df = generate_signals_50pct(df, mid_factor=0.5)
with st.expander("Recent Signals (last 12)", expanded=False):
    show_cols = [c for c in ["close", "signal"] if c in sig_df.columns]
    st.dataframe(sig_df[show_cols].tail(12), use_container_width=True)

# =========================
# 5) Run Simulation (ATM delta model or index proxy)
# =========================
st.subheader("Simulation Results")

if pricing_mode.startswith("ATM"):
    tr, equity, backtest_score = simulate_atm_option_trades(
        sig_df,
        signals_col="signal",
        init_sl_pts=init_sl,
        lot_size=lot,
        mode="delta",               # CE/PE ATM delta simulation
        theta_per_candle=theta,
    )
else:
    # Index proxy (delta ≈ 1, no decay) for comparison
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
    """Max drawdown of an equity curve (negative value)."""
    if series is None or len(series) == 0:
        return 0.0
    roll_max = pd.Series(series).cummax()
    dd = pd.Series(series) - roll_max
    return float(dd.min())  # usually negative

if tr is None or tr.empty:
    st.info("No closed trades in this range yet. Try Daily mode or a longer period.")
    wins = losses = total = 0
    winrate = 0.0
    pnl_total = 0.0
    equity_series = pd.Series(dtype=float)
    max_dd = 0.0
else:
    total = int(len(tr))
    wins = int((tr["pnl_points"] > 0).sum())
    losses = int((tr["pnl_points"] <= 0).sum())
    winrate = (wins / total) * 100.0 if total > 0 else 0.0
    pnl_total = float(tr["pnl_points"].sum())

    # equity can be Series (returned) or build from trades
    if equity is None or (isinstance(equity, pd.Series) and equity.empty):
        equity_series = tr["pnl_points"].cumsum()
    else:
        equity_series = pd.Series(equity, index=tr.index[-len(equity):]) if not isinstance(equity, pd.Series) else equity

    max_dd = _max_drawdown(equity_series)

    cA, cB, cC, cD = st.columns(4)
    cA.metric("Trades", total)
    cB.metric("Win Rate", f"{winrate:.1f}%")
    cC.metric("Total PnL (₹)", f"{pnl_total:,.0f}")
    cD.metric("Max Drawdown (₹)", f"{max_dd:,.0f}")

    st.dataframe(tr.tail(50), use_container_width=True)
    st.line_chart(equity_series, height=220)

# =========================
# 7) Save for AI Console (NOT for Dashboard score)
#    + Save equity curve (serialized) for learning
# =========================
def _serialize_equity(eq: pd.Series) -> list[dict]:
    """Return list of {'t': ISO, 'v': float} for AI console storage."""
    if eq is None or len(eq) == 0:
        return []
    if isinstance(eq, pd.Series):
        items = eq.reset_index()
        # If index is datetime, keep ISO; otherwise just string index
        if isinstance(items.iloc[0, 0], pd.Timestamp):
            return [{"t": str(items.iloc[i, 0]), "v": float(items.iloc[i, 1])} for i in range(len(items))]
        else:
            return [{"t": str(items.iloc[i, 0]), "v": float(items.iloc[i, 1])} for i in range(len(items))]
    # fallback: plain list
    try:
        return [{"t": str(i), "v": float(v)} for i, v in enumerate(list(eq))]
    except Exception:
        return []

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
    "backtest_score": float(backtest_score),            # learning signal only
    "equity_curve": _serialize_equity(equity_series),   # <— stored for AI insights
}

st.session_state.setdefault("performance", {})
st.session_state["performance"]["backtest"] = perf_payload

st.success("✅ Backtest performance saved for AI Console learning (summary + equity curve).")
