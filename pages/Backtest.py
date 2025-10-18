import streamlit as st
import pandas as pd

from utils import (
    DEFAULT_SYMBOL,
    fetch_smart,
    generate_signals_50pct,
    simulate_atm_option_trades,
)

# ------------------------------------------------------
# Backtest – BankNifty ATM Options (with Data Mode Toggle)
# ------------------------------------------------------

st.title("🧪 Backtest (Trailing SL) — BankNifty ATM Options")

# ---------- TRADER-STYLE DATA MODE TOGGLE ----------
st.markdown("### 📊 Data Source Mode")
data_mode = st.radio(
    label="Pick data mode",
    options=["🔴 Live Intraday (5m)", "🟦 Offline / Daily (Safe)"],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
)

# ---------- Inputs ----------
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    # DEFAULT_SYMBOL should be "NSEBANK.NS" in utils.py (cloud-safe)
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

# ---------- Data Fetch (Intraday or Daily, with safe fallbacks) ----------
with st.spinner("Fetching market data..."):
    if data_mode.startswith("🔴"):
    # Live intraday with fallback
    df, used = fetch_smart(symbol, prefer=(prefer_period, prefer_interval))
else:
    # Direct Daily Mode (Cloud Safe)
    import yfinance as yf
    df = yf.download(symbol, period="3mo", interval="1d", progress=False, auto_adjust=True)
    used = ("3mo", "1d")

st.caption(f"Using: period={used[0]}  interval={used[1]}")

if df is None or df.empty:
    st.error("⚠ No data available even after fallbacks.")
    st.stop()

# ---------- Signal Generation (Previous Candle 50% Rule) ----------
sig_df = generate_signals_50pct(df, mid_factor=0.5)

# Quick view of last few signals
with st.expander("Recent Signals (last 12)", expanded=False):
    st.dataframe(sig_df[["close", "signal"]].tail(12), use_container_width=True)

# ---------- Simulation ----------
st.subheader("Simulation Results")

if pricing_mode.startswith("ATM"):
    tr, equity, score = simulate_atm_option_trades(
        sig_df,
        signals_col="signal",
        init_sl_pts=init_sl,
        lot_size=lot,
        mode="delta",               # delta-based ATM model
        theta_per_candle=theta,
    )
else:
    # Index proxy (delta ~ 1, no decay) using same simulator for consistency
    tr, equity, score = simulate_atm_option_trades(
        sig_df,
        signals_col="signal",
        init_sl_pts=init_sl,
        lot_size=lot,
        mode="index",
        theta_per_candle=0.0,
    )

if tr is None or tr.empty:
    st.info("No closed trades in this range yet. Try a longer period or different interval.")
    # Save neutral score so Dashboard remains consistent
    st.session_state.setdefault("scores", {})
    st.session_state["scores"]["backtest"] = 50.0
else:
    # Compute quick stats
    wins = int((tr["pnl_points"] > 0).sum())
    losses = int((tr["pnl_points"] <= 0).sum())
    total = int(len(tr))
    winrate = (wins / total) * 100.0 if total > 0 else 0.0
    pnl_total = float(tr["pnl_points"].sum())

    cA, cB, cC, cD = st.columns(4)
    cA.metric("Trades", total)
    cB.metric("Win Rate", f"{winrate:.1f}%")
    cC.metric("Total PnL (₹)", f"{pnl_total:,.0f}")
    cD.metric("Backtest Score", f"{score:.0f}/100")

    st.dataframe(tr.tail(50), use_container_width=True)

    if equity is not None and not isinstance(equity, pd.Series) and "cum_pnl" in tr.columns:
        # Backward compatibility if equity wasn’t returned as Series
        st.line_chart(tr["cum_pnl"], height=220)
    elif equity is not None and not equity.empty:
        st.line_chart(equity, height=220)

    # Save score for Dashboard fusion
    st.session_state.setdefault("scores", {})
    st.session_state["scores"]["backtest"] = float(score)

st.success("✅ Backtest score updated for Dashboard fusion.")
