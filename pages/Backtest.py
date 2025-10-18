import streamlit as st
import pandas as pd
from utils import (
    DEFAULT_SYMBOL,
    fetch_smart,
    generate_signals_50pct,
    simulate_atm_option_trades,
)

st.title("🧪 Backtest (Trailing SL) — BankNifty ATM Options")

# ---------- Controls ----------
c1, c2, c3 = st.columns([2,1,1])
with c1:
    symbol = st.text_input("Symbol", value="NSEBANK.NS")
with c2:
    prefer_period = st.selectbox("Preferred Period", ["5d","14d","30d"], index=0)
with c3:
    prefer_interval = st.selectbox("Preferred Interval", ["5m","15m","30m","60m","1d"], index=0)

mode = st.radio("Pricing Mode", ["ATM Delta (Sim)","Index Proxy"], index=0, horizontal=True)
init_sl = st.number_input("Initial SL (option points)", min_value=5, max_value=100, value=10, step=1)
lot = st.number_input("Lot Size", min_value=1, max_value=100, value=15, step=1)
theta = st.number_input("Theta per candle (sim)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

# ---------- Data ----------
with st.spinner("Fetching data ..."):
    df, used = fetch_smart(symbol, prefer=(prefer_period, prefer_interval))
st.caption(f"Using: period={used[0]}  interval={used[1]}")

if df.empty:
    st.error("No data available even after fallbacks.")
    st.stop()

# ---------- Signals ----------
sig_df = generate_signals_50pct(df, mid_factor=0.5)
st.write("Recent Signals:", sig_df[["close","signal"]].tail(10))

# ---------- Simulate ----------
st.subheader("Simulation Results")
if mode.startswith("ATM"):
    tr, equity, score = simulate_atm_option_trades(
        sig_df, signals_col="signal", init_sl_pts=init_sl, lot_size=lot,
        mode="delta", theta_per_candle=theta
    )
else:
    # Use index-proxy but via same function (delta=1, no decay)
    tr, equity, score = simulate_atm_option_trades(
        sig_df, signals_col="signal", init_sl_pts=init_sl, lot_size=lot,
        mode="index", theta_per_candle=0.0
    )

if tr.empty:
    st.info("No closed trades in this simulation range.")
else:
    cA, cB, cC, cD = st.columns(4)
    wins = int((tr["pnl_points"] > 0).sum())
    losses = int((tr["pnl_points"] <= 0).sum())
    total = len(tr)
    winrate = (wins/total)*100 if total>0 else 0
    pnl_total = float(tr["pnl_points"].sum())

    cA.metric("Trades", total)
    cB.metric("Win Rate", f"{winrate:.1f}%")
    cC.metric("Total PnL (₹)", f"{pnl_total:,.0f}")
    cD.metric("Backtest Score", f"{score:.0f}/100")

    st.dataframe(tr.tail(50), use_container_width=True)
    if not equity.empty:
        st.line_chart(equity, height=220)

# ---------- Save score to dashboard ----------
st.session_state.setdefault("scores", {})
st.session_state["scores"]["backtest"] = float(score)
st.success("✅ Backtest score updated to Dashboard")
