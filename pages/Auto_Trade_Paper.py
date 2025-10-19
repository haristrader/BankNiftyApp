# pages/Auto_Trade_Paper.py
# -------------------------------------------------------------
# Full Auto Paper-Trade Simulation – BankNifty ATM Options
# - Entries from 5m Price Action (50–60% previous candle mid-zone rule)
# - ATM Delta simulation of option premium + theta decay
# - Trailing SL rules: 10→BE, 20→+10, 30→+15, 50→lock 50%
# - Optional alignment with Dashboard Final Bias
# - Weekend/daily fallback supported
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Auto Paper Trade – BankNifty ATM", layout="wide")

# ----------------------- UI -----------------------
st.title("🧪 Backtest / Paper Trade — BankNifty ATM Options (Auto)")
st.caption("Uses 5m Price Action (50–60% rule) entries, ATM delta model, and trailing SL rules.")

# Data mode
st.subheader("📊 Data Mode")
c_mode, c_sym, c_period, c_interval = st.columns([1.2, 1.2, 1, 1])
with c_mode:
    use_live = st.radio("Mode", ["Live Intraday (5m)", "Offline / Daily (Safe)"], index=0, horizontal=True)
with c_sym:
    symbol = st.text_input("Index Symbol", value="NSEBANK.NS")
with c_period:
    prefer_period = st.selectbox("Preferred Period", ["5d", "7d", "14d", "1mo", "3mo"], index=0)
with c_interval:
    prefer_interval = st.selectbox("Preferred Interval", ["5m", "15m", "1h", "1d"], index=0)

st.caption("Weekend or market closed? Use **Offline/Daily (Safe)** to run a coarse simulation.")

st.markdown("---")

# Pricing mode
st.subheader("💹 Option Pricing (ATM Delta Sim)")
c_delta, c_theta, c_seed, c_lot = st.columns(4)
with c_delta:
    delta = st.number_input("ATM delta (~0.50)", min_value=0.1, max_value=0.9, value=0.50, step=0.01)
with c_theta:
    theta_per_bar = st.number_input("Theta per bar (pts)", min_value=0.0, max_value=5.0, value=0.00, step=0.05)
with c_seed:
    seed_premium = st.number_input("Seed ATM premium at entry (pts)", min_value=10.0, max_value=800.0, value=200.0, step=5.0)
with c_lot:
    lot_size = st.number_input("Lot size (BankNifty)", min_value=1, max_value=100, value=15)

c_sl, c_bias, c_align, c_maxtr = st.columns([1, 1, 1.3, 1])
with c_sl:
    initial_sl = st.number_input("Initial SL (option pts)", min_value=5, max_value=50, value=10, step=1)
with c_bias:
    # Pull from Dashboard if available
    final_decision = st.session_state.get("final_decision", {})
    bias_str = final_decision.get("final_bias", "HOLD")
    st.text_input("Dashboard Bias (from Home)", value=bias_str, key="bias_display", disabled=True)
with c_align:
    align_with_bias = st.checkbox("Only take signals aligned with Dashboard Bias", value=True)
with c_maxtr:
    max_trades = st.number_input("Max trades (cap)", min_value=1, max_value=500, value=100, step=5)

st.markdown("---")

# ----------------------- Data helpers -----------------------
def fetch(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV, tz-naive, clean columns."""
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[['Open','High','Low','Close','Volume']].dropna().copy()
    try:
        df.index = pd.to_datetime(df.index).tz_convert(None)
    except Exception:
        df.index = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]
    return df

def pick_mode(prefer_period: str, prefer_interval: str, use_live: bool):
    if use_live:
        # Force 5m for execution logic
        return "7d", "5m"
    # Daily safe
    return "3mo", "1d"

# ----------------------- Price Action signal -----------------------
def generate_pa_signals(df: pd.DataFrame, mid_low=0.50, mid_high=0.60) -> pd.DataFrame:
    """
    BUY when current low >= prev_mid(50–60%) AND close > prev_high
    SELL when current high <= prev_mid AND close < prev_low
    """
    if df.empty or len(df) < 3:
        return df

    x = df.copy()
    prev_high = x['high'].shift(1)
    prev_low  = x['low'].shift(1)

    # dynamic band between 50% to 60% (use center 55% for trigger, but require low/high within band)
    prev_mid_50 = prev_low + (prev_high - prev_low) * mid_low
    prev_mid_60 = prev_low + (prev_high - prev_low) * mid_high
    prev_mid_center = prev_low + (prev_high - prev_low) * 0.55

    cond_buy  = (x['low'] >= prev_mid_50) & (x['low'] <= prev_mid_60) & (x['close'] > prev_high)
    cond_sell = (x['high'] <= prev_mid_60) & (x['high'] >= prev_mid_50) & (x['close'] < prev_low)

    signal = np.where(cond_buy, "BUY", np.where(cond_sell, "SELL", "HOLD"))
    x['signal'] = signal
    x['prev_mid'] = prev_mid_center
    return x

# ----------------------- Option premium simulator -----------------------
def simulate_premium_path(index_prices: np.ndarray, delta: float, theta: float, start_price: float, direction: str):
    """
    direction: 'CE' -> premium up when index up
               'PE' -> premium up when index down
    """
    prem = [start_price]
    for i in range(1, len(index_prices)):
        move = index_prices[i] - index_prices[i-1]
        if direction == "PE":
            move = -move
        next_p = prem[-1] + delta * move - theta
        prem.append(max(0.5, next_p))  # avoid zero premium
    return np.array(prem)

# ----------------------- Trailing SL engine -----------------------
def apply_trailing_sl(entry, series, initial_sl):
    """
    Given series of premium starting from entry bar (including entry),
    return (exit_index_offset, exit_price, pnl_points)
    Trail rules:
      >=10 → BE ; >=20 → +10 ; >=30 → +15 ; >=50 → lock 50%
    """
    sl = entry - initial_sl
    for i in range(1, len(series)):
        price = float(series[i])
        profit = price - entry
        # step-ups
        if profit >= 10 and sl < entry:
            sl = entry
        if profit >= 20 and sl < entry + 10:
            sl = entry + 10
        if profit >= 30 and sl < entry + 15:
            sl = entry + 15
        if profit >= 50:
            sl = max(sl, price - (profit * 0.5))
        # SL hit?
        if price <= sl:
            exit_px = sl
            return i, exit_px, exit_px - entry
    # no SL hit → exit at last
    exit_px = float(series[-1])
    return len(series)-1, exit_px, exit_px - entry

# ----------------------- Trade simulator -----------------------
def run_sim(df: pd.DataFrame,
            delta: float, theta: float, seed: float,
            initial_sl: int, lot_size: int,
            align_with_bias: bool, bias: str,
            max_trades: int):
    """
    - open only 1 position at a time
    - enter at next bar after signal
    - BUY -> long CE ; SELL -> long PE
    """
    trades = []
    if df.empty or len(df) < 20:
        return pd.DataFrame(trades)

    pos_open = False
    i = 0
    count = 0
    while i < len(df)-1 and count < max_trades:
        sig = df['signal'].iloc[i]
        if sig == "HOLD" or pd.isna(sig):
            i += 1
            continue

        # bias filtering
        if align_with_bias:
            if bias == "BUY" and sig != "BUY":
                i += 1; continue
            if bias == "SELL" and sig != "SELL":
                i += 1; continue
            if bias == "HOLD":
                i += 1; continue

        # enter at next bar close
        entry_idx = i + 1
        if entry_idx >= len(df):
            break
        entry_time = df.index[entry_idx]
        side = "CE" if sig == "BUY" else "PE"
        # simulate option premium from entry to the end (or until exit)
        idx_slice = df['close'].values[entry_idx:]
        prem_path = simulate_premium_path(idx_slice, delta, theta, seed, side)
        exit_off, exit_px, pnl_pts = apply_trailing_sl(seed, prem_path, initial_sl)
        exit_idx = entry_idx + exit_off
        exit_time = df.index[exit_idx]

        trades.append({
            "signal": sig,
            "side": "LONG " + side,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_px": float(seed),
            "exit_px": float(exit_px),
            "pnl_pts": float(pnl_pts),
            "pnl_rupees": float(pnl_pts) * lot_size
        })
        count += 1
        # jump to bar after exit
        i = exit_idx + 1

    return pd.DataFrame(trades)

# ----------------------- RUN -----------------------
period, interval = pick_mode(prefer_period, prefer_interval, use_live=(use_live=="Live Intraday (5m)"))
st.caption(f"Using: period={period} interval={interval}")

with st.spinner("Fetching data..."):
    df_raw = fetch(symbol, period, interval)

if df_raw.empty:
    st.warning("⚠️ No data available (market closed or data source blocked). Try Offline/Daily mode.")
    st.stop()

# signals
df_sig = generate_pa_signals(df_raw)

# preview
with st.expander("Show last 50 rows (signals)"):
    st.dataframe(df_sig.tail(50))

# bias
bias_in = (final_decision.get("final_bias", "HOLD") or "HOLD").upper()

# run simulation
with st.spinner("Running paper-trade simulation..."):
    trades_df = run_sim(
        df_sig, delta=delta, theta=theta_per_bar, seed=seed_premium,
        initial_sl=initial_sl, lot_size=lot_size,
        align_with_bias=align_with_bias, bias=bias_in,
        max_trades=int(max_trades)
    )

st.markdown("---")
st.subheader("📈 Results")

if trades_df.empty:
    st.info("No closed trades in this simulation yet (filters/signals may be too strict for selected data).")
    st.stop()

# stats
wins = (trades_df['pnl_pts'] > 0).sum()
loss = (trades_df['pnl_pts'] <= 0).sum()
total = len(trades_df)
winrate = 100 * wins / total if total else 0
net_pts = trades_df['pnl_pts'].sum()
net_rupees = trades_df['pnl_rupees'].sum()

cA, cB, cC, cD = st.columns(4)
cA.metric("Closed trades", total)
cB.metric("Win rate", f"{winrate:.1f}%")
cC.metric("Total PnL (pts)", f"{net_pts:.2f}")
cD.metric("Total PnL (₹)", f"{net_rupees:,.0f}")

# equity curve
trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)
trades_df["cum_pts"] = trades_df["pnl_pts"].cumsum()
trades_df["cum_rupees"] = trades_df["pnl_rupees"].cumsum()

st.line_chart(trades_df.set_index(pd.to_datetime(trades_df["exit_time"]))["cum_pts"], height=220)

with st.expander("Trades (latest first)"):
    st.dataframe(trades_df.sort_values("exit_time", ascending=False), use_container_width=True)

# download
csv = trades_df.to_csv(index=False).encode()
st.download_button("Download trades CSV", data=csv, file_name="banknifty_paper_trades.csv", mime="text/csv")

# persist for AI
st.session_state["paper_trades"] = trades_df
st.success("✅ Paper-trade run saved to session (AI Console can read it).")
