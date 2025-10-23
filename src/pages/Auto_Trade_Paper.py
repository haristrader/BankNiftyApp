# pages/Auto_Trade_Paper.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt
from datetime import datetime
from src.utils import *
from src.data_engine import *

st.set_page_config(page_title="💰 Auto Paper Trade – BankNifty ATM", layout="wide")
st.title("💹 Auto Paper-Trade (BankNifty ATM Options)")
st.caption("5m Price-Action Entry | Adaptive Theta | Trailing SL | Virtual Capital Simulation")

# Parameters
col1, col2, col3 = st.columns([1, 1, 1])
symbol = col1.text_input("Symbol", "NSEBANK.NS")
period = col2.selectbox("Period", ["5d", "7d", "14d", "1mo"], index=0)
interval = col3.selectbox("Interval", ["5m", "15m"], index=0)

col4, col5, col6 = st.columns(3)
delta = col4.number_input("Delta", 0.1, 1.0, 0.5, 0.05)
theta_bar = col5.number_input("Theta (per bar)", 0.0, 5.0, 0.2, 0.05)
initial_sl = col6.number_input("Initial SL (pts)", 5, 50, 10)

col7, col8, col9 = st.columns(3)
seed_prem = col7.number_input("ATM Premium (seed)", 50.0, 500.0, 200.0, 5.0)
lot = col8.number_input("Lot Size", 1, 50, 15)
if "virtual_capital" not in st.session_state:
    st.session_state["virtual_capital"] = 10000.0
capital = col9.number_input("Virtual Capital (₹)", 1000.0, 1000000.0, st.session_state["virtual_capital"], step=500.0)

st.markdown("---")
st.subheader("📊 Fetching Market Data...")

df, used, msg = fetch_smart(symbol, prefer=(period, interval))
st.caption(msg)

# define df_raw to avoid NameError
df_raw = df.copy() if df is not None else pd.DataFrame()
if df_raw.empty:
    st.error("⚠️ No data found. Try daily or larger period.")
    st.stop()

# Signal logic (keeps same)
def generate_pa_signals(df, low=0.50, high=0.60):
    df = df.copy()
    ph, pl = df["high"].shift(1), df["low"].shift(1)
    mid50, mid60 = pl + (ph - pl) * low, pl + (ph - pl) * high
    cond_buy = (df["low"].between(mid50, mid60)) & (df["close"] > ph)
    cond_sell = (df["high"].between(mid50, mid60)) & (df["close"] < pl)
    df["signal"] = np.select([cond_buy, cond_sell], ["BUY", "SELL"], "HOLD")
    return df

df = generate_pa_signals(df_raw)
st.success("✅ Signals generated.")

def simulate_trades(df, delta, theta, seed, sl0, lot, capital):
    trades = []
    eq = capital
    i = 0
    while i < len(df) - 1:
        if eq <= 0:
            st.warning("💀 Capital exhausted. Simulation stopped.")
            break
        sig = df["signal"].iloc[i]
        if sig == "HOLD":
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= len(df):
            break
        entry_time = df.index[entry_idx]
        entry_price = seed
        side = "CE" if sig == "BUY" else "PE"

        idx_prices = df["close"].iloc[entry_idx:].values
        prem = [entry_price]
        for j in range(1, len(idx_prices)):
            move = idx_prices[j] - idx_prices[j - 1]
            move = move if side == "CE" else -move
            next_p = prem[-1] + delta * move - theta
            prem.append(max(0.5, next_p))

        sl = entry_price - sl0
        pnl, exit_idx = 0, None
        for k in range(1, len(prem)):
            p = prem[k]
            profit = p - entry_price
            if profit >= 10 and sl < entry_price:
                sl = entry_price
            if profit >= 20 and sl < entry_price + 10:
                sl = entry_price + 10
            if profit >= 30 and sl < entry_price + 15:
                sl = entry_price + 15
            if profit >= 50:
                sl = max(sl, p - (profit * 0.5))
            if p <= sl:
                pnl = p - entry_price
                exit_idx = entry_idx + k
                break

        if exit_idx is None:
            exit_idx = len(df) - 1
            pnl = prem[-1] - entry_price

        exit_time = df.index[exit_idx]
        pnl_pts = pnl
        pnl_rupees = pnl_pts * lot
        eq += pnl_rupees

        trades.append({
            "signal": sig,
            "side": side,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "pnl_pts": pnl_pts,
            "pnl_rupees": pnl_rupees,
            "capital": eq,
        })
        i = exit_idx + 1
    return pd.DataFrame(trades), eq

if not is_market_hours_ist():
    theta_bar *= 1.3

with st.spinner("Running paper-trade simulation..."):
    trades, final_capital = simulate_trades(df, delta, theta_bar, seed_prem, initial_sl, lot, capital)

if trades is None or trades.empty:
    st.warning("No trades generated.")
    st.stop()

st.subheader("📈 Trade Summary")
wins = (trades["pnl_pts"] > 0).sum()
losses = (trades["pnl_pts"] <= 0).sum()
net_pnl = trades["pnl_rupees"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trades", len(trades))
c2.metric("Win Rate", f"{wins / max(1, len(trades)) * 100:.1f}%")
c3.metric("Net PnL (₹)", f"{net_pnl:,.0f}")
c4.metric("Final Capital (₹)", f"{final_capital:,.0f}")

st.session_state["virtual_capital"] = final_capital

st.markdown("---")
st.subheader("🕯️ Candle Chart with Entry/Exit Points")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["close"], linewidth=1)
ax.grid(True, linestyle="--", alpha=0.3)

for _, row in trades.iterrows():
    entry_t, exit_t = row["entry_time"], row["exit_time"]
    try:
        entry_px = df.loc[entry_t, "close"]
        exit_px = df.loc[exit_t, "close"]
        ax.scatter(entry_t, entry_px, color="lime", marker="^", s=80)
        ax.scatter(exit_t, exit_px, color="red", marker="v", s=80)
    except Exception:
        pass

ax.set_title(f"{symbol}  |  Trades: {len(trades)}  |  Capital: ₹{final_capital:,.0f}")
st.pyplot(fig)

st.markdown("---")
st.dataframe(trades, use_container_width=True)
st.download_button(
    "⬇️ Download Trade Log (CSV)",
    trades.to_csv(index=False).encode("utf-8"),
    file_name="paper_trades.csv",
    mime="text/csv"
)
