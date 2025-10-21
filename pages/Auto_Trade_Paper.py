# pages/Auto_Trade_Paper.py
# -------------------------------------------------------------
# Full Auto Paper-Trade Simulation – BankNifty ATM Options
# Upgraded 2025-10-21  |  uses utils.fetch_smart()
# -------------------------------------------------------------

import streamlit as st, pandas as pd, numpy as np
from datetime import datetime
from utils import fetch_smart

st.set_page_config(page_title="Auto Paper Trade – BankNifty ATM", layout="wide")

st.title("🧪 Auto Paper-Trade — BankNifty ATM Options")
st.caption("5 min Price-Action 50–60 % entries • ATM Delta/Theta sim • Trailing SL rules")

# ----------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------
c_mode, c_sym, c_period, c_interval = st.columns([1.3, 1.3, 1, 1])
with c_mode:
    live_mode = st.radio("Mode", ["Live 5 min", "Offline Daily (Safe)"], index=0, horizontal=True)
with c_sym:
    symbol = st.text_input("Symbol", "NSEBANK.NS")
with c_period:
    prefer_period = st.selectbox("Period", ["5d","7d","14d","1mo","3mo"], index=0)
with c_interval:
    prefer_interval = st.selectbox("Interval", ["5m","15m","1h","1d"], index=0)

st.caption("Weekend or closed market → choose Offline Safe mode.")

st.markdown("---")

c_delta,c_theta,c_seed,c_lot = st.columns(4)
delta = c_delta.number_input("ATM Δ",0.1,0.9,0.50,0.01)
theta_bar = c_theta.number_input("θ per bar (pts)",0.0,5.0,0.00,0.05)
seed_prem = c_seed.number_input("Seed ATM premium (pts)",10.0,800.0,200.0,5.0)
lot = c_lot.number_input("Lot size",1,100,15)

c_sl,c_bias,c_align,c_cap = st.columns([1,1,1.3,1])
initial_sl = c_sl.number_input("Initial SL (pts)",5,50,10,1)
final_decision = st.session_state.get("final_decision",{})
bias_dash = final_decision.get("final_bias","HOLD")
c_bias.text_input("Dashboard Bias",bias_dash,disabled=True)
align_bias = c_align.checkbox("Align with Dashboard Bias",True)
capital = c_cap.number_input("Capital ₹ per trade",10000,1000000,50000,1000)

st.markdown("---")

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def generate_pa_signals(df, low=0.50, high=0.60):
    if df.empty or len(df)<3: return df
    x=df.copy()
    ph,pl=x['high'].shift(1),x['low'].shift(1)
    mid50,mid60=pl+(ph-pl)*low,pl+(ph-pl)*high
    cond_buy=(x['low'].between(mid50,mid60))&(x['close']>ph)
    cond_sell=(x['high'].between(mid50,mid60))&(x['close']<pl)
    x['signal']=np.select([cond_buy,cond_sell],["BUY","SELL"],"HOLD")
    return x

def simulate_premium_path(idx,delta,theta,start,side):
    prem=[start]
    for i in range(1,len(idx)):
        move=idx[i]-idx[i-1]
        if side=="PE": move=-move
        next_p=prem[-1]+delta*move-theta
        prem.append(max(0.5,next_p))
    return np.array(prem)

def apply_trailing_sl(entry,series,sl0):
    sl=entry-sl0
    for i in range(1,len(series)):
        p=float(series[i]); prof=p-entry
        if prof>=10: sl=max(sl,entry)
        if prof>=20: sl=max(sl,entry+10)
        if prof>=30: sl=max(sl,entry+15)
        if prof>=50: sl=max(sl,p-(prof*0.5))
        if p<=sl: return i,sl,sl-entry
    return len(series)-1,float(series[-1]),series[-1]-entry

def run_sim(df,delta,theta,seed,sl0,lot,align,bias,cap):
    trades=[]
    i=0
    while i<len(df)-1:
        sig=df['signal'].iloc[i]
        if sig=="HOLD": i+=1; continue
        if align:
            if (bias=="BUY" and sig!="BUY") or (bias=="SELL" and sig!="SELL") or bias=="HOLD":
                i+=1; continue
        entry_idx=i+1
        if entry_idx>=len(df): break
        entry_t=df.index[entry_idx]; side="CE" if sig=="BUY" else "PE"
        idx_slice=df['close'].values[entry_idx:]
        prem_path=simulate_premium_path(idx_slice,delta,theta,seed,side)
        exit_off,exit_px,pnl=apply_trailing_sl(seed,prem_path,sl0)
        exit_idx=min(entry_idx+exit_off,len(df)-1)
        exit_t=df.index[exit_idx]
        pnl_r=pnl*lot
        eq_chg=pnl_r/cap*100
        trades.append(dict(signal=sig,side=side,entry_time=entry_t,
            exit_time=exit_t,entry_px=seed,exit_px=exit_px,
            pnl_pts=pnl,pnl_rupees=pnl_r,eq_change=eq_chg))
        i=exit_idx+1
    return pd.DataFrame(trades)

# ----------------------------------------------------------------
# Fetch data
# ----------------------------------------------------------------
st.caption("Fetching candles…")
df_raw,msg=fetch_smart(symbol,(prefer_period,prefer_interval),
                       mode="auto" if live_mode=="Live 5 min" else "daily")
st.caption(msg)
if df_raw.empty:
    st.warning("⚠️ No data found. Try Offline Daily mode.")
    st.stop()

df_sig=generate_pa_signals(df_raw)
with st.expander("Recent signals (50 rows)"):
    st.dataframe(df_sig.tail(50))

bias=bias_dash.upper()
st.info(f"Bias used → {bias}")

# adaptive theta: stronger decay if not market hours
from utils import is_market_hours
if not is_market_hours():
    theta_bar*=1.5

with st.spinner("Simulating…"):
    trades=run_sim(df_sig,delta,theta_bar,seed_prem,initial_sl,lot,align_bias,bias,capital)

st.markdown("---")
st.subheader("📈 Simulation Results")

if trades.empty:
    st.warning("No trades triggered for this data.")
    st.stop()

wins=(trades.pnl_pts>0).sum(); loss=(trades.pnl_pts<=0).sum()
total=len(trades); winrate=wins/total*100
net_pts=trades.pnl_pts.sum(); net_r=trades.pnl_rupees.sum()
dd=np.minimum.accumulate(trades.cummin()["pnl_rupees"]) if "pnl_rupees" in trades else None

c1,c2,c3,c4=st.columns(4)
c1.metric("Trades",total); c2.metric("Win %",f"{winrate:.1f}")
c3.metric("Net PnL (pts)",f"{net_pts:.1f}")
c4.metric("Net PnL ₹",f"{net_r:,.0f}")

trades["cum_pts"]=trades.pnl_pts.cumsum()
trades["cum_rupees"]=trades.pnl_rupees.cumsum()

st.line_chart(trades.set_index(pd.to_datetime(trades.exit_time))["cum_pts"],height=200)

st.bar_chart(trades["pnl_pts"],height=150)

with st.expander("Trades"):
    st.dataframe(trades.sort_values("exit_time",ascending=False),use_container_width=True)

csv=trades.to_csv(index=False).encode()
st.download_button("⬇️ Download CSV",csv,"banknifty_paper_trades.csv","text/csv")

st.session_state["paper_trades"]=trades
st.success("✅ Trades stored → AI Console accessible for analysis.")
