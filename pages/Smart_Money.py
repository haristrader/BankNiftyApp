# pages/03_Smart_Money.py
import streamlit as st, numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import fetch, DEFAULT_SYMBOL

st.title("🧠 Smart Money / Fake Breakout")

symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
period = st.selectbox("Period", ["7d","14d","30d"], index=0)
interval = st.selectbox("Interval", ["5m","15m"], index=0)

df = fetch(symbol, period, interval, auto_adjust=True)
if df.empty:
    st.error("No data."); st.stop()

df["body"] = (df["close"] - df["open"]).abs()
df["range"] = df["high"] - df["low"]
df["upper_wick"] = df["high"] - df[["close","open"]].max(axis=1)
df["lower_wick"] = df[["close","open"]].min(axis=1) - df["low"]
df["wick_ratio"] = np.where(df["range"]>0, (df["upper_wick"]+df["lower_wick"])/df["range"], 0)
df["avg_vol"] = df["volume"].rolling(20).mean()
df["vol_factor"] = df["volume"]/df["avg_vol"]

recent = df.tail(10)
upper_break = (recent["close"] > recent["high"].shift(1))
lower_break = (recent["close"] < recent["low"].shift(1))

fake_buy  = ((upper_break) & (recent["wick_ratio"]>0.4) & (recent["vol_factor"]<1)).sum()
fake_sell = ((lower_break) & (recent["wick_ratio"]>0.4) & (recent["vol_factor"]<1)).sum()
real_buy  = ((upper_break) & (recent["wick_ratio"]<0.3) & (recent["vol_factor"]>1.2)).sum()
real_sell = ((lower_break) & (recent["wick_ratio"]<0.3) & (recent["vol_factor"]>1.2)).sum()

score = 50 + (real_buy + real_sell - fake_buy - fake_sell)*7
score = float(np.clip(score, 0, 100))

bias = "NEUTRAL"
if real_buy > fake_buy and real_buy > real_sell: bias = "BUY"
elif real_sell > fake_sell and real_sell > real_buy: bias = "SELL"

st.metric("Smart Money / Breakout Score", score)
st.info(f"Bias: {bias}")
st.bar_chart(recent[["vol_factor","wick_ratio"]])

st.session_state.setdefault("scores", {})["smartmoney"] = score
st.success("SmartMoney score saved to Dashboard.")
