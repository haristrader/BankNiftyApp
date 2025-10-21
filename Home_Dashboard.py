# Home_Dashboard.py
# ----------------------------------------------------------
# Master Dashboard — BankNifty Algo Terminal (Fusion Engine)
# ----------------------------------------------------------
# Combines Trend / Fibonacci / Smart Money / Bank Impact scores
# into a single weighted 0–100 decision score.
# Saves output → AI Console (session_state["final_decision"])
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="🏠 BankNifty Master Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏠 Master Dashboard — BankNifty Algo Terminal")
st.caption("Open pages: Trend, Fibonacci, Smart Money, Bank Impact → run once to load their data.")

# ----------------------- Helpers -----------------------
def _get_perf():
    return st.session_state.get("performance", {})

def _get_score(perf: dict, key: str):
    """returns (score, bias, meta, ok)"""
    obj = perf.get(key, {})
    sc = obj.get("final_score")
    bias = obj.get("bias") or obj.get("trend") or obj.get("fib_bias")
    meta = {
        "symbol": obj.get("symbol", "^NSEBANK"),
        "tf": obj.get("tf", "-"),
        "mode": obj.get("mode") or ("daily" if obj.get("used", ["", "1d"])[-1] == "1d" else "intraday"),
        "used": obj.get("used", "-"),
    }
    ok = sc is not None
    return sc, bias, meta, ok

def _badge(meta):
    if not meta:
        return ""
    mode = str(meta.get("mode", "")).lower()
    if mode == "intraday":
        return "🟢 Live"
    elif mode == "daily":
        return "🟦 Daily"
    return "⚪"

def _box(title, score, bias, meta):
    with st.container(border=True):
        cols = st.columns([3, 1])
        with cols[0]:
            st.subheader(title)
            st.caption(f"{meta.get('symbol', '')} • TF: {meta.get('tf', '-')} • {_badge(meta)}")
        with cols[1]:
            st.metric("Score (0–100)", f"{score:.0f}" if score is not None else "—")
        if bias:
            b = bias.upper()
            if b == "BUY":
                st.success(f"Bias: {b}")
            elif b == "SELL":
                st.error(f"Bias: {b}")
            else:
                st.info(f"Bias: {b}")
        else:
            st.info("Bias: —")

# ----------------------- Sidebar Weights -----------------------
st.sidebar.header("⚖️ Module Weights (Fusion Engine)")
w_trend = st.sidebar.slider("Trend", 0, 40, 20)
w_fib = st.sidebar.slider("Fibonacci", 0, 40, 25)
w_sm = st.sidebar.slider("Smart Money", 0, 40, 20)
w_bankimpact = st.sidebar.slider("Bank Impact", 0, 30, 10)
w_others = st.sidebar.slider("Others", 0, 30, 10)

st.sidebar.caption("Price Action executes AFTER bias confirmation. Not included in score fusion.")

# ----------------------- Pull module scores -----------------------
perf = _get_perf()

sc_trend, b_trend, m_trend, ok_trend = _get_score(perf, "trend")
sc_fib, b_fib, m_fib, ok_fib = _get_score(perf, "fibonacci")
sc_sm, b_sm, m_sm, ok_sm = _get_score(perf, "smartmoney")
sc_bank, b_bank, m_bank, ok_bank = _get_score(perf, "bankimpact")

# ----------------------- Header Cards -----------------------
st.markdown("### Module Snapshots")
c1, c2, c3, c4 = st.columns(4)
with c1: _box("Trend", sc_trend, b_trend, m_trend)
with c2: _box("Fibonacci", sc_fib, b_fib, m_fib)
with c3: _box("Smart Money", sc_sm, b_sm, m_sm)
with c4: _box("Bank Impact", sc_bank, b_bank, m_bank)

# ----------------------- Final Score Fusion -----------------------
st.markdown("---")
st.subheader("🎯 Final Decision (Weighted Fusion)")

items = []
if ok_trend and w_trend > 0: items.append(("Trend", sc_trend, w_trend))
if ok_fib and w_fib > 0: items.append(("Fibonacci", sc_fib, w_fib))
if ok_sm and w_sm > 0: items.append(("Smart Money", sc_sm, w_sm))
if ok_bank and w_bankimpact > 0: items.append(("Bank Impact", sc_bank, w_bankimpact))

if not items:
    st.warning("No module scores found yet. Open each module (Trend, Fib, SM, Bank) once to populate.")
    st.stop()

num = sum(s * w for _, s, w in items)
den = sum(w for _, _, w in items)
final_score = max(0.0, min(100.0, num / den if den > 0 else 0.0))

if final_score >= 65:
    final_bias = "BUY"
elif final_score <= 35:
    final_bias = "SELL"
else:
    final_bias = "HOLD"

st.progress(int(final_score))
st.write(f"**Final Score:** `{final_score:.1f}/100`  |  **Bias:** `{final_bias}`")

if final_bias == "BUY":
    st.success("BUY side active ✅ → Use Price Action (5m, 50–60%) entries. Trail SL 10→BE, 20→+10, 30→+15, 50→Lock 50%.")
elif final_bias == "SELL":
    st.error("SELL side active ⚠️ → Use Price Action (5m, 50–60%) shorts. Apply trailing SL rules.")
else:
    st.info("Neutral zone ⚪ → Wait for clarity. Only take high conviction setups.")

# breakdown table
with st.expander("📊 Weight Breakdown"):
    df = pd.DataFrame([{"Module": n, "Score": s, "Weight": w, "Weighted": s*w} for n, s, w in items])
    df["Score"] = df["Score"].round(1)
    df["Weighted"] = df["Weighted"].round(1)
    st.dataframe(df, use_container_width=True)

# ----------------------- Save decision for AI Console -----------------------
st.session_state["final_decision"] = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "final_score": float(final_score),
    "final_bias": final_bias,
    "weights": {
        "trend": w_trend,
        "fibonacci": w_fib,
        "smartmoney": w_sm,
        "bankimpact": w_bankimpact,
        "others": w_others,
    },
    "available": [n for n, _, _ in items],
}
st.caption("✅ Final decision saved — AI Console updated in memory.")
