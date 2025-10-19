# Home_Dashboard.py
# ----------------------------------------------------------
# Master Dashboard – BankNifty Algo Terminal (Fusion Engine)
# Merges module scores (Trend / Fibonacci / SmartMoney / BankImpact)
# into a single 0–100 final score + BUY/SELL/HOLD bias.
# Saves decision for AI Console via st.session_state["final_decision"].
# Weekend-safe: handles missing modules & prompts user to open pages.
# ----------------------------------------------------------
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Master Dashboard — BankNifty Algo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏠 Master Dashboard — BankNifty Algo Terminal")
st.caption("Tip: Trend, Fibonacci, SmartMoney, Bank Impact pages ko left sidebar se ek baar open kar lo — unke scores yahan populate ho jayenge.")

# ----------------------- Helpers -----------------------
def _get_perf():
    return st.session_state.get("performance", {})

def _get_score(perf: dict, key: str):
    """returns (score, bias, meta-dict, available:bool)"""
    obj = perf.get(key, {})
    sc  = obj.get("final_score", None)
    bias = obj.get("bias", None) or obj.get("trend", None) or obj.get("fib_bias", None)
    # small meta for badges
    meta = {
        "symbol": obj.get("symbol"),
        "tf": obj.get("tf"),
        "mode": obj.get("mode") or ("daily" if obj.get("used", ["","1d"])[-1] == "1d" else "intraday"),
        "used": obj.get("used"),
    }
    ok = (sc is not None)
    return sc, bias, meta, ok

def _badge(meta):
    if not meta: 
        return ""
    mode = str(meta.get("mode") or "").lower()
    if mode == "intraday":
        return "🔴 Live"
    elif mode == "daily":
        return "🟦 Daily"
    return ""

def _box(title, score, bias, meta, color="gray"):
    with st.container(border=True):
        cols = st.columns([3,1])
        with cols[0]:
            st.subheader(title)
            st.caption(f"{meta.get('symbol','')} • TF: {meta.get('tf','-')} • {_badge(meta)}")
        with cols[1]:
            if score is not None:
                st.metric("Score (0–100)", f"{score:.0f}")
            else:
                st.metric("Score (0–100)", "—")
        # bias line
        if bias:
            if str(bias).upper() == "BUY":
                st.success(f"Bias: {bias}")
            elif str(bias).upper() == "SELL":
                st.error(f"Bias: {bias}")
            else:
                st.info(f"Bias: {bias}")
        else:
            st.info("Bias: —")

# ----------------------- Sidebar Weights -----------------------
st.sidebar.header("⚖️ Module Weights (for Final Score)")
w_trend      = st.sidebar.slider("Trend", 0, 40, 20)
w_fib        = st.sidebar.slider("Fibonacci", 0, 40, 25)
w_sm         = st.sidebar.slider("Smart Money", 0, 40, 20)
w_bankimpact = st.sidebar.slider("Bank Impact", 0, 30, 10)
w_others     = st.sidebar.slider("Others (buffer)", 0, 30, 10)

st.sidebar.caption("Note: Price Action entry/exit execution module ka score nahi hota — woh final bias ke baad trigger hota hai.")

# ----------------------- Pull module scores -----------------------
perf = _get_perf()

sc_trend, b_trend, m_trend, ok_trend = _get_score(perf, "trend")
sc_fib,   b_fib,   m_fib,   ok_fib   = _get_score(perf, "fibonacci")
sc_sm,    b_sm,    m_sm,    ok_sm    = _get_score(perf, "smartmoney")
sc_bank,  b_bank,  m_bank,  ok_bank  = _get_score(perf, "bankimpact")

# ----------------------- Header Cards -----------------------
st.markdown("### Module Snapshots")
c1, c2, c3, c4 = st.columns(4)

with c1:
    _box("Trend", sc_trend, b_trend, m_trend)
with c2:
    _box("Fibonacci", sc_fib, b_fib, m_fib)
with c3:
    _box("Smart Money", sc_sm, b_sm, m_sm)
with c4:
    _box("Bank Impact", sc_bank, b_bank, m_bank)

# ----------------------- Final Score Fusion -----------------------
st.markdown("---")
st.subheader("🎯 Final Decision")

# collect available scores + weights
items = []
if ok_trend and w_trend > 0:
    items.append(("Trend", sc_trend, w_trend))
if ok_fib and w_fib > 0:
    items.append(("Fibonacci", sc_fib, w_fib))
if ok_sm and w_sm > 0:
    items.append(("Smart Money", sc_sm, w_sm))
if ok_bank and w_bankimpact > 0:
    items.append(("Bank Impact", sc_bank, w_bankimpact))

if not items:
    st.warning("Abhi tak kisi module ka score available nahi. 👉 Trend / Fibonacci / Smart Money / Bank Impact pages ko ek baar open/run kar lo.")
    st.stop()

# compute weighted score
num = sum(s * w for _, s, w in items)
den = sum(w for _, _, w in items)
final_score = num / den if den > 0 else 0.0
final_score = max(0.0, min(100.0, final_score))

# final bias thresholds
if final_score >= 65:
    final_bias = "BUY"
elif final_score <= 35:
    final_bias = "SELL"
else:
    final_bias = "HOLD"

# linear gauge
st.progress(int(final_score))
st.write(f"**Final Score:** `{final_score:.1f} / 100`  |  **Bias:** `{final_bias}`")

# Trader-style guidance
if final_bias == "BUY":
    st.success("Plan: BUY-side only. Price Action (5m 50–60% mid-zone) se entries lo. SL trail: 10→BE, 20→+10, 30→+15, 50→lock 50%.")
elif final_bias == "SELL":
    st.error("Plan: SELL-side only. Price Action (5m 50–60% mid-zone) se short entries lo. SL trail rules apply.")
else:
    st.info("Plan: HOLD/Skip until clarity. Sirf high-confidence Price Action signal par small risk lena.")

# breakdown table
with st.expander("Weight Breakdown"):
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
        "trend": w_trend, "fibonacci": w_fib, "smartmoney": w_sm,
        "bankimpact": w_bankimpact, "others": w_others
    },
    "available": [n for n, _, _ in items],
}
st.caption("✅ Final decision saved. AI Console is now aware.")
