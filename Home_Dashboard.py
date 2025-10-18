# Home_Dashboard.py — Final Decision Engine (BUY/SELL/HOLD with trader language)
import streamlit as st
import pandas as pd

# pull defaults for weights/symbol if present
try:
    from utils import WEIGHTS_DEFAULT, DEFAULT_SYMBOL
except Exception:
    WEIGHTS_DEFAULT = {
        "trend": 20, "fibonacci": 25, "priceaction": 15,
        "smartmoney": 20, "backtest": 10, "others": 10
    }
    DEFAULT_SYMBOL = "^NSEBANK"

st.set_page_config(page_title="BankNifty Terminal — Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🏠 Master Dashboard — BankNifty Algo Terminal")

# ---------- session bootstrap ----------
scores = st.session_state.get("scores", {})
st.session_state["scores"] = scores  # ensure it exists

if "weights" not in st.session_state:
    st.session_state["weights"] = WEIGHTS_DEFAULT.copy()
weights = st.session_state["weights"]

with st.sidebar:
    st.header("⚖️ Module Weights (impact on Final Score)")
    colA, colB = st.columns(2)
    with colA:
        weights["trend"]       = st.slider("Trend",        0, 100, weights["trend"])
        weights["fibonacci"]   = st.slider("Fibonacci",    0, 100, weights["fibonacci"])
        weights["priceaction"] = st.slider("Price Action", 0, 100, weights["priceaction"])
    with colB:
        weights["smartmoney"]  = st.slider("Smart Money",  0, 100, weights["smartmoney"])
        weights["backtest"]    = st.slider("Backtest",     0, 100, weights["backtest"])
        weights["others"]      = st.slider("Others",       0, 100, weights["others"])

st.caption("Tip: Open each page in the left sidebar once (Trend/Fibonacci/SmartMoney/PriceAction/Backtest) so scores populate here.")

# ---------- helper to compute final ----------
def compute_final(_scores: dict, _weights: dict) -> tuple[float, pd.DataFrame]:
    rows = []
    num, den = 0.0, 0.0
    for key in ["trend","fibonacci","priceaction","smartmoney","backtest"]:
        sc = _scores.get(key, None)
        w  = float(_weights.get(key, 0)) / 100.0
        contrib = (sc * w) if sc is not None else None
        if sc is not None:
            num += (sc * w)
            den += w
        rows.append({"Module": key.capitalize(), "Score": sc, "Weight%": _weights.get(key, 0), "Weighted": contrib})
    # include 'Others' bucket only for display (not a module score)
    rows.append({"Module": "Others", "Score": None, "Weight%": _weights.get("others", 0), "Weighted": None})

    final = round(num / den, 2) if den > 0 else 0.0
    df = pd.DataFrame(rows)
    return final, df

final_score, table = compute_final(scores, weights)

# ---------- headline cards ----------
cards = st.columns(5)
labels = [("Trend","trend"), ("Fibonacci","fibonacci"),
          ("Price Action","priceaction"), ("Smart Money","smartmoney"),
          ("Backtest","backtest")]
for (label, key), c in zip(labels, cards):
    v = scores.get(key)
    c.metric(label, f"{v:.0f}/100" if isinstance(v,(int,float)) else "—")

st.markdown("---")

# ---------- decision engine with trader language ----------
def build_rationale(sc: dict) -> list[str]:
    R = []
    t  = sc.get("trend")
    f  = sc.get("fibonacci")
    pa = sc.get("priceaction")
    sm = sc.get("smartmoney")
    bt = sc.get("backtest")

    if t is not None:
        if t >= 70:   R.append("Trend strong — bulls controlling structure.")
        elif t <= 30: R.append("Trend weak — bears dominant / momentum soft.")
        else:         R.append("Trend mixed — sideways/rotation possible.")

    if f is not None:
        if f >= 65:   R.append("Fib confluence supportive — 38.2–61.8 zone respected.")
        elif f <= 35: R.append("Fib alignment weak — key retracement not holding.")
        else:         R.append("Fib neutral — no strong cluster visible.")

    if pa is not None:
        if pa >= 70:  R.append("Price Action breakout valid (50% rule bias up).")
        elif pa <= 30:R.append("Price Action breakdown risk (50% rule bias down).")
        else:         R.append("Price Action neutral — wait for candle confirmation.")

    if sm is not None:
        if sm >= 65:  R.append("Smart Money accumulation signs — clean breaks, healthy volume.")
        elif sm <= 35:R.append("Distribution / fakeout risk — wick/volume mismatch.")
        else:         R.append("Smart Money neutral — no clear footprint.")

    if bt is not None:
        if bt >= 55:  R.append("Backtest regime supportive — recent win-rate acceptable.")
        elif bt <= 45:R.append("Backtest regime weak — keep risk tight.")
        else:         R.append("Backtest neutral — average edge.")

    return R

def decision_text(score: float) -> tuple[str, str]:
    if score >= 70:
        return ("BUY", "Market in **Bullish Control** — buying dips / breakout setups preferred.")
    elif score <= 30:
        return ("SELL", "Market in **Bearish Control** — sell rallies / breakdown setups favored.")
    else:
        return ("HOLD", "Neutral zone — let price confirm; protect capital and wait for alignment.")

# ---------- Decision Engine ----------
def decision_text(score: float):
    if score >= 70:
        return ("BUY", "Market in **Bullish Control** — dips may be buying opportunities.")
    elif score <= 30:
        return ("SELL", "Market in **Bearish Control** — rallies may face selling pressure.")
    else:
        return ("HOLD", "Neutral zone — wait for clearer confirmation before entering.")

# Get label and message
label, msg = decision_text(final_score)

st.markdown("### 🎯 Final Decision")
col_dec, col_gauge = st.columns([1, 3])

with col_dec:
    if label == "BUY":
        st.markdown(f"<h3 style='color:green;'>🟢 {label}</h3>", unsafe_allow_html=True)
    elif label == "SELL":
        st.markdown(f"<h3 style='color:red;'>🔴 {label}</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:orange;'>🟡 {label}</h3>", unsafe_allow_html=True)

with col_gauge:
    st.progress(int(final_score))
    st.write(f"**Final Score: {final_score}/100**")

st.write(msg)

# rationale bullets
with st.expander("📌 Why this decision? (Rationale)"):
    reasons = build_rationale(scores)
    if reasons:
        st.markdown("\n".join([f"- {r}" for r in reasons]))
    else:
        st.info("Open analysis pages to generate module scores first.")

# weighted table
st.markdown("### 📊 Module Weights & Contributions")
show = table.copy()
# pretty names
show["Module"] = show["Module"].str.replace("Priceaction","Price Action").str.replace("Smartmoney","Smart Money")
st.dataframe(show, use_container_width=True)

# next steps box
st.markdown("### 🧭 What to do now")
if label == "BUY":
    st.write("- Focus on **long setups** near EMA20 pullbacks or Fib 50–61.8 retrace.")
    st.write("- Use your trailing SL ladder: 10→BE, 20→+10, 30→+15, 50→trail 50% profit.")
elif label == "SELL":
    st.write("- Focus on **short setups** on lower-highs / breakdown retests.")
    st.write("- Keep risk tight; avoid fighting strong spikes; trail per ladder.")
else:
    st.write("- **Avoid fresh risk** until Trend + Price Action align.")
    st.write("- Watch Smart Money / wick+volume for fakeout filters.")

st.caption(f"Symbol: {DEFAULT_SYMBOL}  •  Scores update when you visit pages on the left.  •  Adjust weights to fit your style.")
