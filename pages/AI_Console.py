# pages/06_AI_Console.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="🤖 AI Learning Console", layout="wide")

# -------------------------
# Helpers / Defaults
# -------------------------
MODULES = ["trend", "fibonacci", "smartmoney", "bankimpact", "priceaction", "backtest", "others"]

def _get_perf_store():
    return st.session_state.get("performance", {})

def _get_weights():
    fd = st.session_state.get("final_decision", {})
    return fd.get("weights", {"trend":20,"fibonacci":25,"smartmoney":20,"bankimpact":10,"others":10})

def _ensure_history():
    if "module_history" not in st.session_state:
        # module_history is dict[module] -> list of (iso_ts, score)
        st.session_state["module_history"] = {m: [] for m in MODULES}
    return st.session_state["module_history"]

def _ensure_suggestions():
    if "ai_suggestions" not in st.session_state:
        st.session_state["ai_suggestions"] = []
    return st.session_state["ai_suggestions"]

def add_history_point(module: str, ts: str, score: float):
    hist = _ensure_history()
    hist.setdefault(module, []).append((ts, float(score)))
    # keep only last 1000 entries per module to avoid bloat
    if len(hist[module]) > 2000:
        hist[module] = hist[module][-2000:]

def history_df(module: str) -> pd.DataFrame:
    hist = _ensure_history().get(module, [])
    if not hist:
        return pd.DataFrame(columns=["ts","score"])
    df = pd.DataFrame(hist, columns=["ts","score"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    return df

def suggest_weights(current_weights: dict, perf_averages: dict) -> dict:
    """
    Simple suggestion logic:
      - If recent avg falls >10% vs prior avg => reduce weight by 4 (or 10% of weight)
      - If recent avg improves >10% => increase weight by 3 (or 5% of weight)
    Then normalize to sum 100.
    """
    suggested = current_weights.copy()
    # baseline percent thresholds
    for m, curr_w in current_weights.items():
        v = perf_averages.get(m, {})
        recent = v.get("recent", None)
        prior = v.get("prior", None)
        if recent is None or prior is None:
            continue
        # relative change
        if prior == 0:
            continue
        change = (recent - prior) / (prior + 1e-9)
        if change <= -0.10:
            # weaken
            delta = max(1, round(curr_w * 0.10))  # 10% of weight or at least 1
            suggested[m] = max(0, curr_w - delta)
        elif change >= 0.10:
            delta = max(1, round(curr_w * 0.05))  # small bump
            suggested[m] = min(100, curr_w + delta)
    # normalize to 100 while preserving relative sizes
    total = sum(suggested.values()) or 1
    norm = {k: round(v * 100.0 / total, 1) for k, v in suggested.items()}
    # small correction to make sum 100
    diff = 100.0 - sum(norm.values())
    if abs(diff) >= 0.1:
        # add leftover to the module with highest current weight
        max_k = max(norm.items(), key=lambda x: x[1])[0]
        norm[max_k] = round(norm[max_k] + diff, 1)
    return norm

def compute_perf_averages(window_recent=14, window_prior=14):
    """
    For each module, compute recent avg (last window_recent points)
    and prior avg (previous window_prior points).
    """
    averages = {}
    hist = _ensure_history()
    for m in MODULES:
        df = history_df(m)
        if df.empty or len(df) < 3:
            averages[m] = {"recent": None, "prior": None, "n_recent": 0, "n_prior": 0}
            continue
        # use last window_recent points for recent
        recent_df = df.tail(window_recent)
        prior_df = df.iloc[max(0, len(df) - window_recent - window_prior): len(df) - window_recent]
        ravg = float(recent_df["score"].mean()) if not recent_df.empty else None
        pavg = float(prior_df["score"].mean()) if not prior_df.empty else None
        averages[m] = {"recent": ravg, "prior": pavg, "n_recent": len(recent_df), "n_prior": len(prior_df)}
    return averages

# -------------------------
# UI Layout
# -------------------------
st.title("🤖 AI Learning Console (Phase 1)")

st.markdown(
    """
This console monitors module performance and suggests weight changes.
- Upload historical module scores (CSV) or let modules populate `st.session_state["module_history"]` during normal use.
- Suggestions are **only proposals**. The system will ask before applying any change.
"""
)

# load existing performance & weights
perf = _get_perf_store()
weights = _get_weights()
_ensure_history()
_ensure_suggestions()

left, mid, right = st.columns([1.3, 1, 0.8])

with left:
    st.header("Current Snapshot")
    st.subheader("Current Weights")
    w_df = pd.DataFrame([weights], index=["weight"]).T
    w_df = w_df.reset_index().rename(columns={"index":"module", 0:"weight"})
    st.dataframe(w_df.set_index("module"), use_container_width=True)
    st.markdown("---")
    st.subheader("Live module scores (session)")
    if not perf:
        st.info("No module scores in session yet. Open pages (Trend, Fibonacci, Smart Money, Bank Impact) to populate.")
    else:
        for k, v in perf.items():
            sc = v.get("final_score")
            bias = v.get("bias", v.get("trend","-"))
            used = v.get("used", "-")
            st.metric(k.title(), f"{sc if sc is not None else '-'}", delta=None)
            st.caption(f"Bias: {bias} • Source: {used}")

with mid:
    st.header("Module History & Analysis")
    st.markdown("**Upload CSV (optional)** — CSV format: module,ts,score  (ts = ISO string or date).")
    uploaded = st.file_uploader("Upload module history CSV (optional)", type=["csv"])
    if uploaded:
        try:
            csv = pd.read_csv(uploaded)
            # validate required columns
            if set(["module","ts","score"]).issubset(csv.columns):
                # append rows
                added = 0
                for _, r in csv.iterrows():
                    mod = str(r["module"]).strip().lower()
                    if mod not in MODULES:
                        continue
                    ts = pd.to_datetime(r["ts"]).isoformat()
                    try:
                        score = float(r["score"])
                    except Exception:
                        continue
                    add_history_point(mod, ts, score)
                    added += 1
                st.success(f"Added {added} rows to history.")
            else:
                st.error("CSV must contain columns: module, ts, score")
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))

    st.markdown("### Recent trends")
    window_recent = st.slider("Recent window (points)", 5, 50, 14, 1)
    averages = compute_perf_averages(window_recent=window_recent, window_prior=window_recent)
    # compact table
    rows = []
    for m in MODULES:
        a = averages.get(m, {})
        rows.append({
            "module": m,
            "recent_avg": round(a.get("recent", np.nan),2) if a.get("recent") is not None else np.nan,
            "prior_avg": round(a.get("prior", np.nan),2) if a.get("prior") is not None else np.nan,
            "n_recent": a.get("n_recent",0),
            "n_prior": a.get("n_prior",0)
        })
    avg_df = pd.DataFrame(rows).set_index("module")
    st.dataframe(avg_df, use_container_width=True)
    st.markdown("---")
    st.markdown("### Charts (click module)")
    sel = st.selectbox("Module", MODULES, index=0)
    df_plot = history_df(sel)
    if df_plot.empty:
        st.info("No history for selected module. Upload CSV or use the platform to collect scores.")
    else:
        st.line_chart(df_plot["score"])

with right:
    st.header("Suggestion Engine")
    st.markdown("Detects module drift and proposes weight adjustments.")
    threshold = st.slider("Detect threshold (% change)", 5, 50, 10, 1)
    # compute averages again but with threshold used below
    perf_avgs = compute_perf_averages(window_recent=window_recent, window_prior=window_recent)
    st.write("Detected changes (recent vs prior):")
    detect_rows = []
    for m, info in perf_avgs.items():
        recent = info.get("recent")
        prior = info.get("prior")
        if recent is None or prior is None:
            delta_pct = None
        else:
            delta_pct = round((recent - prior) / (prior + 1e-9) * 100.0, 2)
        detect_rows.append({"module":m, "recent": recent, "prior": prior, "delta_%": delta_pct})
    detect_df = pd.DataFrame(detect_rows).set_index("module")
    st.dataframe(detect_df, use_container_width=True)

    if st.button("Generate suggested weights"):
        suggested = suggest_weights(weights, perf_avgs)
        st.session_state["ai_last_suggestion"] = {"time": datetime.utcnow().isoformat(), "suggested": suggested}
        st.success("Suggested weights generated — review below.")
    last = st.session_state.get("ai_last_suggestion")
    if last:
        st.markdown("**Last suggestion**")
        st.write(last["time"])
        sug = last["suggested"]
        sug_df = pd.DataFrame([sug]).T.rename(columns={0:"weight"})
        st.dataframe(sug_df, use_container_width=True)
        st.markdown("Apply suggestion?")
        cols_apply = st.columns([1,1,1])
        with cols_apply[0]:
            if st.button("Apply to session weights"):
                # store override
                st.session_state["weights_override"] = sug
                _ensure_suggestions().append({
                    "time": datetime.utcnow().isoformat(),
                    "type": "apply_session",
                    "suggested": sug,
                    "source": "AI Console"
                })
                st.success("Weights applied in session. Go to Dashboard to see effect.")
        with cols_apply[1]:
            if st.button("Log suggestion only"):
                _ensure_suggestions().append({
                    "time": datetime.utcnow().isoformat(),
                    "type": "log_only",
                    "suggested": sug,
                    "source": "AI Console"
                })
                st.info("Suggestion logged.")
        with cols_apply[2]:
            if st.button("Discard suggestion"):
                st.session_state.pop("ai_last_suggestion", None)
                st.info("Suggestion removed.")

st.markdown("---")
st.header("AI Console Logs & History")
sugs = _ensure_suggestions()
if not sugs:
    st.info("No AI suggestions logged yet.")
else:
    s_df = pd.DataFrame(sugs)
    st.dataframe(s_df.sort_values("time", ascending=False).head(50), use_container_width=True)

st.markdown("---")
st.write("💾 Export / Import history")
col_e1, col_e2 = st.columns(2)
with col_e1:
    if st.button("Export module history (CSV)"):
        # build CSV from session history
        out = []
        hist = _ensure_history()
        for m, rows in hist.items():
            for ts, sc in rows:
                out.append({"module":m, "ts":ts, "score":sc})
        dfout = pd.DataFrame(out)
        b = dfout.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=b, file_name="module_history.csv", mime="text/csv")
with col_e2:
    if st.button("Clear in-memory history"):
        st.session_state["module_history"] = {m: [] for m in MODULES}
        st.success("Cleared module history in session memory.")

st.caption("Note: This console operates in session-memory. For persistent storage use Supabase integration (future).")
