# utils.py  —  CLEAN, PRODUCTION-SAFE
from __future__ import annotations

import os
import io
import math
import time
import json
import typing as T
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from data_engine import fetch_smart
# --- Optional dependencies (present in repo) ---
# tv_client: our TradingView lightweight client (get_ohlcv)
# supabase_client: preconfigured Supabase client + helpers
try:
    from tv_client import get_ohlcv as tv_get_ohlcv
except Exception:
    tv_get_ohlcv = None

try:
    from supabase_client import supabase  # expects SUPABASE_URL / SUPABASE_KEY
except Exception:
    supabase = None

# yfinance as secondary live source
try:
    import yfinance as yf
except Exception:
    yf = None


# =========================
# ---- CONSTANTS/CONFIG ---
# =========================

# Default (index)
DEFAULT_SYMBOL = "NSEBANK.NS"    # your app mostly uses this

# Max history yfinance allows per interval (coarse)
INTERVAL_LIMITS = {
    "1m": "7d",   "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d",
    "60m": "730d","90m": "60d","1h": "730d",   # 1h alias
    "1d": "max", "5d": "max", "1wk": "max", "1mo": "max"
}

# TradingView mapping (TV uses different exchange codes)
def map_tv_symbol(symbol: str) -> str:
    s = symbol.upper().strip()
    # Common mappings
    if s in {"^NSEBANK", "NSEBANK", "NSEBANK.NS", "BANKNIFTY", "^NIFTYBANK"}:
        return "NSE:NIFTYBANK"
    if s in {"^NSEI", "NSEI", "NIFTY", "NIFTY50", "NIFTY.NS"}:
        return "NSE:NIFTY"
    # Equities likely okay as NSE:<ticker-without-.NS>
    if s.endswith(".NS"):
        return f"NSE:{s[:-3]}"
    return s  # fallback


# =========================
# ----- TIME HELPERS ------
# =========================

def now_ist() -> datetime:
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))

def is_weekend(dt: datetime | None = None) -> bool:
    d = dt or now_ist()
    return d.weekday() >= 5  # Sat(5), Sun(6)

def is_market_hours(dt: datetime | None = None) -> bool:
    """Rough NSE cash-market hours check (09:15–15:30 IST)."""
    d = dt or now_ist()
    if is_weekend(d):
        return False
    hm = d.hour * 60 + d.minute
    return (9*60 + 15) <= hm <= (15*60 + 30)

def weekend_safe_period(prefer: tuple[str, str] | None) -> tuple[str, str]:
    """If weekend/off-hours, force safe daily."""
    if is_market_hours():
        return prefer or ("5d", "5m")
    # Safe daily during closed hours
    return ("3mo", "1d")


# =========================
# ----- SANITIZERS --------
# =========================

def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce OHLCV into numeric, DateTimeIndex tz-naive, columns: open high low close volume."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    out = df.copy()

    # Try to standardize columns
    ren = {
        "Open":"open","High":"high","Low":"low","Close":"close",
        "Adj Close":"adj_close","Volume":"volume",
    }
    out.columns = [c.lower() for c in out.columns]
    out = out.rename(columns=ren)

    # If yfinance style multi-index columns, flatten
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join([str(x) for x in col if x]) for col in out.columns]

    # Fill missing main cols if possible
    for alt, main in [("adj_close","close")]:
        if main not in out.columns and alt in out.columns:
            out[main] = out[alt]

    # Strong select
    keep = [c for c in ["open","high","low","close","volume"] if c in out.columns]
    out = out[keep]

    # Ensure datetime index
    if not isinstance(out.index, pd.DatetimeIndex):
        # try common columns
        for guess in ["datetime","date","time","timestamp"]:
            if guess in df.columns:
                out.index = pd.to_datetime(df[guess], errors="coerce")
                break
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")

    # Drop bad
    out = out[~out.index.to_series().isna()]
    try:
        out.index = out.index.tz_convert(None)
    except Exception:
        try:
            out.index = out.index.tz_localize(None)
        except Exception:
            pass

    # Numeric coercion
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna()
    out = out.sort_index()
    return out


# =========================
# ----- FETCHERS ----------
# =========================

def fetch_tv(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """TradingView primary fetch (fast)."""
    if tv_get_ohlcv is None:
        return pd.DataFrame()
    try:
        tv_symbol = map_tv_symbol(symbol)
        df = tv_get_ohlcv(tv_symbol, period=period, interval=interval)
        return sanitize_ohlcv(df)
    except Exception:
        return pd.DataFrame()

def fetch(symbol: str, period: str, interval: str, auto_adjust: bool = True, mode: str = "auto"):
    """
    Wrapper used across pages.
    period: "5d"/"1mo"/"3mo"/"1y"/"max" (your UI options)
    interval: "5m"/"15m"/"1h"/"1d" (your UI options)
    mode: "auto"|"daily-only"|"intraday"
    """
    # route to smart engine
    df, meta = fetch_smart(symbol, prefer_period=period, prefer_interval=interval, mode=mode)

    # sanitize (downstream pages expect numeric, no nans at ends)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["close"])
    return out
    
def min_period(requested: str, limit: str) -> str:
    """Return min(requested, limit) in 'Xd'/'Xmo'/'Yd' style — very coarse compare."""
    if limit == "max":
        return requested
    # convert to days approx for compare
    def to_days(p: str) -> int:
        s = p.lower()
        if s.endswith("d"):
            return int(s[:-1])
        if s.endswith("wk"):
            return int(s[:-2]) * 7
        if s.endswith("mo"):
            return int(s[:-2]) * 30
        if s.endswith("y"):
            return int(s[:-1]) * 365
        return 999999
    return requested if to_days(requested) <= to_days(limit) else limit


# ============== Supabase cache ==============

def supabase_table() -> T.Optional[str]:
    return "daily_cache"

def fetch_supabase_daily(symbol: str) -> pd.DataFrame:
    """Try load last 6 months daily from Supabase cache."""
    if supabase is None:
        return pd.DataFrame()
    try:
        table = supabase_table()
        resp = supabase.table(table).select("*").eq("symbol", symbol).order("date", desc=False).execute()
        rows = resp.data or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.set_index("date")
        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
        return sanitize_ohlcv(df)
    except Exception:
        return pd.DataFrame()

def save_supabase_daily(symbol: str, df: pd.DataFrame) -> None:
    """Upsert compact daily to Supabase."""
    if supabase is None or df is None or df.empty:
        return
    try:
        table = supabase_table()
        out = []
        daily = df.copy()
        if "1d" not in str(daily.index.freq or ""):
            # resample to daily if not daily
            daily = (
                daily[["open","high","low","close","volume"]]
                .resample("1D")
                .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
            ).dropna()
        for dt, row in daily.tail(180).iterrows():
            out.append({
                "symbol": symbol,
                "date": dt.date().isoformat(),
                "o": float(row["open"]),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
                "v": float(row["volume"]),
            })
        if not out:
            return
        supabase.table(table).upsert(out, on_conflict="symbol,date").execute()
    except Exception:
        pass


# =========================
# ----- PUBLIC API --------
# =========================

def fetch(
    symbol: str = DEFAULT_SYMBOL,
    period: str = "14d",
    interval: str = "5m",
    auto_adjust: bool = True
) -> pd.DataFrame:
    """Simple direct yfinance fetch (legacy)."""
    return fetch_yf(symbol, period, interval, auto_adjust=auto_adjust)

def fetch_smart(
    symbol: str = DEFAULT_SYMBOL,
    prefer: tuple[str, str] | None = ("5d", "5m"),
    mode: str = "auto",  # 'auto' | 'live' | 'daily'
    auto_adjust: bool = True,
) -> tuple[pd.DataFrame, tuple[str, str], str]:
    """
    FINAL unified loader:
      Priority: TradingView -> Yahoo -> Supabase (daily cache)
      Safe weekend/daily fallback
      Returns: df, (period, interval), msg
    """
    # Decide effective (period, interval)
    if mode == "daily":
        eff_p, eff_i = ("3mo", "1d")
    elif mode == "live":
        eff_p, eff_i = prefer or ("5d","5m")
    else:
        eff_p, eff_i = weekend_safe_period(prefer)

    # Attempt 1: TradingView (intraday or daily)
    msg_parts = []
    df = pd.DataFrame()
    used = (eff_p, eff_i)

    if eff_i not in {"1d","1wk","1mo"}:  # only try TV for intra first
        dftv = fetch_tv(symbol, eff_p, eff_i)
        if len(dftv) > 0:
            msg_parts.append(f"TV✓ {eff_p}/{eff_i}")
            df = dftv
        else:
            msg_parts.append(f"TV×")
    else:
        # daily on TV (can still try)
        dftv = fetch_tv(symbol, eff_p, eff_i)
        if len(dftv) > 0:
            msg_parts.append(f"TV✓ {eff_p}/{eff_i}")
            df = dftv
        else:
            msg_parts.append("TV×")

    # Attempt 2: Yahoo
    if df.empty:
        dfyf = fetch_yf(symbol, eff_p, eff_i, auto_adjust=auto_adjust)
        if len(dfyf) > 0:
            msg_parts.append(f"YF✓ {eff_p}/{eff_i}")
            df = dfyf
        else:
            msg_parts.append("YF×")

    # Attempt 3: Supabase DAILY if still empty
    if df.empty:
        dfd = fetch_supabase_daily(symbol)
        if len(dfd) > 0:
            msg_parts.append("SB✓ daily")
            df = dfd
            used = ("3mo", "1d")
        else:
            msg_parts.append("SB×")

    # As a courtesy, cache daily if we loaded something recent
    try:
        if not df.empty and eff_i in {"1d","1wk","1mo"}:
            save_supabase_daily(symbol, df)
    except Exception:
        pass

    # Final guard
    if df is None or df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"]), used, "No data available even after fallbacks."

    return df, used, " | ".join(msg_parts)


# =========================
# ---- INDICATOR HELPERS ---
# =========================

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50)

def zscore(series: pd.Series, n: int = 20) -> pd.Series:
    m = series.rolling(n).mean()
    s = series.rolling(n).std(ddof=0)
    return (series - m) / (s.replace(0, np.nan))


# =========================
# --- FIBONACCI HELPERS ---
# =========================

def swing(df: pd.DataFrame, lookback: int = 200) -> tuple[float, float, str]:
    """Flexible swing hi/low inside window (not strictly first/last)."""
    if df is None or df.empty:
        return (np.nan, np.nan, "flat")
    recent = df.tail(max(lookback, 2))
    sh = float(recent["high"].max()); sl = float(recent["low"].min())
    last_close = float(recent["close"].iloc[-1])
    mid = (sh + sl) / 2.0
    direction = "down" if last_close > mid else "up"
    return sh, sl, direction

def fib_levels(high: float, low: float, direction: str = "up") -> dict:
    d = max(1e-9, high - low)
    if direction == "up":
        lv = {
            "0%": high, "23.6%": high - 0.236*d, "38.2%": high - 0.382*d,
            "50%": high - 0.5*d, "61.8%": high - 0.618*d, "78.6%": high - 0.786*d, "100%": low
        }
    else:
        lv = {
            "0%": low, "23.6%": low + 0.236*d, "38.2%": low + 0.382*d,
            "50%": low + 0.5*d, "61.8%": low + 0.618*d, "78.6%": low + 0.786*d, "100%": high
        }
    return {k: float(v) for k, v in lv.items()}

def fib_confidence(df: pd.DataFrame, lookback: int = 200) -> tuple[float, dict]:
    """Score closeness to 38.2/50/61.8 (higher=closer)."""
    if df is None or df.empty:
        return 0.0, {}
    sh, sl, direction = swing(df, lookback)
    lv = fib_levels(sh, sl, "up" if direction == "up" else "down")
    close = float(df["close"].iloc[-1])
    targets = [lv["38.2%"], lv["50%"], lv["61.8%"]]
    dist = min(abs(close - t) for t in targets)
    rng = max(1.0, (sh - sl) / 6.0)
    score = max(0.0, 100.0 - (dist / rng) * 100.0)
    return round(score, 2), lv


# =========================
# ----- TREND HELPERS -----
# =========================

def trend_score(df: pd.DataFrame) -> tuple[float, str]:
    """EMA stack + RSI roll into a 0–100 score + textual bias."""
    if df is None or df.empty or len(df) < 50:
        return 0.0, "No data"
    c = df["close"]
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    r = rsi(c, 14).iloc[-1]

    pts = 0
    if e20.iloc[-1] > e50.iloc[-1]: pts += 20
    if e50.iloc[-1] > e200.iloc[-1]: pts += 30
    if c.iloc[-1] > e20.iloc[-1]: pts += 20
    if r > 55: pts += 15
    if r > 60: pts += 15
    score = float(pts)

    bias = "Bullish" if score >= 60 else ("Bearish" if score <= 40 else "Neutral")
    return score, bias


# =========================
# ----- MISC HELPERS ------
# =========================

def last_daily_close(symbol: str = DEFAULT_SYMBOL) -> tuple[float | None, str]:
    """Get last daily close using smart daily loader; returns (close, msg)."""
    df, used, msg = fetch_smart(symbol, prefer=("3mo","1d"), mode="daily")
    if df.empty:
        return None, msg
    return float(df["close"].iloc[-1]), msg
