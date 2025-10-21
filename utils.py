# utils.py — unified, production-safe version (BankNifty Algo System)

from __future__ import annotations
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict

# --- Imports from our engine ---
from data_engine import get_candles, ema_rsi_block, midzone_signals, smart_money_flags

# Optional clients (future use)
try:
    from tv_client import get_ohlcv as tv_get_ohlcv
except Exception:
    tv_get_ohlcv = None

try:
    from supabase_client import supabase
except Exception:
    supabase = None


# =========================
# ---- CONSTANTS ----------
# =========================

DEFAULT_SYMBOL = "NSEBANK.NS"

INTERVAL_LIMITS = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d",
    "60m": "730d", "90m": "60d", "1h": "730d",
    "1d": "max", "5d": "max", "1wk": "max", "1mo": "max"
}


# =========================
# ---- TIME HELPERS -------
# =========================

def now_ist() -> datetime:
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def is_weekend(dt: datetime | None = None) -> bool:
    d = dt or now_ist()
    return d.weekday() >= 5


def is_market_hours(dt: datetime | None = None) -> bool:
    """NSE market hours check (9:15–15:30 IST)."""
    d = dt or now_ist()
    if is_weekend(d):
        return False
    hm = d.hour * 60 + d.minute
    return (9*60 + 15) <= hm <= (15*60 + 30)


def weekend_safe_period(prefer: tuple[str, str] | None = ("5d", "5m")) -> tuple[str, str]:
    """Return (period, interval) adjusted for weekends."""
    if is_market_hours():
        return prefer or ("5d", "5m")
    return ("3mo", "1d")


# =========================
# ---- FETCHING ----------
# =========================

def fetch_smart(
    symbol: str = DEFAULT_SYMBOL,
    prefer: tuple[str, str] = ("5d", "5m"),
    mode: str = "auto",  # 'auto' | 'daily' | 'live'
) -> Tuple[pd.DataFrame, str]:
    """
    Unified data loader using yfinance (primary) + smart weekend fallback.
    """
    period, interval = weekend_safe_period(prefer) if mode == "auto" else prefer
    df = get_candles(symbol, period=period, interval=interval)

    msg = f"Data fetched ({period}/{interval})"
    if df.empty and interval != "1d":
        # fallback
        df = get_candles(symbol, period="6mo", interval="1d")
        msg = "No intraday data — fallback to daily."
    if df.empty:
        msg = "No data available even after fallback."

    return df, msg


# =========================
# ---- INDICATOR HELPERS --
# =========================

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def zscore(series: pd.Series, n: int = 20) -> pd.Series:
    m = series.rolling(n).mean()
    s = series.rolling(n).std(ddof=0)
    return (series - m) / (s.replace(0, np.nan))


# =========================
# ---- FIBONACCI HELPERS --
# =========================

def swing(df: pd.DataFrame, lookback: int = 200) -> tuple[float, float, str]:
    """Identify swing high/low within a lookback period."""
    if df.empty:
        return (np.nan, np.nan, "flat")
    recent = df.tail(max(lookback, 2))
    sh = float(recent["high"].max())
    sl = float(recent["low"].min())
    last_close = float(df["close"].iloc[-1])
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
    """Measure how close price is to major Fibonacci levels."""
    if df.empty:
        return 0.0, {}
    sh, sl, direction = swing(df, lookback)
    lv = fib_levels(sh, sl, direction)
    close = float(df["close"].iloc[-1])
    targets = [lv["38.2%"], lv["50%"], lv["61.8%"]]
    dist = min(abs(close - t) for t in targets)
    rng = max(1.0, (sh - sl) / 6.0)
    score = max(0.0, 100.0 - (dist / rng) * 100.0)
    return round(score, 2), lv


# =========================
# ---- TREND HELPERS ------
# =========================

def trend_score(df: pd.DataFrame) -> tuple[float, str]:
    """Calculate trend strength score (EMA + RSI)."""
    if df.empty or len(df) < 50:
        return 0.0, "No data"
    c = df["close"]
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    r = rsi(c, 14).iloc[-1]

    pts = 0
    if e20.iloc[-1] > e50.iloc[-1]: pts += 25
    if e50.iloc[-1] > e200.iloc[-1]: pts += 25
    if c.iloc[-1] > e20.iloc[-1]: pts += 25
    if r > 55: pts += 15
    score = float(pts)

    bias = "Bullish" if score >= 60 else ("Bearish" if score <= 40 else "Neutral")
    return score, bias


# =========================
# ---- MIDZONE 50–60% -----
# =========================

def midzone_flag(df: pd.DataFrame, lookback: int = 200) -> pd.DataFrame:
    """Add column 'midzone' where close is between 50–60% Fibonacci."""
    sh, sl, direction = swing(df, lookback)
    fib = fib_levels(sh, sl, direction)
    lo = min(fib["50%"], fib["61.8%"])
    hi = max(fib["50%"], fib["61.8%"])
    out = df.copy()
    out["midzone"] = out["close"].between(lo, hi).astype(int)
    return out


# =========================
# ---- SMART MONEY --------
# =========================

def smart_money(df: pd.DataFrame) -> pd.DataFrame:
    """Detect possible traps (volume spikes + wick)."""
    return smart_money_flags(df)


# =========================
# ---- WRAPPERS -----------
# =========================

def with_ema_rsi(df: pd.DataFrame) -> pd.DataFrame:
    return ema_rsi_block(df)


def with_midzone(df: pd.DataFrame) -> pd.DataFrame:
    return midzone_signals(df)


def with_smart_money(df: pd.DataFrame) -> pd.DataFrame:
    return smart_money_flags(df)
