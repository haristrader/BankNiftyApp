# data_engine.py
from __future__ import annotations
import os
import time
import math
import datetime as dt
from typing import Tuple, Dict, Optional

import pandas as pd
import numpy as np
import yfinance as yf

# ---------- Config ----------
# Index symbol normalizer (Yahoo finance tickers)
INDEX_MAP = {
    "BANKNIFTY": "^NSEBANK",
    "NSEBANK.NS": "^NSEBANK",
    "NIFTYBANK.NS": "^NSEBANK",
    "NIFTY BANK": "^NSEBANK",
    "BANKNIFTY.NS": "^NSEBANK",
}

BANK_UNIVERSE = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"
]

# yfinance interval constraints (doc-based)
# (this helps us auto-downgrade period if the combo is invalid)
YF_MAX_BY_INTERVAL = {
    "1m":  "7d",
    "2m":  "60d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",   # ~2y
    "90m": "730d",
    "1h":  "730d",   # alias to 60m, yfinance accepts '60m'
    "1d":  "max",
    "5d":  "max",
    "1wk": "max",
    "1mo": "max",
    "3mo": "max",
}

# -------- Helpers --------
def _normalize_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if s in INDEX_MAP:
        return INDEX_MAP[s]
    # if someone passed NSEBANK.NS, map to ^NSEBANK
    if s.endswith(".NS") and s.startswith("NSEBANK"):
        return "^NSEBANK"
    return s

def _fix_interval(interval: str) -> str:
    # unify friendly strings
    m = interval.lower().strip()
    if m == "1h":
        return "60m"
    return m

def _safe_period_for_interval(interval: str, period: str) -> str:
    """If user asks invalid combo, auto-downgrade period safely."""
    interval = _fix_interval(interval)
    maxp = YF_MAX_BY_INTERVAL.get(interval, "60d")
    if maxp == "max":
        return period
    # convert like '5d','60d','1y' to comparable days
    def to_days(p: str) -> int:
        p = p.lower()
        if p.endswith("d"): return int(p[:-1])
        if p.endswith("mo"): return int(p[:-2]) * 30
        if p.endswith("y"): return int(p[:-1]) * 365
        if p in ("max",): return 10_000
        return 7  # default
    if to_days(period) > to_days(maxp):
        return maxp
    return period

# -------- Public API --------
def get_candles(
    symbol: str,
    period: str = "5d",
    interval: str = "5m",
    auto_adjust: bool = True,
    prepost: bool = False,
    tz_localize: bool = True
) -> pd.DataFrame:
    """
    Unified market data fetcher using yfinance with safe fallbacks.
    Returns columns: ['open','high','low','close','volume'] (lowercase)
    Index = DatetimeIndex (tz-aware if available).
    """
    yf_symbol = _normalize_symbol(symbol)
    interval = _fix_interval(interval)
    period = _safe_period_for_interval(interval, period)

    try:
        df = yf.download(
            tickers=yf_symbol,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            prepost=prepost,
            progress=False,
            threads=True
        )
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    # Standardize columns
    cols = {c.lower():c for c in df.columns.str.lower()}
    for need in ["open","high","low","close","volume"]:
        if need not in cols:
            # Try with Yahoo naming
            if need.capitalize() in df.columns:
                cols[need] = need.capitalize()
    rename_map = {cols.get(k, k): k for k in ["open","high","low","close","volume"] if cols.get(k)}
    sdf = df.rename(columns={v:k for k,v in rename_map.items()})
    # Keep only these
    sdf = sdf[["open","high","low","close","volume"]].copy()

    # ensure datetime index
    if not isinstance(sdf.index, pd.DatetimeIndex):
        sdf.index = pd.to_datetime(sdf.index)

    if tz_localize:
        try:
            # yfinance returns tz-aware sometimes; normalize to naive local
            sdf.index = sdf.index.tz_localize(None)
        except Exception:
            pass

    return sdf.dropna(how="any")

# ---- Indicators ----
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=close.index).ewm(alpha=1/length, adjust=False).mean()
    roll_dn = pd.Series(loss, index=close.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_dn.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def ema_rsi_block(df: pd.DataFrame, ema_fast=20, ema_slow=50, rsi_len=14) -> pd.DataFrame:
    out = df.copy()
    out[f"ema{ema_fast}"] = ema(out["close"], ema_fast)
    out[f"ema{ema_slow}"] = ema(out["close"], ema_slow)
    out[f"rsi{rsi_len}"] = rsi(out["close"], rsi_len)
    return out

# ---- Fibonacci Swing (simple) ----
def last_swing(df: pd.DataFrame, lookback: int = 200) -> Tuple[float, float]:
    """
    Find swing high/low in the last `lookback` candles (not necessarily extremes of whole dataset).
    """
    data = df.tail(lookback)
    s_low = float(data["low"].min())
    s_high = float(data["high"].max())
    return s_low, s_high

def fib_levels(swing_low: float, swing_high: float) -> Dict[str, float]:
    diff = swing_high - swing_low
    return {
        "0%": swing_high,
        "23.6%": swing_high - 0.236*diff,
        "38.2%": swing_high - 0.382*diff,
        "50%": swing_high - 0.5*diff,
        "60%": swing_high - 0.6*diff,   # your special band
        "61.8%": swing_high - 0.618*diff,
        "78.6%": swing_high - 0.786*diff,
        "100%": swing_low
    }

# ---- Mid-zone signal (50–60%) ----
def midzone_signals(df: pd.DataFrame, swing_lookback=200) -> pd.DataFrame:
    """
    Mark when close enters 50–60% zone between latest swing high/low.
    """
    out = df.copy()
    s_low, s_high = last_swing(out, lookback=swing_lookback)
    fib = fib_levels(s_low, s_high)
    lo = min(fib["50%"], fib["60%"])
    hi = max(fib["50%"], fib["60%"])
    out["midzone"] = (out["close"].between(lo, hi)).astype(int)
    out["midzone_low"] = lo
    out["midzone_high"] = hi
    return out

# ---- Smart Money (simple “trap/absorption” heuristic) ----
def smart_money_flags(df: pd.DataFrame, wick_ratio=0.6, vol_mult=1.5) -> pd.DataFrame:
    """
    Very light heuristic:
      - large wick relative to candle size -> possible trap/absorption
      - volume spike vs rolling mean
    """
    out = df.copy()
    body = (out["close"] - out["open"]).abs()
    upper_wick = out["high"] - out[["close","open"]].max(axis=1)
    lower_wick = out[["close","open"]].min(axis=1) - out["low"]
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["upper_wick_ratio"] = (upper_wick / rng).fillna(0)
    out["lower_wick_ratio"] = (lower_wick / rng).fillna(0)
    out["vol_ma"] = out["volume"].rolling(50).mean()
    # trap if wick big and volume spike
    out["trap_up"] = ((out["upper_wick_ratio"] > wick_ratio) & (out["volume"] > vol_mult*out["vol_ma"])).astype(int)
    out["trap_dn"] = ((out["lower_wick_ratio"] > wick_ratio) & (out["volume"] > vol_mult*out["vol_ma"])).astype(int)
    return out

# ---- Bank Impact helper ----
def bank_universe() -> list:
    return BANK_UNIVERSE[:]
