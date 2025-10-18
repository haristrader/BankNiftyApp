# utils.py  — FINAL STABLE VERSION (No pd.notna, No .str issues)
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

# Default Index
DEFAULT_SYMBOL = "^NSEBANK"

# Weights (Dashboard Me Use)
WEIGHTS_DEFAULT = {
    "trend": 20,
    "fibonacci": 25,
    "priceaction": 15,
    "smartmoney": 20,
    "backtest": 10,
    "others": 10,
}

# ------------ FETCH DATA ------------
def fetch(symbol: str = DEFAULT_SYMBOL, period: str = "14d", interval: str = "5m", auto_adjust: bool = True) -> pd.DataFrame:
    """
    Download OHLCV via yfinance and return a cleaned DataFrame with columns:
    ['open','high','low','close','volume'] and DateTimeIndex (tz-naive).
    Robust to MultiIndex/duplicate columns.
    """
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=auto_adjust,
            group_by="column",     # <- important: avoid MultiIndex
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # If MultiIndex slipped through, flatten it
    if isinstance(df.columns, pd.MultiIndex):
        # use the last level name (e.g., ('^NSEBANK','Open') -> 'Open')
        df.columns = [str(col[-1]) if isinstance(col, tuple) else str(col) for col in df.columns]

    # Standardize names and keep only needed
    df = df.rename(columns={
        "Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
    })

    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    if not keep:
        return pd.DataFrame()
    df = df[keep].copy()

    # tz-naive, sorted
    try:
        df.index = pd.to_datetime(df.index).tz_convert(None)
    except Exception:
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return sanitize_ohlcv(df)


# ------------ SANITIZE DATA ------------
def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe OHLCV sanitizer:
    - Drops duplicate columns
    - If MultiIndex columns exist, flattens
    - Ensures each OHLCV is a 1-D Series before numeric coercion
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # If MultiIndex columns exist (paranoia check), flatten again
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[-1]) if isinstance(col, tuple) else str(col) for col in out.columns]

    # Drop duplicated column names, keep first occurrence
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]

    # Coerce each target column to a 1-D Series, then to numeric
    cleaned = pd.DataFrame(index=out.index)
    for col in ["open","high","low","close","volume"]:
        if col not in out.columns:
            continue
        col_obj = out[col]
        # If due to oddities this is a DataFrame (multiple same-named cols), take first
        if isinstance(col_obj, pd.DataFrame):
            col_obj = col_obj.iloc[:, 0]
        # Now ensure numeric
        cleaned[col] = pd.to_numeric(col_obj, errors="coerce")

    # Drop rows with missing required values
    cleaned = cleaned.dropna(subset=["open","high","low","close","volume"])

    # Ensure DateTimeIndex name
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
        cleaned = cleaned.dropna().sort_index()

    cleaned.index.name = "Date"
    return cleaned


# ------------ FIBONACCI HELPERS ------------
def swing(df: pd.DataFrame, lookback: int = 200):
    recent = df.tail(max(lookback, 2))
    sh = float(recent["high"].max())
    sl = float(recent["low"].min())
    last_close = float(df["close"].iloc[-1])
    mid = (sh + sl) / 2
    direction = "down" if last_close > mid else "up"
    return sh, sl, direction


def fib_levels(high: float, low: float, direction: str = "up") -> dict:
    d = high - low if high != low else 1e-9
    if direction == "up":
        return {
            "0%": high, "23.6%": high - 0.236*d, "38.2%": high - 0.382*d,
            "50%": high - 0.5*d, "61.8%": high - 0.618*d, "78.6%": high - 0.786*d, "100%": low
        }
    return {
        "0%": low, "23.6%": low + 0.236*d, "38.2%": low + 0.382*d,
        "50%": low + 0.5*d, "61.8%": low + 0.618*d, "78.6%": low + 0.786*d, "100%": high
    }


def fib_confidence(df: pd.DataFrame, lookback: int = 200) -> tuple[float, dict]:
    sh, sl, dirn = swing(df, lookback)
    lv = fib_levels(sh, sl, "up" if dirn == "up" else "down")
    close = float(df["close"].iloc[-1])
    targets = [lv["38.2%"], lv["50%"], lv["61.8%"]]
    dist = min(abs(close - t) for t in targets)
    rng = max(1.0, (sh - sl) / 6.0)
    score = max(0.0, 100.0 - (dist / rng) * 100.0)
    return round(score, 2), lv
