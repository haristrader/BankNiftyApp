# utils.py — NSE-first data engine (TradingView-like headers), cloud-safe
from __future__ import annotations

import time
import math
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf  # kept as backup only

# ------------------ Defaults & Weights ------------------
# Use NSE-friendly default everywhere
DEFAULT_SYMBOL = "NSEBANK.NS"

WEIGHTS_DEFAULT = {
    "trend": 20,
    "fibonacci": 25,
    "priceaction": 15,
    "smartmoney": 20,
    "backtest": 10,
    "others": 10,
}

# intraday/daily combos we may attempt in yfinance BACKUP only
INTERVAL_FALLBACKS: list[tuple[str, str]] = [
    ("5d", "5m"),
    ("14d", "15m"),
    ("30d", "30m"),
    ("60d", "60m"),
    ("3mo", "1d"),
    ("2y", "1wk"),
]

# ------------- TradingView-like headers for NSE -------------
_TV_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 TradingView/1.0"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}

# ---------- Symbol mapping (our inputs → NSE index name) ----------
def _to_nse_index_name(symbol: str) -> str:
    """Map variations to NSE index string used by the chart API."""
    s = (symbol or "").upper().strip()
    if s in {"^NSEBANK", "NSEBANK", "NSEBANK.NS", "BANKNIFTY", "BANKNIFTY.NS"}:
        return "NIFTY BANK"
    return "NIFTY BANK"  # we keep app BankNifty-only by design

# ---------- Interval mapping for NSE API ----------
def _interval_to_nse(interval: str) -> str:
    m = {
        "5m": "5minute",
        "15m": "15minute",
        "60m": "60minute",
        "1h": "60minute",
        "1d": "1day",
        "1D": "1day",
        "daily": "1day",
    }
    return m.get(interval, "5minute")

def _period_days(period: str) -> int:
    """Approximate number of days for simple from/to windows."""
    m = {
        "5d": 5, "7d": 7, "10d": 10, "14d": 14, "30d": 30, "60d": 60,
        "3mo": 90, "6mo": 180, "1y": 365, "2y": 730
    }
    return m.get(period, 14)

# ------------------ NSE fetcher ------------------
def fetch_nse_index(
    symbol: str = DEFAULT_SYMBOL,
    period: str = "14d",
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Pull OHLC from NSE 'chart-databyindex' API.
    Returns DataFrame with ['open','high','low','close','volume'] and DateTimeIndex.
    """
    idx_name = _to_nse_index_name(symbol)
    nse_interval = _interval_to_nse(interval)

    # Build from/to (epoch seconds)
    days = _period_days(period)
    to_ts = int(time.time())
    from_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    # NSE endpoints commonly used for charts:
    # Newer index chart endpoint includes 'candles'
    # /api/chart-databyindex?index=<IDX>&indices=true&from=<epoch>&to=<epoch>
    base = "https://www.nseindia.com"
    chart_url = f"{base}/api/chart-databyindex?index={requests.utils.quote(idx_name)}&indices=true&from={from_ts}&to={to_ts}"

    # Some deployments require a specific interval endpoint:
    # /api/candles/index?index=<IDX>&from=<epoch>&to=<epoch>&interval=<iv>
    candles_url = f"{base}/api/candles/index?index={requests.utils.quote(idx_name)}&from={from_ts}&to={to_ts}&interval={nse_interval}"

    sess = requests.Session()
    try:
        # Boot cookies
        sess.get(base, headers=_TV_HEADERS, timeout=10)
        # Try candles endpoint first (rich OHLC)
        r = sess.get(candles_url, headers=_TV_HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            # Expected: {"candles":[["YYYY-MM-DDTHH:MM:SS", o,h,l,c,vol], ...]}
            if isinstance(data, dict) and "candles" in data and isinstance(data["candles"], list) and len(data["candles"]) > 0:
                rows = data["candles"]
                df = pd.DataFrame(rows, columns=["Date", "open", "high", "low", "close", "volume"])
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
                return sanitize_ohlcv(df)

        # Fallback: chart-databyindex (sometimes returns 'candles' too)
        r2 = sess.get(chart_url, headers=_TV_HEADERS, timeout=15)
        if r2.status_code == 200:
            data2 = r2.json()
            # Some variants embed "candles"; older variants embed "grapthData"
            if isinstance(data2, dict):
                if "candles" in data2 and isinstance(data2["candles"], list) and len(data2["candles"]) > 0:
                    rows = data2["candles"]
                    df = pd.DataFrame(rows, columns=["Date", "open", "high", "low", "close", "volume"])
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
                    return sanitize_ohlcv(df)
                elif "grapthData" in data2 and isinstance(data2["grapthData"], list) and len(data2["grapthData"]) > 0:
                    # grapthData is typically [ts_ms, close]; create pseudo-OHLC
                    rows = data2["grapthData"]
                    tmp = pd.DataFrame(rows, columns=["ts", "close"])
                    tmp["Date"] = pd.to_datetime(tmp["ts"], unit="ms", utc=True).dt.tz_convert(None)
                    tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
                    tmp = tmp.dropna(subset=["Date", "close"]).set_index("Date").sort_index()
                    # build OHLC with small rolling window as fallback
                    tmp["open"] = tmp["close"].shift(1).fillna(tmp["close"])
                    tmp["high"] = tmp[["open", "close"]].max(axis=1)
                    tmp["low"] = tmp[["open", "close"]].min(axis=1)
                    tmp["volume"] = 0.0
                    df = tmp[["open", "high", "low", "close", "volume"]]
                    return sanitize_ohlcv(df)

    except Exception:
        pass

    # If NSE paths failed, return empty (caller will try yfinance)
    return pd.DataFrame()

# ------------------ yfinance backup fetcher ------------------
def fetch_yf(symbol: str, period: str, interval: str, auto_adjust: bool = True) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol, period=period, interval=interval,
            progress=False, auto_adjust=auto_adjust, group_by="column"
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[-1]) if isinstance(col, tuple) else str(col) for col in df.columns]

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
    })
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if not keep:
        return pd.DataFrame()
    df = df[keep].copy()

    try:
        df.index = pd.to_datetime(df.index).tz_convert(None)
    except Exception:
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return sanitize_ohlcv(df)

# ------------------ Smart fetch (NSE first → yfinance fallback) ------------------
def fetch(
    symbol: str = DEFAULT_SYMBOL,
    period: str = "14d",
    interval: str = "5m",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    # Try NSE engine
    df = fetch_nse_index(symbol, period, interval)
    if df is not None and not df.empty:
        return df

    # Try yfinance current request
    df = fetch_yf(symbol, period, interval, auto_adjust)
    if df is not None and not df.empty:
        return df

    # Try yfinance fallbacks
    for per, iv in INTERVAL_FALLBACKS:
        df = fetch_yf(symbol, per, iv, auto_adjust)
        if df is not None and not df.empty:
            return df

    return pd.DataFrame()

def fetch_smart(
    symbol: str = DEFAULT_SYMBOL,
    prefer: tuple[str, str] | None = None,
    auto_adjust: bool = True,
    allow_symbol_fallback: bool = False,  # keep False per your preference
) -> tuple[pd.DataFrame, tuple[str, str]]:
    tries: list[tuple[str, str]] = []
    if prefer and isinstance(prefer, tuple) and len(prefer) == 2:
        tries.append(prefer)
    for t in INTERVAL_FALLBACKS:
        if t not in tries:
            tries.append(t)

    # First pass: NSE with requested -> then yfinance
    for per, iv in tries:
        df = fetch_nse_index(symbol, per, iv)
        if df is not None and not df.empty:
            return df, (per, iv)
        df = fetch_yf(symbol, per, iv, auto_adjust)
        if df is not None and not df.empty:
            return df, (per, iv)

    # Optional symbol fallback (rarely needed)
    if allow_symbol_fallback:
        alt = "NSEBANK.NS"
        for per, iv in [("3mo", "1d"), ("2y", "1wk")]:
            df = fetch_nse_index(alt, per, iv)
            if df is not None and not df.empty:
                return df, (per, iv)
            df = fetch_yf(alt, per, iv, auto_adjust)
            if df is not None and not df.empty:
                return df, (per, iv)

    return pd.DataFrame(), (tries[0][0] if tries else "5d", tries[0][1] if tries else "5m")

# ------------------ Sanitizer ------------------
def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[-1]) if isinstance(col, tuple) else str(col) for col in out.columns]
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]
    cleaned = pd.DataFrame(index=out.index)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            col_obj = out[col]
            if isinstance(col_obj, pd.DataFrame):
                col_obj = col_obj.iloc[:, 0]
            cleaned[col] = pd.to_numeric(col_obj, errors="coerce")
    cleaned = cleaned.dropna(subset=["open", "high", "low", "close", "volume"])
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
        cleaned = cleaned.dropna().sort_index()
    cleaned.index.name = "Date"
    return cleaned

# ------------------ Signals (50% rule) ------------------
def generate_signals_50pct(df: pd.DataFrame, mid_factor: float = 0.5) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    sig = ["HOLD"]
    for i in range(1, len(out)):
        ph = float(out["high"].iloc[i-1]); pl = float(out["low"].iloc[i-1])
        prev_mid = pl + (ph - pl) * mid_factor
        c = out.iloc[i]
        if (c["low"] >= prev_mid) and (c["close"] > ph):
            sig.append("BUY")
        elif (c["high"] <= prev_mid) and (c["close"] < pl):
            sig.append("SELL")
        else:
            sig.append("HOLD")
    out["signal"] = sig
    return out

# ------------------ ATM Option Simulator (delta model) ------------------
def _round_to_strike(x: float, step: int = 100) -> int:
    return int(round(x / step) * step)

def simulate_atm_option_trades(
    df: pd.DataFrame,
    signals_col: str = "signal",
    init_sl_pts: float = 10.0,
    lot_size: int = 15,
    mode: str = "delta",            # "delta" or "index"
    delta_atm: float = 0.5,
    theta_per_candle: float = 0.0,
):
    if df is None or df.empty or signals_col not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float), 50.0

    prices = df["close"].astype(float).values
    times = list(df.index)

    pos=None; entry_p=None; sl=None
    base_mult = 0.01  # 1% of index as baseline premium
    trades=[]; prem=0.0

    for i in range(len(df)-1):
        sig = str(df[signals_col].iloc[i])
        u_now = float(prices[i]); u_next = float(prices[i+1])
        du = u_next - u_now
        t_next = times[i+1]

        dp = (delta_atm * du - theta_per_candle) if mode == "delta" else du

        if pos is None:
            if sig == "BUY":
                pos = "CE"; entry_p = max(5.0, base_mult * u_next); prem = entry_p; sl = entry_p - init_sl_pts
            elif sig == "SELL":
                pos = "PE"; entry_p = max(5.0, base_mult * u_next); prem = entry_p; sl = entry_p - init_sl_pts
            continue

        if pos == "CE":
            prem = prem + max(-prem, dp)
            profit = prem - entry_p
            if profit >= 10 and sl < entry_p: sl = entry_p
            if profit >= 20 and sl < entry_p + 10: sl = entry_p + 10
            if profit >= 30 and sl < entry_p + 15: sl = entry_p + 15
            if profit >= 50:
                new_sl = prem - (profit * 0.5)
                if new_sl > sl: sl = new_sl
            if prem <= sl:
                trades.append(dict(side="LONG CE", entry=entry_p, exit=sl, exit_time=t_next,
                                   pnl=(sl - entry_p) * lot_size))
                pos=None; entry_p=None; sl=None

        elif pos == "PE":
            prem = prem + max(-prem, -dp)
            profit = prem - entry_p
            if profit >= 10 and sl < entry_p: sl = entry_p
            if profit >= 20 and sl < entry_p + 10: sl = entry_p + 10
            if profit >= 30 and sl < entry_p + 15: sl = entry_p + 15
            if profit >= 50:
                new_sl = prem - (profit * 0.5)
                if new_sl > sl: sl = new_sl
            if prem <= sl:
                trades.append(dict(side="LONG PE", entry=entry_p, exit=sl, exit_time=t_next,
                                   pnl=(sl - entry_p) * lot_size))
                pos=None; entry_p=None; sl=None

    tr = pd.DataFrame(trades)
    if tr.empty:
        return tr, pd.Series(dtype=float), 50.0

    tr["pnl_points"] = tr["pnl"]
    tr["cum_pnl"] = tr["pnl_points"].cumsum()
    winrate = float((tr["pnl_points"] > 0).mean() * 100.0)
    score = float(np.clip(winrate, 0, 100))
    return tr, tr["cum_pnl"], score

# ------------------ Fibonacci helpers ------------------
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
            "0%": high, "23.6%": high - 0.236 * d, "38.2%": high - 0.382 * d,
            "50%": high - 0.5 * d, "61.8%": high - 0.618 * d,
            "78.6%": high - 0.786 * d, "100%": low
        }
    else:
        return {
            "0%": low, "23.6%": low + 0.236 * d, "38.2%": low + 0.382 * d,
            "50%": low + 0.5 * d, "61.8%": low + 0.618 * d,
            "78.6%": low + 0.786 * d, "100%": high
        }

def fib_confidence(df: pd.DataFrame, lookback: int = 200) -> tuple[float, dict]:
    sh, sl, dirn = swing(df, lookback)
    lv = fib_levels(sh, sl, "up" if dirn == "up" else "down")
    close = float(df["close"].iloc[-1])
    targets = [lv["38.2%"], lv["50%"], lv["61.8%"]]
    dist = min(abs(close - t) for t in targets)
    rng = max(1.0, (sh - sl) / 6.0)
    score = max(0.0, 100.0 - (dist / rng) * 100.0)
    return round(score, 2), lv            prem = prem + max(-prem, -dp)  # put gains when underlying falls
            profit = prem - entry_p
            if profit >= 10 and sl < entry_p:          sl = entry_p
            if profit >= 20 and sl < entry_p + 10:     sl = entry_p + 10
            if profit >= 30 and sl < entry_p + 15:     sl = entry_p + 15
            if profit >= 50:
                new_sl = prem - (profit * 0.5)
                if new_sl > sl: sl = new_sl
            if prem <= sl:
                trades.append(dict(side="LONG PE", entry=entry_p, exit=sl, exit_time=t_next,
                                   pnl=(sl-entry_p)*lot_size))
                pos=None; entry_p=None; sl=None

    tr = pd.DataFrame(trades)
    if tr.empty:
        return tr, pd.Series(dtype=float), 50.0

    tr["pnl_points"] = tr["pnl"]        # already in option points * lot
    tr["cum_pnl"] = tr["pnl_points"].cumsum()
    winrate = float((tr["pnl_points"] > 0).mean() * 100.0)
    score = float(np.clip(winrate, 0, 100))
    return tr, tr["cum_pnl"], score
# ------------------ Defaults & Weights ------------------
DEFAULT_SYMBOL = "NSEBANK.NS"

WEIGHTS_DEFAULT = {
    "trend": 20,
    "fibonacci": 25,
    "priceaction": 15,
    "smartmoney": 20,
    "backtest": 10,
    "others": 10,
}

# Yahoo intraday limits are strict; use these safe fallbacks (BankNifty only)
# Order matters: we try narrow intraday first, then broaden to daily/weekly.
INTERVAL_FALLBACKS: list[tuple[str, str]] = [
    ("5d",  "5m"),   # allowed intraday combo
    ("14d", "15m"),
    ("30d", "30m"),
    ("60d", "60m"),
    ("3mo", "1d"),   # market closed or intraday blocked
    ("2y",  "1wk"),  # ultimate fallback for plotting/scores
]

# ------------------ Core Fetchers ------------------
def fetch(
    symbol: str = DEFAULT_SYMBOL,
    period: str = "14d",
    interval: str = "5m",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV via yfinance and return a cleaned DataFrame with columns:
    ['open','high','low','close','volume'] and DateTimeIndex (tz-naive).
    Robust to MultiIndex/duplicate columns.
    NOTE: This tries ONLY the provided (period, interval). For smart cascading,
    use fetch_smart(...).
    """
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=auto_adjust,
            group_by="column",  # avoid MultiIndex
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex if any (paranoia)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[-1]) if isinstance(col, tuple) else str(col) for col in df.columns]

    # Standardize names
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
    })

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
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


def fetch_smart(
    symbol: str = DEFAULT_SYMBOL,
    prefer: tuple[str, str] | None = None,
    auto_adjust: bool = True,
    allow_symbol_fallback: bool = False,   # you earlier said NO fallback; keep False
) -> tuple[pd.DataFrame, tuple[str, str]]:
    """
    Try (period, interval) pairs until data arrives.
    Returns (df, (used_period, used_interval)).

    If allow_symbol_fallback=True, the function will also try 'NSEBANK.NS'
    in daily/weekly as a last resort (off by default per your preference).
    """
    tries: list[tuple[str, str]] = []
    if prefer and isinstance(prefer, tuple) and len(prefer) == 2:
        tries.append(prefer)
    for t in INTERVAL_FALLBACKS:
        if t not in tries:
            tries.append(t)

    # First pass: try requested symbol only
    for per, iv in tries:
        df = fetch(symbol, per, iv, auto_adjust=auto_adjust)
        if df is not None and not df.empty:
            return df, (per, iv)

    # Optional second pass: try symbol fallback (disabled by default)
    if allow_symbol_fallback:
        backup_symbol = "NSEBANK.NS"
        for per, iv in [("3mo", "1d"), ("2y", "1wk")]:
            df = fetch(backup_symbol, per, iv, auto_adjust=auto_adjust)
            if df is not None and not df.empty:
                return df, (per, iv)

    # Nothing worked
    return pd.DataFrame(), (tries[0][0] if tries else "5d", tries[0][1] if tries else "5m")

# ------------------ Sanitizer ------------------
def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe OHLCV sanitizer:
    - Flattens MultiIndex columns if any
    - Removes duplicate columns
    - Ensures each target column is 1-D Series, coerces to numeric
    - Drops incomplete rows
    - Ensures tz-naive DateTimeIndex named 'Date'
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[-1]) if isinstance(col, tuple) else str(col) for col in out.columns]

    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]

    cleaned = pd.DataFrame(index=out.index)
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            continue
        col_obj = out[col]
        if isinstance(col_obj, pd.DataFrame):     # if somehow duplicate-named columns
            col_obj = col_obj.iloc[:, 0]
        cleaned[col] = pd.to_numeric(col_obj, errors="coerce")

    cleaned = cleaned.dropna(subset=["open", "high", "low", "close", "volume"])

    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
        cleaned = cleaned.dropna().sort_index()

    cleaned.index.name = "Date"
    return cleaned

# ------------------ Fibonacci Helpers ------------------
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
            "0%": high, "23.6%": high - 0.236 * d, "38.2%": high - 0.382 * d,
            "50%": high - 0.5 * d, "61.8%": high - 0.618 * d,
            "78.6%": high - 0.786 * d, "100%": low
        }
    else:
        return {
            "0%": low, "23.6%": low + 0.236 * d, "38.2%": low + 0.382 * d,
            "50%": low + 0.5 * d, "61.8%": low + 0.618 * d,
            "78.6%": low + 0.786 * d, "100%": high
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
