# utils.py — Clean FINAL (Fast mode, no cache)
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf  # backup only

# =============================================================================
# Global defaults & weights
# =============================================================================

# 👉 BankNifty default (works on cloud with yfinance fallback & our NSE engine)
DEFAULT_SYMBOL = "NSEBANK.NS"

WEIGHTS_DEFAULT: Dict[str, int] = {
    "trend": 20,
    "fibonacci": 25,
    "priceaction": 15,
    "smartmoney": 20,
    "backtest": 10,
    "others": 10,
}

# Backup fallbacks (used only if NSE engine fails)
INTERVAL_FALLBACKS: List[Tuple[str, str]] = [
    ("5d", "5m"),
    ("14d", "15m"),
    ("30d", "30m"),
    ("60d", "60m"),
    ("3mo", "1d"),
    ("2y", "1wk"),
]

# =============================================================================
# NSE (TradingView-like) engine
# =============================================================================

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

def _to_nse_index_name(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    if s in {"^NSEBANK", "NSEBANK", "NSEBANK.NS", "BANKNIFTY", "BANKNIFTY.NS"}:
        return "NIFTY BANK"
    # App is BankNifty-only by design
    return "NIFTY BANK"

def _interval_to_nse(interval: str) -> str:
    m = {
        "1m": "1minute",
        "2m": "2minute",
        "5m": "5minute",
        "10m": "10minute",
        "15m": "15minute",
        "30m": "30minute",
        "60m": "60minute",
        "1h": "60minute",
        "1d": "1day",
        "1D": "1day",
        "daily": "1day",
    }
    return m.get(interval, "5minute")

def _period_days(period: str) -> int:
    m = {
        "5d": 5, "7d": 7, "10d": 10, "14d": 14, "30d": 30, "60d": 60,
        "3mo": 90, "6mo": 180, "1y": 365, "2y": 730
    }
    return m.get(period, 14)

def fetch_nse_index(
    symbol: str = DEFAULT_SYMBOL,
    period: str = "14d",
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Try NSE chart APIs to fetch OHLCV for BankNifty.
    Returns DataFrame with columns: open, high, low, close, volume (DateTimeIndex).
    """
    try:
        idx = _to_nse_index_name(symbol)
        iv = _interval_to_nse(interval)
        days = _period_days(period)

        to_ts = int(time.time())
        frm_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

        base = "https://www.nseindia.com"
        candles_url = (
            f"{base}/api/candles/index?"
            f"index={requests.utils.quote(idx)}&from={frm_ts}&to={to_ts}&interval={iv}"
        )
        chart_url = (
            f"{base}/api/chart-databyindex?"
            f"index={requests.utils.quote(idx)}&indices=true&from={frm_ts}&to={to_ts}"
        )

        sess = requests.Session()
        # bootstrap cookies
        sess.get(base, headers=_TV_HEADERS, timeout=10)

        # Primary: candles endpoint
        r = sess.get(candles_url, headers=_TV_HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and isinstance(data.get("candles"), list) and data["candles"]:
                rows = data["candles"]
                df = pd.DataFrame(rows, columns=["Date", "open", "high", "low", "close", "volume"])
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
                return sanitize_ohlcv(df)

        # Fallback: chart-databyindex (some envs expose 'candles' or 'grapthData')
        r2 = sess.get(chart_url, headers=_TV_HEADERS, timeout=15)
        if r2.status_code == 200:
            data2 = r2.json()
            if isinstance(data2, dict):
                if isinstance(data2.get("candles"), list) and data2["candles"]:
                    rows = data2["candles"]
                    df = pd.DataFrame(rows, columns=["Date", "open", "high", "low", "close", "volume"])
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
                    return sanitize_ohlcv(df)
                if isinstance(data2.get("grapthData"), list) and data2["grapthData"]:
                    rows = data2["grapthData"]
                    tmp = pd.DataFrame(rows, columns=["ts", "close"])
                    tmp["Date"] = pd.to_datetime(tmp["ts"], unit="ms", utc=True).dt.tz_convert(None)
                    tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
                    tmp = tmp.dropna(subset=["Date", "close"]).set_index("Date").sort_index()
                    # create pseudo OHLC
                    tmp["open"] = tmp["close"].shift(1).fillna(tmp["close"])
                    tmp["high"] = tmp[["open", "close"]].max(axis=1)
                    tmp["low"]  = tmp[["open", "close"]].min(axis=1)
                    tmp["volume"] = 0.0
                    df = tmp[["open", "high", "low", "close", "volume"]]
                    return sanitize_ohlcv(df)

    except Exception:
        # swallow errors and let caller fallback
        pass

    return pd.DataFrame()

# =============================================================================
# yfinance backup (used only if NSE engine fails)
# =============================================================================

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
        df.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in df.columns]

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

# =============================================================================
# Public fetch APIs (used by pages)
# =============================================================================

def fetch(
    symbol: str = DEFAULT_SYMBOL,
    period: str = "14d",
    interval: str = "5m",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Simple fetch: NSE first, then yfinance, then fallbacks."""
    df = fetch_nse_index(symbol, period, interval)
    if df is not None and not df.empty:
        return df

    df = fetch_yf(symbol, period, interval, auto_adjust)
    if df is not None and not df.empty:
        return df

    for per, iv in INTERVAL_FALLBACKS:
        df = fetch_yf(symbol, per, iv, auto_adjust)
        if df is not None and not df.empty:
            return df

    return pd.DataFrame()

def fetch_smart(
    symbol: str = DEFAULT_SYMBOL,
    prefer: Tuple[str, str] | None = None,
    auto_adjust: bool = True,
    allow_symbol_fallback: bool = False,
) -> Tuple[pd.DataFrame, Tuple[str, str]]:
    """Try prefer first; NSE first → yfinance; then fallbacks."""
    tries: List[Tuple[str, str]] = []
    if prefer and isinstance(prefer, tuple) and len(prefer) == 2:
        tries.append(prefer)
    for t in INTERVAL_FALLBACKS:
        if t not in tries:
            tries.append(t)

    for per, iv in tries:
        df = fetch_nse_index(symbol, per, iv)
        if df is not None and not df.empty:
            return df, (per, iv)
        df = fetch_yf(symbol, per, iv, auto_adjust)
        if df is not None and not df.empty:
            return df, (per, iv)

    if allow_symbol_fallback:
        alt = "NSEBANK.NS"
        for per, iv in [("3mo", "1d"), ("2y", "1wk")]:
            df = fetch_nse_index(alt, per, iv)
            if df is not None and not df.empty:
                return df, (per, iv)
            df = fetch_yf(alt, per, iv, auto_adjust)
            if df is not None and not df.empty:
                return df, (per, iv)

    # if nothing worked, return empty with first try tuple
    if tries:
        return pd.DataFrame(), tries[0]
    return pd.DataFrame(), ("5d", "5m")

# =============================================================================
# Data cleaning
# =============================================================================

def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric OHLCV + DateTimeIndex."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in out.columns]
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]

    cleaned = pd.DataFrame(index=out.index)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            series = out[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            cleaned[col] = pd.to_numeric(series, errors="coerce")
    cleaned = cleaned.dropna(subset=["open", "high", "low", "close", "volume"])

    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
        cleaned = cleaned.dropna().sort_index()
    else:
        try:
            cleaned.index = cleaned.index.tz_convert(None)
        except Exception:
            pass

    cleaned.index.name = "Date"
    return cleaned

# =============================================================================
# Strategy helpers
# =============================================================================

def generate_signals_50pct(df: pd.DataFrame, mid_factor: float = 0.5) -> pd.DataFrame:
    """5m rule: current low≥prev_mid & close>prev_high → BUY; opposite for SELL."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    sig = ["HOLD"]
    for i in range(1, len(out)):
        ph = float(out["high"].iloc[i-1])
        pl = float(out["low"].iloc[i-1])
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

def simulate_atm_option_trades(
    df: pd.DataFrame,
    signals_col: str = "signal",
    init_sl_pts: float = 10.0,
    lot_size: int = 15,
    mode: str = "delta",            # "delta" or "index"
    delta_atm: float = 0.5,
    theta_per_candle: float = 0.0,
):
    """Simple ATM CE/PE sim with trailing ladder. Returns trades_df, equity_series, score."""
    if df is None or df.empty or signals_col not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float), 50.0

    prices = df["close"].astype(float).values
    times = list(df.index)

    pos = None
    entry_p = None
    sl = None
    trades = []
    prem = 0.0
    base_mult = 0.01  # baseline premium ~1% of index

    for i in range(len(df) - 1):
        sig = str(df[signals_col].iloc[i])
        u_now = float(prices[i])
        u_next = float(prices[i + 1])
        du = u_next - u_now
        t_next = times[i + 1]

        dp = (delta_atm * du - theta_per_candle) if mode == "delta" else du

        if pos is None:
            if sig == "BUY":
                pos = "CE"
                entry_p = max(5.0, base_mult * u_next)
                prem = entry_p
                sl = entry_p - init_sl_pts
            elif sig == "SELL":
                pos = "PE"
                entry_p = max(5.0, base_mult * u_next)
                prem = entry_p
                sl = entry_p - init_sl_pts
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
                pos = None; entry_p = None; sl = None

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
                pos = None; entry_p = None; sl = None

    tr = pd.DataFrame(trades)
    if tr.empty:
        return tr, pd.Series(dtype=float), 50.0

    tr["pnl_points"] = tr["pnl"]
    tr["cum_pnl"] = tr["pnl_points"].cumsum()
    winrate = float((tr["pnl_points"] > 0).mean() * 100.0)
    score = float(np.clip(winrate, 0, 100))
    return tr, tr["cum_pnl"], score

# =============================================================================
# Fibonacci helpers
# =============================================================================

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

def fib_confidence(df: pd.DataFrame, lookback: int = 200) -> Tuple[float, dict]:
    sh, sl, dirn = swing(df, lookback)
    lv = fib_levels(sh, sl, "up" if dirn == "up" else "down")
    close = float(df["close"].iloc[-1])
    targets = [lv["38.2%"], lv["50%"], lv["61.8%"]]
    dist = min(abs(close - t) for t in targets)
    rng = max(1.0, (sh - sl) / 6.0)
    score = max(0.0, 100.0 - (dist / rng) * 100.0)
    return round(score, 2), lv
