# tv_client.py
# TradingView data client for NSE Index (BankNifty) – intraday & daily
# Works with: BANKNIFTY (index), exchange="NSE"

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import os
import pandas as pd

try:
    from tvDatafeed import TvDatafeed, Interval
    _TV_AVAILABLE = True
except Exception:
    _TV_AVAILABLE = False

# -------- Public API --------

def tv_supported() -> bool:
    """Return True if TradingView client is importable."""
    return _TV_AVAILABLE

def tv_fetch_candles(
    symbol: str = "BANKNIFTY",
    exchange: str = "NSE",
    interval: str = "5m",
    period: str = "5d",
    bars_limit: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV from TradingView for NSE index.
    symbol: 'BANKNIFTY'
    exchange: 'NSE'
    interval: '1m'|'5m'|'15m'|'1h'|'1d'
    period: '2d','5d','7d','14d','1mo','3mo','6mo','1y','max' (best-effort bars calc)
    bars_limit: override bar count
    """
    if not _TV_AVAILABLE:
        raise RuntimeError("tvdatafeed not available. Install in requirements.txt.")

    tv_interval = _map_interval(interval)
    bars = bars_limit if bars_limit else _estimate_bars(period, interval)

    # TradingView sometimes throttles anonymous sessions.
    # You can set env vars TV_USER / TV_PASS in Streamlit Cloud secrets if needed.
    user = username or os.getenv("TV_USER", None)
    pwd  = password or os.getenv("TV_PASS", None)

    tv = TvDatafeed(username=user, password=pwd)  # logs in; if None -> guest mode

    df = tv.get_hist(
        symbol=symbol,
        exchange=exchange,
        interval=tv_interval,
        n_bars=bars
    )
    if df is None or df.empty:
        raise RuntimeError("TradingView returned empty dataset.")

    # tvdatafeed returns columns: open high low close volume, and Datetime index (tz-aware sometimes)
    out = df.copy()
    out.index.name = "Date"
    out.reset_index(inplace=True)
    # force tz-naive
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None, ambiguous="NaT", nonexistent="NaT")

    out = out.rename(columns={
        "open":"open","high":"high","low":"low","close":"close","volume":"volume"
    })[["Date","open","high","low","close","volume"]].dropna()

    out.set_index("Date", inplace=True)
    return out.sort_index()

# -------- Helpers --------

def _map_interval(iv: str) -> "Interval":
    iv = iv.lower().strip()
    if iv in ("1m","1min","1"):   return Interval.in_1_minute
    if iv in ("2m","2min"):       return Interval.in_2_minute
    if iv in ("5m","5min"):       return Interval.in_5_minute
    if iv in ("15m","15min"):     return Interval.in_15_minute
    if iv in ("30m","30min"):     return Interval.in_30_minute
    if iv in ("1h","60m"):        return Interval.in_1_hour
    if iv in ("4h","240m"):       return Interval.in_4_hour
    if iv in ("1d","d","daily"):  return Interval.in_daily
    raise ValueError(f"Unsupported interval: {iv}")

def _estimate_bars(period: str, interval: str) -> int:
    """
    Crude estimate of bars from period & interval for NSE cash session (~6.25h ~375m).
    Good enough for pulling last N bars window per page.
    """
    # days from period
    p = period.lower().strip()
    if p.endswith("mo"):
        months = int(p.replace("mo",""))
        days = months * 30
    elif p.endswith("y"):
        years = int(p.replace("y",""))
        days = years * 365
    elif p.endswith("d"):
        days = int(p.replace("d",""))
    elif p == "max":
        days = 365 * 5
    else:
        days = 7  # default

    # bars/day by interval
    iv = interval.lower().strip()
    if iv == "1m":
        bars_day = 375
    elif iv == "5m":
        bars_day = 75
    elif iv == "15m":
        bars_day = 25
    elif iv in ("1h","60m"):
        bars_day = 6
    elif iv in ("1d","d","daily"):
        bars_day = 1
    else:
        bars_day = 50

    # safety cap
    bars = min(days * bars_day, 15000)
    return max(bars, bars_day * 2)  # at least 2 sessions
