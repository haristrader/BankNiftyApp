# data_engine.py
from __future__ import annotations

import os
import time
import json
import math
import typing as t
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import httpx

# Optional fallbacks
try:
    import yfinance as yf
except Exception:
    yf = None

# ---- CONFIG ---------------------------------------------------------------

ALPHA_KEY = os.getenv("ALPHAVANTAGE_KEY", "").strip()
TD_KEY     = os.getenv("TWELVEDATA_KEY", "").strip()

# supported intraday TF map (ui → api)
TF_MAP_TWELVE = {
    "5m": "5min",
    "15m": "15min",
    "1h": "60min",
}
# daily for both
DAILY_TF = "1day"

# NSE symbol mapping for APIs
# Pages me aap NSEBANK.NS, HDFCBANK.NS etc pass karoge
def map_symbol_for_api(symbol: str) -> dict:
    s = symbol.strip().upper()
    out = {"alpha": s, "td": s, "yahoo": symbol}

    # Common NSE conversions
    if s.endswith(".NS"):
        base = s.replace(".NS", "")
        out["alpha"] = f"NSE:{base}"
        out["td"] = f"NSE:{base}"

    # BankNifty special handling
    if s in ("NSEBANK.NS", "NIFTYBANK.NS", "^NSEBANK", "^NIFTYBANK", "NSEBANK"):
        out["alpha"] = "NSE:BANKNIFTY"
        out["td"] = "NSE:BANKNIFTY"
        out["yahoo"] = "^NSEBANK"

    return out

# ---- NORMALIZATION --------------------------------------------------------

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not len(df):
        return df
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.set_index("datetime")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.set_index("date")
    elif df.index.name not in ("date", "datetime"):
        # try index parse
        try:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        except Exception:
            pass
    df = df.sort_index()
    return df

def _to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # try common columns
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("open") and "open" not in ren.values(): ren[c] = "open"
        elif lc.startswith("high") and "high" not in ren.values(): ren[c] = "high"
        elif lc.startswith("low") and "low" not in ren.values(): ren[c] = "low"
        elif lc.startswith("close") and "close" not in ren.values(): ren[c] = "close"
        elif lc in ("volume", "vol"): ren[c] = "volume"
    if ren:
        df = df.rename(columns=ren)
    for need in ["open", "high", "low", "close"]:
        if need not in df.columns:
            df[need] = np.nan
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[["open","high","low","close","volume"]]

# ---- LIVE FETCHERS --------------------------------------------------------

async def _get_json(url: str, params: dict) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

def _get_json_sync(url: str, params: dict) -> dict:
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()

def fetch_twelvedata(symbol: str, interval: str, outputsize: int = 5000) -> pd.DataFrame:
    if not TD_KEY:
        return pd.DataFrame()
    sym = map_symbol_for_api(symbol)["td"]
    params = dict(
        symbol=sym, interval=interval, outputsize=outputsize,
        apikey=TD_KEY, timezone="UTC"
    )
    url = "https://api.twelvedata.com/time_series"
    try:
        data = _get_json_sync(url, params)
        if "values" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        # columns: datetime, open, high, low, close, volume
        df = _ensure_dt_index(df)
        df = _to_ohlcv(df.apply(pd.to_numeric, errors="ignore"))
        return df
    except Exception:
        return pd.DataFrame()

def fetch_alphavantage_daily(symbol: str) -> pd.DataFrame:
    # TIME_SERIES_DAILY_ADJUSTED
    if not ALPHA_KEY:
        return pd.DataFrame()
    sym = map_symbol_for_api(symbol)["alpha"]
    url = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": sym, "outputsize": "full", "apikey": ALPHA_KEY}
    try:
        data = _get_json_sync(url, params)
        # response has 'Time Series (Daily)'
        key = None
        for k in data.keys():
            if "Time Series" in k:
                key = k
                break
        if not key:
            return pd.DataFrame()
        rows = []
        for dt, vals in data[key].items():
            rows.append({
                "date": dt,
                "open": float(vals.get("1. open", "nan")),
                "high": float(vals.get("2. high", "nan")),
                "low": float(vals.get("3. low", "nan")),
                "close": float(vals.get("4. close", "nan")),
                "volume": float(vals.get("6. volume", "0")),
            })
        df = pd.DataFrame(rows)
        df = _ensure_dt_index(df)
        df = _to_ohlcv(df)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_yfinance(symbol: str, period: str, interval: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if isinstance(df, pd.DataFrame) and len(df):
            df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
            df.index = pd.to_datetime(df.index, utc=True)
            df = _to_ohlcv(df)
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ---- PUBLIC: SMART FETCH --------------------------------------------------

def fetch_smart(
    symbol: str,
    prefer_period: str = "5d",
    prefer_interval: str = "5m",
    mode: str = "auto"  # "daily-only" forces Alpha; "intraday" forces TD
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (df, meta)
    meta = {"source": "twelvedata/alphavantage/yfinance", "used": {"period":.., "interval":..}}
    """
    meta = {"source": None, "used": {"period": prefer_period, "interval": prefer_interval}}

    # Decide daily vs intraday
    intraday = prefer_interval in TF_MAP_TWELVE

    if mode == "daily-only":
        intraday = False
    elif mode == "intraday":
        intraday = True

    # Try intraday first (TwelveData)
    if intraday:
        td_interval = TF_MAP_TWELVE.get(prefer_interval, "5min")
        df = fetch_twelvedata(symbol, interval=td_interval)
        if len(df):
            meta["source"] = "twelvedata"
            return df, meta

    # Daily (history): Alpha Vantage
    df = fetch_alphavantage_daily(symbol)
    if len(df):
        meta["source"] = "alphavantage"
        return df, meta

    # HARD fallback: yfinance (kept for indices/edge)
    yf_period = prefer_period if intraday else "5y"
    yf_interval = prefer_interval if intraday else "1d"
    df = fetch_yfinance(map_symbol_for_api(symbol)["yahoo"], period=yf_period, interval=yf_interval)
    if len(df):
        meta["source"] = "yfinance"
        return df, meta

    return pd.DataFrame(), meta
