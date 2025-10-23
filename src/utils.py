# src/utils.py
"""
Central utilities for BankNiftyApp
- fetch_smart: unified data fetcher (yfinance primary, AlphaVantage optional, CSV cache fallback)
- caching helpers
- is_market_hours() check (IST)
"""

import os
import time
import datetime
import pandas as pd

# Primary data provider
import yfinance as yf

# Optional: AlphaVantage (if user provides API key in environment)
try:
    from alpha_vantage.timeseries import TimeSeries
except Exception:
    TimeSeries = None

# Cache settings
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

DEFAULT_SYMBOL = "NSEBANK.NS"

def _cache_path(symbol: str, period: str, interval: str):
    safe = symbol.replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe}__{period}__{interval}.csv")

def _cache_save(symbol: str, period: str, interval: str, df: pd.DataFrame):
    try:
        path = _cache_path(symbol, period, interval)
        # write index as first column for easy reload
        df.to_csv(path)
    except Exception:
        pass

def _cache_load(symbol: str, period: str, interval: str):
    path = _cache_path(symbol, period, interval)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df
        except Exception:
            return None
    return None

def is_market_hours_ist() -> bool:
    # Indian market hours: Mon-Fri 09:15–15:30 IST
    now_utc = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    ist = now_utc.astimezone(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    if ist.weekday() >= 5:
        return False
    h, m = ist.hour, ist.minute
    start = (9, 15)
    end = (15, 30)
    after_start = (h > start[0]) or (h == start[0] and m >= start[1])
    before_end = (h < end[0]) or (h == end[0] and m <= end[1])
    return after_start and before_end

def _try_alpha_vantage(symbol: str, period: str, interval: str):
    # optional fallback: AlphaVantage (requires ALPHAVANTAGE_API_KEY in env)
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if api_key is None or TimeSeries is None:
        return None, "AlphaVantage unavailable"
    try:
        ts = TimeSeries(key=api_key, output_format='pandas', indexing_type='date')
        # map to a function based on interval
        if interval in ("1d", "daily"):
            df, meta = ts.get_daily_adjusted(symbol.replace(".NS", ""), outputsize='full')
            df = df.rename(columns={"1. open":"open","2. high":"high","3. low":"low","4. close":"close","6. volume":"volume"})
            df = df[["open","high","low","close","volume"]]
            return df, "AlphaVantage (daily)"
        # intraday support (can be 1min,5min,15min,30min,60min)
        if interval.endswith("m"):
            mins = interval.replace("m","")
            func = getattr(ts, f"get_intraday", None)
            if func:
                df, meta = ts.get_intraday(symbol.replace(".NS", ""), interval=f"{mins}min", outputsize='compact')
                df = df.rename(columns={"1. open":"open","2. high":"high","3. low":"low","4. close":"close","5. volume":"volume"})
                return df[["open","high","low","close","volume"]], "AlphaVantage (intraday)"
    except Exception as e:
        return None, f"AlphaVantage error: {e}"
    return None, "AlphaVantage: unsupported interval"

def _read_period_days(period: str):
    # convert common periods to days (best-effort)
    if isinstance(period, (list, tuple)):
        period = period[0]
    if isinstance(period, str):
        if period.endswith("d"):
            return int(period[:-1])
        if period.endswith("mo"):
            return int(period[:-2]) * 30
        if period == "max":
            return 3650
    return 7

def fetch_smart(symbol: str = DEFAULT_SYMBOL, period: str = "5d", interval: str = "5m", prefer=None):
    """
    Universal fetcher used across pages.
    - Accepts prefer=(period, interval) or period & interval separately.
    Returns: (df, used, msg)
      df -> pandas DataFrame with OHLCV and datetime index
      used -> (period_used, interval_used)
      msg -> status string for UI
    Behavior:
      - Try yfinance.download (primary)
      - If fails or empty, try AlphaVantage if API key present
      - If still fails, try cached CSV for requested period/interval
      - Last fallback: try daily 6mo/1d via yfinance
    """

    # allow prefer tuple
    if prefer and isinstance(prefer, (list, tuple)) and len(prefer) == 2:
        period, interval = prefer[0], prefer[1]
    used = (period, interval)
    msg_parts = []

    # 1) Primary: yfinance
    try:
        # yfinance expects e.g., period="5d", interval="5m"
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            # Ensure standard column names lowercase
            df = df.rename(columns={c: c.lower() for c in df.columns})
            _cache_save(symbol, period, interval, df)
            msg_parts.append(f"Live: yfinance ({period}/{interval})")
            return df, used, " | ".join(msg_parts)
        else:
            msg_parts.append("yfinance returned empty")
    except Exception as e:
        msg_parts.append(f"yfinance error: {e}")

    # 2) Try AlphaVantage (if configured)
    av_df, av_msg = _try_alpha_vantage(symbol, period, interval)
    if av_df is not None and not av_df.empty:
        av_df = av_df.rename(columns={c: c.lower() for c in av_df.columns})
        _cache_save(symbol, period, interval, av_df)
        msg_parts.append(av_msg)
        return av_df, used, " | ".join(msg_parts)
    else:
        msg_parts.append(av_msg)

    # 3) Check local cache (specific period/interval)
    cached = _cache_load(symbol, period, interval)
    if cached is not None and not cached.empty:
        msg_parts.append(f"Cache used: {period}/{interval} ({len(cached)} rows)")
        return cached, used, " | ".join(msg_parts)

    # 4) Daily fallback (6mo/1d)
    try:
        df_daily = yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=False)
        if df_daily is not None and not df_daily.empty:
            df_daily = df_daily.rename(columns={c: c.lower() for c in df_daily.columns})
            _cache_save(symbol, "6mo", "1d", df_daily)
            msg_parts.append("Fallback: yfinance (6mo/1d)")
            return df_daily, ("6mo", "1d"), " | ".join(msg_parts)
        else:
            msg_parts.append("daily fallback empty")
    except Exception as e:
        msg_parts.append(f"daily fallback error: {e}")

    # 5) Last resort: try any cached file that matches symbol
    # pick most recent cache for symbol
    candidates = []
    base = os.path.join(CACHE_DIR, "")
    for f in os.listdir(CACHE_DIR):
        if f.startswith(symbol.replace("/", "_")) or f.startswith(symbol.replace(".", "_")):
            candidates.append(os.path.join(CACHE_DIR, f))
    if candidates:
        # prefer largest/more recent
        candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
        try:
            df_any = pd.read_csv(candidates[0], index_col=0, parse_dates=True)
            msg_parts.append(f"Using other cache: {os.path.basename(candidates[0])}")
            return df_any, used, " | ".join(msg_parts)
        except Exception:
            msg_parts.append("other cache load failed")

    # completely failed
    msg_parts.append("All sources failed")
    return pd.DataFrame(), used, " | ".join(msg_parts)
