# tv_client.py
from __future__ import annotations
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from functools import lru_cache

# --- map our app's interval -> TradingView interval
_TV_INTERVAL = {
    "5m": Interval.in_5_minute,
    "15m": Interval.in_15_minute,
    "1h": Interval.in_1_hour,
    "1d": Interval.in_daily,
}

# --- rough bars-per-period for NSE equity hours (intraday)
def _bars_for(period:str, interval:str) -> int:
    # simple & safe: over-fetch a bit, trim downstream
    table = {
        "5m":  {"5d": 600, "14d": 1400, "1mo": 2500, "3mo": 6000},
        "15m": {"5d": 250, "14d": 600, "1mo": 1200, "3mo": 3000},
        "1h":  {"14d": 400, "1mo": 800, "3mo": 1800, "6mo": 3600},
        "1d":  {"3mo": 120, "6mo": 240, "1y": 400, "max": 3000},
    }
    if interval not in table or period not in table[interval]:
        return 1000
    return table[interval][period]

@lru_cache(maxsize=1)
def _client() -> TvDatafeed:
    # guest mode works; no creds needed
    return TvDatafeed()

def _ensure_tv_symbol(symbol:str) -> tuple[str, str]:
    """
    Map our symbols to TradingView format. For BankNifty:
    - Yahoo: 'NSEBANK.NS' or '^NSEBANK'
    - TradingView: 'NSE:BANKNIFTY'
    """
    s = symbol.upper()
    if "NSEBANK" in s or "BANKNIFTY" in s:
        return "NSE", "BANKNIFTY"
    # add more if you add stocks later (e.g. "HDFCBANK.NS" -> ("NSE","HDFCBANK"))
    if s.endswith(".NS"):
        base = s.replace(".NS", "")
        return "NSE", base
    # default safe
    return "NSE", "BANKNIFTY"

def tv_fetch(symbol:str, period:str="5d", interval:str="5m") -> pd.DataFrame:
    """
    Get OHLCV from TradingView. Returns DataFrame with:
    index=datetime, columns=[open,high,low,close,volume]
    """
    exch, tv_sym = _ensure_tv_symbol(symbol)
    tv_int = _TV_INTERVAL.get(interval, Interval.in_5_minute)
    n_bars = _bars_for(period, interval)

    df = _client().get_hist(symbol=tv_sym, exchange=exch, interval=tv_int, n_bars=n_bars)
    if df is None or df.empty:
        return pd.DataFrame()

    # tvDatafeed columns: ['datetime','open','high','low','close','volume']
    out = df.copy()
    out.index.name = "Date"
    out = out.rename(columns=str.lower)
    out = out[["open","high","low","close","volume"]].sort_index()
    try:
        out.index = pd.to_datetime(out.index, utc=True).tz_convert(None)
    except Exception:
        out.index = pd.to_datetime(out.index)
    return out
