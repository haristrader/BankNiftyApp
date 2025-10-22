# src/utils.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Tuple, Optional

# --- Optional: supabase client (only if you set env and installed supabase) ---
try:
    from supabase import create_client, Client

import os

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
supabase: Optional[Client] = None

try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"[utils] Supabase init failed: {e}") if SUPABASE_URL and SUPABASE_KEY else None
except Exception:
    supabase = None

# ----------------- Globals -----------------
DEFAULT_SYMBOL = "NSEBANK.NS"
DEFAULT_INTERVAL = "5m"
DEFAULT_PERIOD = "5d"

# ----------------- Market hours util -----------------
def is_market_hours() -> bool:
    """
    Quick IST-based market hours check.
    Convert to UTC times: 9:15 IST = 3:45 UTC, 15:30 IST = 10:00 UTC
    (Simple check — good enough for UI messages)
    """
    now_utc = datetime.utcnow().time()
    open_utc = time(3, 45)
    close_utc = time(10, 0)
    return open_utc <= now_utc <= close_utc

# ----------------- Candle fetcher -----------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Convert multiindex or plain columns to lowercase strings
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                # prefer first meaningful level
                cols.append(str(c[0]).lower())
            else:
                cols.append(str(c).lower())
        df.columns = cols
    else:
        df.columns = [str(c).lower() for c in df.columns]
    return df

def get_candles(symbol: str = DEFAULT_SYMBOL, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """
    Download OHLCV using yfinance and return a clean DataFrame.
    Returns empty DataFrame on error.
    """
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = _normalize_columns(df)
        # Reset index -> keep timestamp as 'date'
        df = df.reset_index()
        # some downloads name column 'Datetime' or 'Date'
        for candidate in ("datetime", "date", "index"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "date"})
                break
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        # ensure OHLCV exists (some intervals not supported)
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(df.columns)):
            # try to map common alt names
            cols = set(df.columns)
            if "adj close" in cols and "close" not in cols:
                df = df.rename(columns={"adj close": "close"})
            else:
                # missing required fields -> return empty so callers can handle gracefully
                return pd.DataFrame()
        df = df.dropna(how="any")
        return df
    except Exception as e:
        print(f"[utils.get_candles] Exception: {e}")
        return pd.DataFrame()

# ----------------- Smart wrapper -----------------
def fetch_smart(symbol: str = DEFAULT_SYMBOL, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> Tuple[pd.DataFrame, str]:
    """
    Unified fetch for pages. Returns (df, message).
    Keeps signature simple (no unexpected kwargs).
    """
    try:
        df = get_candles(symbol=symbol, period=period, interval=interval)
        if df.empty:
            msg = "No data received. Market may be closed or source unreachable."
            return pd.DataFrame(), msg
        return df, f"Fetched {len(df)} rows for {symbol} ({interval})"
    except Exception as e:
        return pd.DataFrame(), f"Fetch error: {e}"

# ----------------- Save candles (supabase) -----------------
def save_candles_supabase(df: pd.DataFrame, table_name: str = "candles") -> str:
    if df is None or df.empty:
        return "No data to save"
    if supabase is None:
        return "Supabase not configured"
    try:
        records = df.to_dict(orient="records")
        supabase.table(table_name).insert(records).execute()
        return f"Saved {len(records)} rows to {table_name}"
    except Exception as e:
        return f"Supabase save error: {e}"

# backward-compatible alias
def sb_save_candles(*args, **kwargs):
    return save_candles_supabase(*args, **kwargs)

# ----------------- Signal generator -----------------
def generate_signals_50pct(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not {"high", "low", "close"}.issubset(set(df.columns)):
        return pd.DataFrame()
    df = df.copy()
    df["mid"] = (df["high"] + df["low"]) / 2.0
    df["signal"] = np.where(df["close"] > df["mid"], 1, -1)
    return df

# ----------------- ATM simulation stub (used by backtest pages) -----------------
def simulate_atm_option_trades(df: pd.DataFrame, **kwargs) -> dict:
    """
    Minimal stub to satisfy imports. Returns a simple summary.
    Replace with real simulation when available.
    """
    if df is None or df.empty:
        return {"trades": [], "summary": "no data"}
    # simple placeholder: count rows
    return {"trades": [], "summary": f"simulated over {len(df)} bars"}

# ----------------- 50pct alias same name used elsewhere -----------------
def generate_signals_50pct_simple(df):
    return generate_signals_50pct(df)

# ----------------- Record module score (AI console) -----------------
def sb_record_module_score(module: str, score: float, table_name: str = "module_scores") -> bool:
    try:
        if supabase is None:
            print("[sb_record_module_score] Supabase not configured")
            return False
        record = {"module": module, "score": float(score), "ts": datetime.utcnow().isoformat()}
        supabase.table(table_name).insert(record).execute()
        return True
    except Exception as e:
        print(f"[sb_record_module_score] Error: {e}")
        return False

# ----------------- Helpers -----------------
def format_date(dt):
    if isinstance(dt, (pd.Timestamp, datetime)):
        return dt.isoformat()
    return str(dt)
