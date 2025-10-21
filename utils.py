import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from supabase import create_client, Client

# --- SUPABASE SETUP (Replace with your project credentials) ---
SUPABASE_URL = "https://your-project-url.supabase.co"
SUPABASE_KEY = "your-anon-or-service-key"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------
# 🧱 BASIC DATA FETCHERS
# ---------------------------------------------------------------

def get_candles(symbol="NSEBANK.NS", period="5d", interval="5m"):
    """
    Fetch OHLCV candles from Yahoo Finance.
    Handles MultiIndex safely and returns a clean DataFrame.
    """
    df = yf.download(symbol, period=period, interval=interval, progress=False)

    # Normalize columns (handle MultiIndex safely)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    df.reset_index(inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    return df


def fetch_smart(symbol="NSEBANK.NS", period="5d", interval="5m"):
    """
    Wrapper for get_candles with smart error handling.
    """
    try:
        df = get_candles(symbol, period, interval)
        msg = f"✅ Data fetched successfully: {len(df)} rows."
        return df, msg
    except Exception as e:
        msg = f"❌ Fetch error: {str(e)}"
        return pd.DataFrame(), msg


# ---------------------------------------------------------------
# 🧩 SUPABASE OPERATIONS
# ---------------------------------------------------------------

def save_candles_supabase(df, table_name="banknifty_candles"):
    """
    Save OHLC data to Supabase table.
    """
    if df.empty:
        return "⚠️ No data to upload."

    records = df.to_dict(orient="records")
    try:
        response = supabase.table(table_name).insert(records).execute()
        return f"✅ Uploaded {len(records)} rows to Supabase table: {table_name}"
    except Exception as e:
        return f"❌ Supabase upload failed: {str(e)}"


# ---------------------------------------------------------------
# 🔁 BACKWARD COMPATIBILITY ALIASES (for old imports)
# ---------------------------------------------------------------

def sb_save_candles(*args, **kwargs):
    """
    Alias for older modules still importing sb_save_candles
    """
    return save_candles_supabase(*args, **kwargs)


def generate_signals_50pct(df):
    """
    Simple mid-zone signal generator used in older Backtest versions.
    """
    if df.empty or "high" not in df or "low" not in df or "close" not in df:
        return df

    df["mid"] = (df["high"] + df["low"]) / 2
    df["signal"] = np.where(df["close"] > df["mid"], 1, -1)
    return df


# ---------------------------------------------------------------
# 🧠 HELPER UTILITIES
# ---------------------------------------------------------------

def format_date(dt):
    """Convert datetime to standard ISO format."""
    if isinstance(dt, (datetime, pd.Timestamp)):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt)


def log_message(msg):
    """Console + Supabase logging (optional extension)."""
    print(msg)
    # Optional: store logs in Supabase if needed
    # supabase.table("logs").insert({"timestamp": datetime.now().isoformat(), "message": msg}).execute()
    return msg
