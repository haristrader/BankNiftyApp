import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time
from supabase import create_client, Client

# ---------------------------------------------------------------
# 🧱 SUPABASE CONFIG (update your credentials)
# ---------------------------------------------------------------
SUPABASE_URL = "https://your-project-url.supabase.co"
SUPABASE_KEY = "your-anon-or-service-key"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------------
# ⚙️ GLOBAL CONSTANTS
# ---------------------------------------------------------------
DEFAULT_SYMBOL = "NSEBANK.NS"     # for pages importing from utils
DEFAULT_INTERVAL = "5m"
DEFAULT_PERIOD = "5d"

# ---------------------------------------------------------------
# 🧩 MARKET HOURS CHECK
# ---------------------------------------------------------------
def is_market_hours():
    """Returns True if current time is between 9:15 and 15:30 IST."""
    now = datetime.utcnow().time()
    market_open = time(3, 45)   # 9:15 IST in UTC
    market_close = time(10, 0)  # 15:30 IST in UTC
    return market_open <= now <= market_close

# ---------------------------------------------------------------
# 📊 CANDLE FETCHING
# ---------------------------------------------------------------
def get_candles(symbol=DEFAULT_SYMBOL, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    """Download OHLCV data from Yahoo Finance and normalize column names."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)

        # Fix MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        df.reset_index(inplace=True)
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"❌ get_candles error: {str(e)}")
        return pd.DataFrame()

# ---------------------------------------------------------------
# 🔹 SMART FETCH WRAPPER
# ---------------------------------------------------------------
def fetch_smart(symbol=DEFAULT_SYMBOL, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    """Unified fetch layer for all modules."""
    try:
        df = get_candles(symbol, period, interval)
        msg = f"✅ {symbol}: {len(df)} rows fetched ({interval})" if not df.empty else f"⚠️ No data found."
        return df, msg
    except Exception as e:
        return pd.DataFrame(), f"❌ Fetch failed: {str(e)}"

# ---------------------------------------------------------------
# 💾 SUPABASE UPLOAD
# ---------------------------------------------------------------
def save_candles_supabase(df, table_name="banknifty_candles"):
    """Upload candles to Supabase (safe version)."""
    if df.empty:
        return "⚠️ No data to upload."
    try:
        records = df.to_dict(orient="records")
        supabase.table(table_name).insert(records).execute()
        return f"✅ Uploaded {len(records)} rows to {table_name}"
    except Exception as e:
        return f"❌ Supabase upload failed: {str(e)}"

# backward alias for old imports
def sb_save_candles(*args, **kwargs):
    return save_candles_supabase(*args, **kwargs)

# ---------------------------------------------------------------
# 🧮 SIGNAL GENERATION
# ---------------------------------------------------------------
def generate_signals_50pct(df):
    """Basic 50% mid-zone signal generator."""
    if df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return df
    df["mid"] = (df["high"] + df["low"]) / 2
    df["signal"] = np.where(df["close"] > df["mid"], 1, -1)
    return df

# ---------------------------------------------------------------
# 🧠 RECORDING MODULE SCORES
# ---------------------------------------------------------------
def sb_record_module_score(module_name, score, table_name="module_scores"):
    """
    Record AI module output score to Supabase for AI console analytics.
    """
    try:
        record = {
            "module": module_name,
            "score": float(score),
            "timestamp": datetime.utcnow().isoformat()
        }
        supabase.table(table_name).insert(record).execute()
        print(f"✅ Recorded {module_name}: {score}")
        return True
    except Exception as e:
        print(f"❌ Score record failed: {str(e)}")
        return False

# ---------------------------------------------------------------
# 🧠 HELPER UTILITIES
# ---------------------------------------------------------------
def format_date(dt):
    if isinstance(dt, (datetime, pd.Timestamp)):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt)

def log_message(msg):
    """Print message (and optionally log to Supabase)."""
    print(msg)
    # Optional: supabase.table("logs").insert({"msg": msg, "time": datetime.utcnow().isoformat()}).execute()
    return msg
