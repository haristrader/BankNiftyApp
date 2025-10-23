import os
import pandas as pd
import requests
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta

# ---- CONFIG ----
CACHE_DIR = os.path.join(os.getcwd(), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
ALPHA_KEY = "demo86C2FRPD9UG1ZJUX"  # 🔑 Replace with your Alpha Vantage API key (free signup)

def _cache_path(symbol, tag):
    safe = symbol.replace(".", "_").replace("^", "")
    return os.path.join(CACHE_DIR, f"{safe}_{tag}.parquet")

def _cache_save(symbol, tag, df):
    path = _cache_path(symbol, tag)
    df.to_parquet(path, index=True)

def _cache_load(symbol, tag):
    path = _cache_path(symbol, tag)
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            if not df.empty:
                return df
        except Exception:
            pass
    return None

# ---- TIER 1: NSE LIVE (for 5m, 1m) ----
def fetch_nse_live(symbol="NSEBANK"):
    try:
        url = f"https://www.nseindia.com/api/chart-databyindex?index={symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()["grapthData"]
        df = pd.DataFrame(data, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        return df, "nse", "✅ Live data from NSE India"
    except Exception as e:
        return pd.DataFrame(), "nse", f"⚠ NSE API failed — {e}"

# ---- TIER 2: Alpha Vantage (historical) ----
def fetch_alpha(symbol="^NSEBANK", outputsize="full"):
    try:
        ts = TimeSeries(key=ALPHA_KEY, output_format="pandas")
        data, _ = ts.get_daily(symbol=symbol, outputsize=outputsize)
        data.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        }, inplace=True)
        _cache_save(symbol, "5y", data)
        return data, "alpha", "✅ 5-year data from Alpha Vantage"
    except Exception as e:
        return pd.DataFrame(), "alpha", f"⚠ Alpha failed — {e}"

# ---- TIER 3: Yahoo fallback ----
def fetch_yahoo(symbol="^NSEBANK", period="5y", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        _cache_save(symbol, f"{period}_{interval}", df)
        return df, "yahoo", "✅ Yahoo fallback data"
    except Exception as e:
        return pd.DataFrame(), "yahoo", f"⚠ Yahoo failed — {e}"

# ---- MASTER FETCH FUNCTION ----
def fetch_master(symbol="^NSEBANK", mode="auto"):
    # try cache
    cached = _cache_load(symbol, "latest")
    if cached is not None:
        return cached, "cache", "✅ Loaded from cache"

    # try live (NSE)
    df, src, msg = fetch_nse_live(symbol)
    if not df.empty:
        _cache_save(symbol, "latest", df)
        return df, src, msg

    # try alpha
    df, src, msg = fetch_alpha(symbol)
    if not df.empty:
        _cache_save(symbol, "latest", df)
        return df, src, msg

    # try yahoo
    df, src, msg = fetch_yahoo(symbol)
    if not df.empty:
        _cache_save(symbol, "latest", df)
        return df, src, msg

    # final fallback: old cache
    old = _cache_load(symbol, "latest")
    if old is not None:
        return old, "cache-old", "⚠ Using old cache data"

    return pd.DataFrame(), "none", "❌ All data sources failed"
