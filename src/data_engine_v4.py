import os
import pandas as pd
import requests
from alpha_vantage.timeseries import TimeSeries
from nsepython import nsefetch
from datetime import datetime, timedelta

# ====== CONFIG ======
CACHE_DIR = os.path.join(os.getcwd(), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

ALPHA_KEY = "8S9QC24XEGWNQNII"  # 🔑 Replace with your actual Alpha Vantage key


# ====== UTILITIES ======
def _cache_path(symbol, tag):
    safe = symbol.replace(".", "_").replace("^", "")
    return os.path.join(CACHE_DIR, f"{safe}_{tag}.parquet")

def _cache_save(symbol, tag, df):
    try:
        df.to_parquet(_cache_path(symbol, tag), index=True)
    except Exception as e:
        print("Cache save error:", e)

def _cache_load(symbol, tag):
    path = _cache_path(symbol, tag)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except:
            pass
    return None


# ====== 1️⃣ LIVE DATA (NSE India official) ======
def fetch_nse_live(symbol="NIFTY BANK"):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com"
        }

        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        if symbol.upper() == "NIFTY BANK":
            url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY BANK"
        else:
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"

        response = session.get(url, headers=headers, timeout=10)
        data = response.json()

        if "data" in data and len(data["data"]) > 0:
            df = pd.DataFrame(data["data"])
            if "lastPrice" in df.columns:
                df["Close"] = pd.to_numeric(df["lastPrice"].str.replace(",", ""), errors="coerce")
                df["timestamp"] = pd.Timestamp.now()
                df.set_index("timestamp", inplace=True)
                return df, "NSE", "✅ Live price fetched via alternate NSE endpoint"

        raise Exception("Empty response structure")

    except Exception as e:
        return pd.DataFrame(), "NSE", f"⚠ Alternate NSE live fetch failed: {e}"


# ====== 2️⃣ HISTORICAL (Alpha Vantage – 5 years) ======
def fetch_alpha(symbol="^NSEBANK"):
    try:
        ts = TimeSeries(key=ALPHA_KEY, output_format="pandas")
        data, _ = ts.get_daily(symbol=symbol, outputsize="full")
        data.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        }, inplace=True)
        _cache_save(symbol, "5y", data)
        return data, "Alpha", "✅ Historical (5y) data from Alpha Vantage"
    except Exception as e:
        return pd.DataFrame(), "Alpha", f"⚠ Alpha failed: {e}"


# ====== 3️⃣ NSE Option Chain Backup (for intraday if needed) ======
def fetch_nse_option(symbol="BANKNIFTY"):
    try:
        data = nsefetch(f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}")
        df = pd.json_normalize(data["records"]["data"])
        return df, "NSE-Option", "✅ Option data fetched (NSE)"
    except Exception as e:
        return pd.DataFrame(), "NSE-Option", f"⚠ Option chain failed: {e}"


# ====== MASTER FETCH ======
def fetch_master(symbol="NIFTY BANK", mode="auto"):
    # Cache check first
    cached = _cache_load(symbol, "latest")
    if cached is not None:
        return cached, "Cache", "✅ Loaded from cache"

    # Live data
    df, src, msg = fetch_nse_live(symbol)
    if not df.empty:
        _cache_save(symbol, "latest", df)
        return df, src, msg

    # Alpha fallback (historical)
    df, src, msg = fetch_alpha("^NSEBANK")
    if not df.empty:
        _cache_save(symbol, "latest", df)
        return df, src, msg

    # NSE option fallback
    df, src, msg = fetch_nse_option()
    if not df.empty:
        _cache_save(symbol, "latest", df)
        return df, src, msg

    # Final fallback: local Yahoo cache
try:
    path = "cache/banknifty_daily_5y.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        return df, "LocalCache", "✅ Loaded cached 5-year daily BankNifty data"
except Exception as e:
    print("Cache fallback failed:", e)

