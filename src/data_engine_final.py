import pandas as pd
import datetime as dt
from nsetools import Nse
import investpy
import os

# Create cache folder if not exist
os.makedirs("cache", exist_ok=True)

def fetch_investpy(symbol="BANKNIFTY", from_date="01/01/2019", to_date=None):
    """Fetch 5-year historical data using investpy."""
    try:
        if to_date is None:
            to_date = dt.datetime.now().strftime("%d/%m/%Y")
        data = investpy.get_index_historical_data(
            index=symbol,
            country="india",
            from_date=from_date,
            to_date=to_date
        )
        data.to_csv(f"cache/{symbol}_daily.csv")
        return data, "InvestPy", "✅ Data fetched from InvestPy (5y)"
    except Exception as e:
        return pd.DataFrame(), "InvestPy", f"❌ InvestPy Error: {e}"

def fetch_nse_live(symbol="NIFTY BANK"):
    """Fetch live last price from NSE."""
    try:
        nse = Nse()
        q = nse.get_index_quote(symbol)
        df = pd.DataFrame([{
            "Date": pd.Timestamp.now(),
            "Close": q["lastPrice"]
        }])
        df.set_index("Date", inplace=True)
        return df, "NSE", "✅ Live price fetched from NSE"
    except Exception as e:
        return pd.DataFrame(), "NSE", f"❌ NSE Error: {e}"

def fetch_local_cache(symbol="BANKNIFTY"):
    """Fallback to local cache if API fails."""
    path = f"cache/{symbol}_daily.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return df, "LocalCache", "✅ Loaded cached data"
    return pd.DataFrame(), "LocalCache", "❌ No local cache found"

def fetch_master(symbol="BANKNIFTY", live=False):
    """Master function for data fetch"""
    if live:
        df, src, msg = fetch_nse_live("NIFTY BANK")
        if not df.empty:
            return df, src, msg

    df, src, msg = fetch_investpy(symbol)
    if not df.empty:
        return df, src, msg

    df, src, msg = fetch_local_cache(symbol)
    return df, src, msg
