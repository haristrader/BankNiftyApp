import yfinance as yf
import pandas as pd
import os

# cache folder bana lo agar nahi hai
os.makedirs("cache", exist_ok=True)

print("⏳ Downloading 5-year daily BankNifty data...")
df = yf.download("^NSEBANK", period="5y", interval="1d", progress=False)

if df.empty:
    print("❌ Yahoo returned empty data.")
else:
    df.to_parquet("cache/banknifty_daily_5y.parquet")
    print("✅ 5-year daily BankNifty cached successfully.")
    print("File saved as: cache/banknifty_daily_5y.parquet")
