import requests
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from nsepython import nsefetch

print("=== TEST 1: NSE ===")
try:
    r = requests.get("https://www.nseindia.com/api/chart-databyindex?index=NIFTY BANK",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    print("✅ NSE working:", len(r.json().get("grapthData", [])), "records")
except Exception as e:
    print("❌ NSE Error:", e)

print("\n=== TEST 2: Alpha Vantage ===")
try:
    ts = TimeSeries(key="YOUR_KEY_HERE", output_format="pandas")
    data, _ = ts.get_daily(symbol="^NSEBANK", outputsize="compact")
    print("✅ Alpha working:", data.shape)
except Exception as e:
    print("❌ Alpha Error:", e)

print("\n=== TEST 3: Yahoo ===")
try:
    df = yf.download("^NSEBANK", period="1mo", interval="1d")
    print("✅ Yahoo working:", df.shape)
except Exception as e:
    print("❌ Yahoo Error:", e)
