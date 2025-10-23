from smartapi import SmartConnect
import pandas as pd
from datetime import datetime, timedelta

api_key = "hTogHfod"
client_code = "YOUR_CLIENT_CODE"
password = "YOUR_PASSWORD"
totp = "YOUR_TOTP"

obj = SmartConnect(api_key=api_key)
data = obj.generateSession(client_code, password, totp)

token = data['data']['jwtToken']
feedToken = obj.getfeedToken()

# BANKNIFTY SYMBOL DETAILS
symbol_token = "99926009"   # BankNifty Spot token
exchange = "NSE"

# TIME RANGE
to_date = datetime.now()
from_date = to_date - timedelta(days=5)

# FETCH HISTORICAL DATA
historic_data = obj.getCandleData({
    "exchange": exchange,
    "symboltoken": symbol_token,
    "interval": "FIVE_MINUTE",
    "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
    "todate": to_date.strftime("%Y-%m-%d %H:%M")
})

df = pd.DataFrame(historic_data['data'])
print(df.tail())
