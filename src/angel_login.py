from smartapi import SmartConnect
import pyotp

# === ANGEL ONE CREDENTIALS ===
API_KEY = "hTogHfod"
CLIENT_CODE = "H15001"
PASSWORD = "3195"  # apna Angel password yahan daal
TOTP_SECRET = "LT6DRMTCL4ED4MOYWFT5ZAKQXE"

# === LOGIN PROCESS ===
def get_angel_session():
    try:
        totp = pyotp.TOTP(TOTP_SECRET).now()
        obj = SmartConnect(api_key=API_KEY)
        data = obj.generateSession(CLIENT_CODE, PASSWORD, totp)

        feedToken = obj.getfeedToken()
        refreshToken = data['data']['refreshToken']

        print("✅ Login Successful")
        print("Feed Token:", feedToken)
        return obj, feedToken, refreshToken
    except Exception as e:
        print("❌ Login Failed:", e)
        return None, None, None

if __name__ == "__main__":
    get_angel_session()
