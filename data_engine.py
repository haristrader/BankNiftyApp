import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import save_candles_supabase  # compatible alias for sb_save_candles

# ---------------------------------------------------------------
# 🧱 CONFIG
# ---------------------------------------------------------------
DEFAULT_SYMBOL = "^NSEBANK"  # works reliably with Yahoo Finance
DEFAULT_PERIOD = "5d"
DEFAULT_INTERVAL = "5m"


# ---------------------------------------------------------------
# 🔹 CORE FUNCTION: FETCH & NORMALIZE
# ---------------------------------------------------------------

def get_candles(symbol=DEFAULT_SYMBOL, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    """
    Fetch OHLCV data safely from Yahoo Finance and normalize columns.
    Handles MultiIndex issue and converts to lowercase column names.
    """
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)

        # ---- FIX: Handle MultiIndex safely ----
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                c[0].lower() if isinstance(c, tuple) else str(c).lower()
                for c in df.columns
            ]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        # Reset and clean
        df.reset_index(inplace=True)
        df.rename(columns={"datetime": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.dropna(inplace=True)

        # Ensure required columns
        required_cols = {"open", "high", "low", "close", "volume"}
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan

        return df

    except Exception as e:
        print(f"❌ Error fetching candles: {str(e)}")
        return pd.DataFrame()


# ---------------------------------------------------------------
# 🔹 FETCHER FOR SMART MODULES (TREND / FIB / PRICEACTION)
# ---------------------------------------------------------------

def fetch_smart(symbol=DEFAULT_SYMBOL, period="5d", interval="5m", prefer="auto"):
    """
    Smart fetch layer for all modules.
    prefer = 'auto' | 'supabase' | 'live'
    """
    df = pd.DataFrame()
    msg = ""

    try:
        df = get_candles(symbol, period, interval)
        if not df.empty:
            msg = f"✅ {symbol}: {len(df)} rows loaded ({interval}, {period})"
        else:
            msg = f"⚠️ {symbol}: No data available for {interval} / {period}"

    except Exception as e:
        msg = f"❌ Fetch error: {str(e)}"

    return df, msg


# ---------------------------------------------------------------
# 🔹 FEATURE: BASIC INDICATORS (EMA, RSI)
# ---------------------------------------------------------------

def compute_indicators(df):
    """
    Compute EMA and RSI for any DataFrame (used by Trend, PriceAction, etc.)
    """
    if df.empty:
        return df

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


# ---------------------------------------------------------------
# 🔹 FEATURE: SUPABASE SAVE WRAPPER
# ---------------------------------------------------------------

def upload_to_supabase(df, table="banknifty_candles"):
    """
    Optional upload helper for saving candles to Supabase.
    """
    if df.empty:
        return "⚠️ No data to upload."
    return save_candles_supabase(df, table_name=table)


# ---------------------------------------------------------------
# 🔹 FEATURE: PRICE ACTION SIGNAL (Mid-Zone Strategy)
# ---------------------------------------------------------------

def generate_signals(df):
    """
    Simple price-action strategy signal generator.
    1 = Buy signal, -1 = Sell signal
    """
    if df.empty:
        return df

    df["mid_zone"] = (df["high"] + df["low"]) / 2
    df["signal"] = np.where(df["close"] > df["mid_zone"], 1, -1)
    return df


# ---------------------------------------------------------------
# 🔹 FEATURE: TREND STRENGTH (EMA-RSI COMBO)
# ---------------------------------------------------------------

def trend_strength(df):
    """
    Combine EMA and RSI into a single trend strength metric.
    """
    if df.empty or "ema20" not in df or "rsi" not in df:
        return 0

    ema_score = np.where(df["ema20"].iloc[-1] > df["ema50"].iloc[-1], 1, -1)
    rsi_score = 1 if df["rsi"].iloc[-1] > 50 else -1

    return round((ema_score + rsi_score) / 2, 2)


# ---------------------------------------------------------------
# 🔹 FEATURE: AUTO PAPER-TRADE SIMULATION ENGINE (Virtual Capital)
# ---------------------------------------------------------------

def simulate_paper_trade(df, initial_capital=10000, lot_size=15):
    """
    Simulate virtual trading using simple signals.
    Shows how much capital would change on signal-based trades.
    """
    if df.empty or "signal" not in df:
        return initial_capital, []

    capital = initial_capital
    trade_log = []

    last_signal = 0
    entry_price = 0

    for i, row in df.iterrows():
        if row["signal"] != last_signal:
            # Close previous position
            if last_signal != 0 and entry_price > 0:
                pnl = (row["close"] - entry_price) * lot_size * last_signal
                capital += pnl
                trade_log.append({
                    "date": row["date"],
                    "action": "EXIT",
                    "price": row["close"],
                    "pnl": pnl,
                    "balance": capital
                })
            # Open new position
            last_signal = row["signal"]
            entry_price = row["close"]
            trade_log.append({
                "date": row["date"],
                "action": "ENTRY",
                "signal": last_signal,
                "price": row["close"],
                "balance": capital
            })

    return capital, trade_log
