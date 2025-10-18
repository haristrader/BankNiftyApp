# utils.py — Final, cloud-safe, weekend-safe (BankNifty only by default)
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

# -------- ATM Option helpers (delta-sim mode) --------
def _round_to_strike(x: float, step: int = 100) -> int:
    # BankNifty strikes: 100 step
    return int(round(x / step) * step)

def generate_signals_50pct(df: pd.DataFrame, mid_factor: float = 0.5) -> pd.DataFrame:
    """Create BUY/SELL/HOLD signals using previous-candle 50% rule."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    sig = ["HOLD"]
    for i in range(1, len(out)):
        ph, pl = float(out["high"].iloc[i-1]), float(out["low"].iloc[i-1])
        prev_mid = pl + (ph - pl) * mid_factor
        c = out.iloc[i]
        if (c["low"] >= prev_mid) and (c["close"] > ph):
            sig.append("BUY")
        elif (c["high"] <= prev_mid) and (c["close"] < pl):
            sig.append("SELL")
        else:
            sig.append("HOLD")
    out["signal"] = sig
    return out

def simulate_atm_option_trades(
    df: pd.DataFrame,
    signals_col: str = "signal",
    init_sl_pts: float = 10.0,
    lot_size: int = 15,               # BankNifty lot (update when exchange changes)
    mode: str = "delta",              # "delta" (ATM ~0.5) or "index" proxy
    delta_atm: float = 0.5,           # ATM delta approximation
    theta_per_candle: float = 0.0,    # simple time decay per candle (0 for off)
):
    """
    Simulate trades on ATM options using underlying candles:
    - On BUY → pick CE ATM strike (nearest 100) at entry candle
    - On SELL → pick PE ATM strike (nearest 100)
    - Price model:
        mode="delta":  premium change ≈ delta_atm * dUnderlying  - theta_per_candle
        mode="index":  premium change ≈ dUnderlying  (old index-proxy)
    - Trailing ladder (10→BE, 20→+10, 30→+15, 50→trail 50%)
    Returns: trades_df, equity_series (cum PnL), backtest_score(0-100 by winrate)
    """
    if df is None or df.empty or signals_col not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float), 50.0

    prices = df["close"].astype(float).values
    times = list(df.index)

    pos=None; entry_p=None; sl=None
    # For options we track premium. Start premium at 0 for change calc; add base at entry:
    # We'll build premium series incrementally from entry using delta approximation.
    # To keep realistic base, assume entry premium ≈ max(0, 0.01*underlying)  (≈1% of index)
    base_mult = 0.01

    trades=[]
    prem = 0.0

    for i in range(len(df)-1):
        sig = str(df[signals_col].iloc[i])
        u_now = float(prices[i])
        u_next = float(prices[i+1])
        du = u_next - u_now
        t_next = times[i+1]

        # compute option premium tick-to-tick change
        if mode == "delta":
            dp = delta_atm * du - theta_per_candle
        else:  # "index" proxy
            dp = du

        if pos is None:
            if sig == "BUY":
                pos="CE"                       # Long Call ATM
                entry_p = max(5.0, base_mult*u_next)  # entry premium baseline
                prem = entry_p
                sl = entry_p - init_sl_pts
            elif sig == "SELL":
                pos="PE"                       # Long Put ATM
                entry_p = max(5.0, base_mult*u_next)
                prem = entry_p
                sl = entry_p - init_sl_pts
            continue

        # update premium by model
        # For CE: +dp when du>0 ; For PE: invert sign
        if pos == "CE":
            prem = prem + max(-prem, dp)   # premium cannot go < 0
            profit = prem - entry_p
            # trailing ladder
            if profit >= 10 and sl < entry_p:          sl = entry_p
            if profit >= 20 and sl < entry_p + 10:     sl = entry_p + 10
            if profit >= 30 and sl < entry_p + 15:     sl = entry_p + 15
            if profit >= 50:
                new_sl = prem - (profit * 0.5)
                if new_sl > sl: sl = new_sl
            # exit on SL
            if prem <= sl:
                trades.append(dict(side="LONG CE", entry=entry_p, exit=sl, exit_time=t_next,
                                   pnl=(sl-entry_p)*lot_size))
                pos=None; entry_p=None; sl=None

        elif pos == "PE":
            prem = prem + max(-prem, -dp)  # put gains when underlying falls
            profit = prem - entry_p
            if profit >= 10 and sl < entry_p:          sl = entry_p
            if profit >= 20 and sl < entry_p + 10:     sl = entry_p + 10
            if profit >= 30 and sl < entry_p + 15:     sl = entry_p + 15
            if profit >= 50:
                new_sl = prem - (profit * 0.5)
                if new_sl > sl: sl = new_sl
            if prem <= sl:
                trades.append(dict(side="LONG PE", entry=entry_p, exit=sl, exit_time=t_next,
                                   pnl=(sl-entry_p)*lot_size))
                pos=None; entry_p=None; sl=None

    tr = pd.DataFrame(trades)
    if tr.empty:
        return tr, pd.Series(dtype=float), 50.0

    tr["pnl_points"] = tr["pnl"]        # already in option points * lot
    tr["cum_pnl"] = tr["pnl_points"].cumsum()
    winrate = float((tr["pnl_points"] > 0).mean() * 100.0)
    score = float(np.clip(winrate, 0, 100))
    return tr, tr["cum_pnl"], score
# ------------------ Defaults & Weights ------------------
DEFAULT_SYMBOL = "^NSEBANK"

WEIGHTS_DEFAULT = {
    "trend": 20,
    "fibonacci": 25,
    "priceaction": 15,
    "smartmoney": 20,
    "backtest": 10,
    "others": 10,
}

# Yahoo intraday limits are strict; use these safe fallbacks (BankNifty only)
# Order matters: we try narrow intraday first, then broaden to daily/weekly.
INTERVAL_FALLBACKS: list[tuple[str, str]] = [
    ("5d",  "5m"),   # allowed intraday combo
    ("14d", "15m"),
    ("30d", "30m"),
    ("60d", "60m"),
    ("3mo", "1d"),   # market closed or intraday blocked
    ("2y",  "1wk"),  # ultimate fallback for plotting/scores
]

# ------------------ Core Fetchers ------------------
def fetch(
    symbol: str = DEFAULT_SYMBOL,
    period: str = "14d",
    interval: str = "5m",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV via yfinance and return a cleaned DataFrame with columns:
    ['open','high','low','close','volume'] and DateTimeIndex (tz-naive).
    Robust to MultiIndex/duplicate columns.
    NOTE: This tries ONLY the provided (period, interval). For smart cascading,
    use fetch_smart(...).
    """
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=auto_adjust,
            group_by="column",  # avoid MultiIndex
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex if any (paranoia)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[-1]) if isinstance(col, tuple) else str(col) for col in df.columns]

    # Standardize names
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
    })

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if not keep:
        return pd.DataFrame()
    df = df[keep].copy()

    # tz-naive, sorted
    try:
        df.index = pd.to_datetime(df.index).tz_convert(None)
    except Exception:
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return sanitize_ohlcv(df)


def fetch_smart(
    symbol: str = DEFAULT_SYMBOL,
    prefer: tuple[str, str] | None = None,
    auto_adjust: bool = True,
    allow_symbol_fallback: bool = False,   # you earlier said NO fallback; keep False
) -> tuple[pd.DataFrame, tuple[str, str]]:
    """
    Try (period, interval) pairs until data arrives.
    Returns (df, (used_period, used_interval)).

    If allow_symbol_fallback=True, the function will also try 'NSEBANK.NS'
    in daily/weekly as a last resort (off by default per your preference).
    """
    tries: list[tuple[str, str]] = []
    if prefer and isinstance(prefer, tuple) and len(prefer) == 2:
        tries.append(prefer)
    for t in INTERVAL_FALLBACKS:
        if t not in tries:
            tries.append(t)

    # First pass: try requested symbol only
    for per, iv in tries:
        df = fetch(symbol, per, iv, auto_adjust=auto_adjust)
        if df is not None and not df.empty:
            return df, (per, iv)

    # Optional second pass: try symbol fallback (disabled by default)
    if allow_symbol_fallback:
        backup_symbol = "NSEBANK.NS"
        for per, iv in [("3mo", "1d"), ("2y", "1wk")]:
            df = fetch(backup_symbol, per, iv, auto_adjust=auto_adjust)
            if df is not None and not df.empty:
                return df, (per, iv)

    # Nothing worked
    return pd.DataFrame(), (tries[0][0] if tries else "5d", tries[0][1] if tries else "5m")

# ------------------ Sanitizer ------------------
def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe OHLCV sanitizer:
    - Flattens MultiIndex columns if any
    - Removes duplicate columns
    - Ensures each target column is 1-D Series, coerces to numeric
    - Drops incomplete rows
    - Ensures tz-naive DateTimeIndex named 'Date'
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[-1]) if isinstance(col, tuple) else str(col) for col in out.columns]

    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]

    cleaned = pd.DataFrame(index=out.index)
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            continue
        col_obj = out[col]
        if isinstance(col_obj, pd.DataFrame):     # if somehow duplicate-named columns
            col_obj = col_obj.iloc[:, 0]
        cleaned[col] = pd.to_numeric(col_obj, errors="coerce")

    cleaned = cleaned.dropna(subset=["open", "high", "low", "close", "volume"])

    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
        cleaned = cleaned.dropna().sort_index()

    cleaned.index.name = "Date"
    return cleaned

# ------------------ Fibonacci Helpers ------------------
def swing(df: pd.DataFrame, lookback: int = 200):
    recent = df.tail(max(lookback, 2))
    sh = float(recent["high"].max())
    sl = float(recent["low"].min())
    last_close = float(df["close"].iloc[-1])
    mid = (sh + sl) / 2
    direction = "down" if last_close > mid else "up"
    return sh, sl, direction

def fib_levels(high: float, low: float, direction: str = "up") -> dict:
    d = high - low if high != low else 1e-9
    if direction == "up":
        return {
            "0%": high, "23.6%": high - 0.236 * d, "38.2%": high - 0.382 * d,
            "50%": high - 0.5 * d, "61.8%": high - 0.618 * d,
            "78.6%": high - 0.786 * d, "100%": low
        }
    else:
        return {
            "0%": low, "23.6%": low + 0.236 * d, "38.2%": low + 0.382 * d,
            "50%": low + 0.5 * d, "61.8%": low + 0.618 * d,
            "78.6%": low + 0.786 * d, "100%": high
        }

def fib_confidence(df: pd.DataFrame, lookback: int = 200) -> tuple[float, dict]:
    sh, sl, dirn = swing(df, lookback)
    lv = fib_levels(sh, sl, "up" if dirn == "up" else "down")
    close = float(df["close"].iloc[-1])
    targets = [lv["38.2%"], lv["50%"], lv["61.8%"]]
    dist = min(abs(close - t) for t in targets)
    rng = max(1.0, (sh - sl) / 6.0)
    score = max(0.0, 100.0 - (dist / rng) * 100.0)
    return round(score, 2), lv
