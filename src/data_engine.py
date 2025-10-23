# src/data_engine.py
import pandas as pd
from typing import Tuple
from src.utils import fetch_smart as get_candles

def get_candles_safe(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Wrapper that ensures columns are strings and lower-case.
    Returns empty df on any problem.
    """
    try:
        df = get_candles(symbol=symbol, period=period, interval=interval)
        if df is None:
            return pd.DataFrame()
        # ensure columns are strings and lowercase
        try:
            df.columns = [str(c).lower() for c in df.columns]
        except Exception:
            # if MultiIndex sneaks in, flatten first
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [str(c[0]).lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
            else:
                df.columns = [str(c).lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"[data_engine.get_candles_safe] error: {e}")
        return pd.DataFrame()

# Backwards-compatible function name used in many pages
def get_candles(symbol: str, period: str, interval: str) -> pd.DataFrame:
    return get_candles_safe(symbol, period, interval)
