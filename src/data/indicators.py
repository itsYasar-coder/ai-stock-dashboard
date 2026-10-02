"""Technical indicators computed on OHLCV DataFrames.

All functions take a DataFrame with a "Close" column (as returned by
``StockDataFetcher.fetch_history``) and return a Series aligned to it.

TODO: add more indicators as the dashboard grows (ATR, stochastic,
VWAP, ADX).
"""

from __future__ import annotations

import pandas as pd


def sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Simple moving average of the closing price."""
    return df["Close"].rolling(window=window, min_periods=window).mean()


def ema(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Exponential moving average of the closing price."""
    return df["Close"].ewm(span=window, adjust=False).mean()


def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = df["Close"].diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram as a 3-column DataFrame."""
    macd_line = ema(df, fast) - ema(df, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "MACD": macd_line,
            "Signal": signal_line,
            "Histogram": macd_line - signal_line,
        }
    )


def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Upper / middle / lower Bollinger Bands."""
    middle = df["Close"].rolling(window=window, min_periods=window).mean()
    std = df["Close"].rolling(window=window, min_periods=window).std()
    return pd.DataFrame(
        {
            "BB_Middle": middle,
            "BB_Upper": middle + num_std * std,
            "BB_Lower": middle - num_std * std,
        }
    )


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with all indicator columns appended."""
    out = df.copy()
    out["SMA_20"] = sma(out, 20)
    out["SMA_50"] = sma(out, 50)
    out["EMA_20"] = ema(out, 20)
    out["RSI_14"] = rsi(out)
    out = out.join(macd(out))
    out = out.join(bollinger_bands(out))
    return out
