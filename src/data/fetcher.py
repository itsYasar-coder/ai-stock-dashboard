"""Market data fetching for NSE/BSE symbols via yfinance.

TODO: add caching (disk or st.cache_data), retry/backoff on network errors,
and an alternative provider (e.g. NSE API) as a fallback source.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from config.settings import SETTINGS

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetches historical and live market data for Indian stocks."""

    def __init__(self, period: str = SETTINGS.default_period, interval: str = SETTINGS.default_interval):
        self.period = period
        self.interval = interval

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fetch_history(self, symbol: str, period: str | None = None) -> pd.DataFrame:
        """Download OHLCV history for a symbol.

        Args:
            symbol: Plain NSE/BSE ticker, e.g. "RELIANCE", "RELIANCE.BO".
            period: Override the instance default (e.g. "6mo", "1y", "5y").

        Returns:
            DataFrame indexed by Date with columns
            [Open, High, Low, Close, Volume].
        """
        ticker_symbol = self.to_yahoo_symbol(symbol)
        logger.info("Fetching history for %s (%s)", symbol, ticker_symbol)
        df = yf.download(
            ticker_symbol,
            period=period or self.period,
            interval=self.interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"No data returned for symbol '{symbol}' ({ticker_symbol})")
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance returns (Price, Ticker) columns even for a single
            # symbol; flatten so df["Close"] is a Series downstream.
            df.columns = df.columns.get_level_values(0)
        return df

    def fetch_quote(self, symbol: str) -> dict[str, Any]:
        """Return a live-ish quote snapshot (price, change, day range)."""
        ticker = yf.Ticker(self.to_yahoo_symbol(symbol))
        info: dict[str, Any] = {}
        # TODO: narrow the fields once we know which ones we display;
        # .fast_info is cheaper than .info for price data.
        fast = ticker.fast_info
        info["last_price"] = fast.get("last_price") if hasattr(fast, "get") else getattr(fast, "last_price", None)
        info["currency"] = "INR"
        return info

    def save_history(self, symbol: str, df: pd.DataFrame) -> None:
        """Persist downloaded history to data/raw as CSV for offline reuse."""
        from config.settings import RAW_DATA_DIR

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DATA_DIR / f"{symbol.replace('.', '_')}_{self.period}_{self.interval}.csv"
        df.to_csv(path)
        logger.info("Saved %s rows for %s to %s", len(df), symbol, path)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_yahoo_symbol(symbol: str, suffix: str = SETTINGS.exchange_suffix) -> str:
        """Map a plain ticker to a Yahoo Finance symbol.

        "RELIANCE" -> "RELIANCE.NS"; symbols that already carry an exchange
        suffix (".NS" / ".BO") are returned unchanged.
        """
        symbol = symbol.strip().upper()
        if "." in symbol:
            return symbol
        return f"{symbol}{suffix}"
