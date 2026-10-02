"""Tests for the data fetching helpers (offline — no network calls)."""

import pandas as pd

from src.data.fetcher import StockDataFetcher


def test_plain_symbol_gets_nse_suffix():
    assert StockDataFetcher.to_yahoo_symbol("reliance") == "RELIANCE.NS"


def test_symbols_with_suffix_are_unchanged():
    assert StockDataFetcher.to_yahoo_symbol("RELIANCE.NS") == "RELIANCE.NS"
    assert StockDataFetcher.to_yahoo_symbol("itc.BO") == "ITC.BO"


def test_symbol_is_uppercased_and_stripped():
    assert StockDataFetcher.to_yahoo_symbol("  tcs ") == "TCS.NS"


def test_fetch_history_flattens_multiindex_columns(monkeypatch):
    """Regression: current yfinance returns (Price, Ticker) columns even for
    one symbol; fetch_history must flatten them so df["Close"] is a Series."""
    import src.data.fetcher as fetcher_module

    index = pd.bdate_range("2026-01-01", periods=5)
    tuples = [(field, "RELIANCE.NS") for field in ("Open", "High", "Low", "Close", "Volume")]
    multi = pd.DataFrame(
        [[101.0, 103.0, 100.0, 102.0, 1_000.0] for _ in range(5)],
        index=index,
        columns=pd.MultiIndex.from_tuples(tuples),
    )
    monkeypatch.setattr(
        fetcher_module.yf, "download", lambda *args, **kwargs: multi
    )

    df = StockDataFetcher().fetch_history("RELIANCE")

    assert not isinstance(df.columns, pd.MultiIndex)
    assert isinstance(df["Close"], pd.Series)
    assert float(df["Close"].iloc[-1]) == 102.0
