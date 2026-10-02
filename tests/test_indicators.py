"""Tests for technical indicator functions (synthetic data, offline)."""

import numpy as np
import pandas as pd
import pytest

from src.data.indicators import bollinger_bands, ema, macd, rsi, sma


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    closes = 100 + np.cumsum(rng.normal(0, 1, 120))  # random walk around 100
    index = pd.bdate_range("2026-01-01", periods=120)
    return pd.DataFrame({"Close": closes}, index=index)


def test_sma_equals_rolling_mean(df):
    expected = df["Close"].rolling(20).mean()
    pd.testing.assert_series_equal(sma(df, 20), expected)


def test_ema_is_positive_and_aligned(df):
    result = ema(df, 20)
    assert len(result) == len(df)
    assert (result > 0).all()


def test_rsi_bounded_0_100(df):
    result = rsi(df).dropna()
    assert ((result >= 0) & (result <= 100)).all()


def test_macd_columns(df):
    result = macd(df)
    assert list(result.columns) == ["MACD", "Signal", "Histogram"]
    pd.testing.assert_series_equal(result["Histogram"], result["MACD"] - result["Signal"], check_names=False)


def test_bollinger_bands_ordering(df):
    result = bollinger_bands(df).dropna()
    assert (result["BB_Upper"] > result["BB_Middle"]).all()
    assert (result["BB_Middle"] > result["BB_Lower"]).all()
