"""Tests for the AI stock recommender (fully stubbed — no network, no real LLM)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.ai.llm_client import LLMClientError
from src.ai.recommender import RecommendationError, StockRecommender


# --------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------- #

def make_ohlcv(base: float = 100.0, drift: float = 0.0, seed: int = 0, periods: int = 120) -> pd.DataFrame:
    """Deterministic synthetic OHLCV frame (random walk with drift)."""
    rng = np.random.default_rng(seed)
    closes = base * np.cumprod(1 + rng.normal(drift, 0.01, periods))
    index = pd.bdate_range("2026-01-01", periods=periods)
    return pd.DataFrame(
        {
            "Open": closes * 0.999,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": rng.integers(100_000, 1_000_000, periods).astype(float),
        },
        index=index,
    )


class FakeLLM:
    """Stands in for LLMClient: canned response or exception."""

    def __init__(self, response=None, configured: bool = True):
        self.response = response
        self.is_configured = configured
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeFetcher:
    """Stands in for StockDataFetcher: serves prebuilt frames, no network."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def fetch_history(self, symbol: str, period: str | None = None) -> pd.DataFrame:
        return self.frames[symbol]


CANNED_RESPONSE = json.dumps(
    {
        "market_outlook": "Banking names show constructive momentum.",
        "top_5_picks": [
            # Mixed types on purpose: string buy price with thousands comma,
            # int/float/string targets — all must be coerced.
            {"trade_type": "Intraday", "stock_name": "reliance", "buy_price": "2,950.50",
             "targets": [3000, 3050.25, "3100"], "sell_price": 2900},
            {"trade_type": "Longterm", "stock_name": "TCS", "buy_price": 4100,
             "targets": ["4200.00", "4300.00", "4400.00"], "sell_price": 3950},
            {"trade_type": "Intraday", "stock_name": "INFY", "buy_price": 1600,
             "targets": ["1610.00", "1620.00"], "sell_price": 1580},
            {"stock_name": "SBIN", "buy_price": 800, "targets": [], "sell_price": 770},
            {"trade_type": "Longterm", "stock_name": "ITC", "buy_price": 450,
             "targets": ["460.00", "470.00", "480.00"], "sell_price": 430},
            # Sixth pick — must be truncated away.
            {"trade_type": "Longterm", "stock_name": "WIPRO", "buy_price": 500,
             "targets": ["510.00"], "sell_price": 480},
        ],
    }
)


def make_frames(count: int, seed_offset: int = 100) -> dict[str, pd.DataFrame]:
    return {
        f"STOCK{i}": make_ohlcv(base=100 + i * 10, drift=0.002, seed=seed_offset + i)
        for i in range(count)
    }


# --------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------- #

def test_extract_json_direct():
    payload = StockRecommender._extract_json('{"a": 1}')
    assert payload == {"a": 1}


def test_extract_json_from_markdown_fence():
    raw = "Here are the picks:\n```json\n{\"market_outlook\": \"x\", \"top_5_picks\": []}\n```\nEnjoy!"
    assert StockRecommender._extract_json(raw)["market_outlook"] == "x"


def test_extract_json_embedded_in_prose():
    raw = 'Sure! {"a": {"b": 2}} hope that helps'
    assert StockRecommender._extract_json(raw) == {"a": {"b": 2}}


def test_extract_json_invalid_raises():
    with pytest.raises(RecommendationError):
        StockRecommender._extract_json("not json at all")


# --------------------------------------------------------------------- #
# Validation / coercion
# --------------------------------------------------------------------- #

def test_parse_coerces_types_and_truncates_to_five():
    result = StockRecommender(FakeLLM())._parse_recommendations(CANNED_RESPONSE)

    assert result["market_outlook"] == "Banking names show constructive momentum."
    assert len(result["top_5_picks"]) == 5  # sixth pick dropped

    first = result["top_5_picks"][0]
    assert first["stock_name"] == "RELIANCE"  # uppercased
    assert first["buy_price"] == 2950.50  # "2,950.50" -> float
    assert first["targets"] == ["3000.00", "3050.25", "3100.00"]  # all -> 2-decimal strings
    assert first["sell_price"] == 2900.0

    infy = next(p for p in result["top_5_picks"] if p["stock_name"] == "INFY")
    assert infy["targets"] == ["1610.00", "1620.00", "1640.00"]  # padded: last step (20) extrapolated

    sbin = next(p for p in result["top_5_picks"] if p["stock_name"] == "SBIN")
    assert sbin["trade_type"] == "Longterm"  # default filled
    assert sbin["targets"] == ["816.00", "832.00", "848.00"]  # derived: long (stop < buy)


def test_derived_targets_for_short_side():
    targets = StockRecommender._coerce_targets(None, buy_price=100.0, stop_price=105.0)
    assert targets == ["98.00", "96.00", "94.00"]  # short: objectives below entry


def test_validate_rejects_non_dict_payload():
    with pytest.raises(RecommendationError):
        StockRecommender(FakeLLM())._validate_recommendations(["not", "a", "dict"])


def test_validate_rejects_payload_without_valid_picks():
    with pytest.raises(RecommendationError):
        StockRecommender(FakeLLM())._validate_recommendations(
            {"market_outlook": "meh", "top_5_picks": [{"stock_name": "X", "buy_price": "abc"}]}
        )


# --------------------------------------------------------------------- #
# Mover ranking
# --------------------------------------------------------------------- #

def test_rank_movers_orders_by_absolute_change():
    frames = {
        "FLAT1": make_ohlcv(base=100, seed=1),
        "SPIKE_UP": make_ohlcv(base=100, seed=2),
        "SPIKE_DOWN": make_ohlcv(base=100, seed=3),
        "FLAT2": make_ohlcv(base=100, seed=4),
    }
    # Pin the last close to a fixed multiple of the previous close so the
    # daily % change is exact (+7% / -6%) regardless of the random walk.
    for symbol, factor in (("SPIKE_UP", 1.07), ("SPIKE_DOWN", 0.94)):
        frame = frames[symbol]
        frame.loc[frame.index[-1], "Close"] = frame["Close"].iloc[-2] * factor

    recommender = StockRecommender(FakeLLM(), fetcher=FakeFetcher(frames), movers=3)
    movers = recommender._rank_movers(frames)

    assert [m["symbol"] for m in movers][:2] == ["SPIKE_UP", "SPIKE_DOWN"]
    assert len(movers) == 3  # limited by `movers`
    assert all("enriched" in m and "RSI_14" in m["enriched"] for m in movers)


# --------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------- #

def test_full_flow_with_llm():
    frames = make_frames(6, seed_offset=200)
    llm = FakeLLM(response=CANNED_RESPONSE)
    recommender = StockRecommender(llm, fetcher=FakeFetcher(frames), universe=list(frames))
    result = recommender.get_top_recommendations()

    assert result["market_outlook"] == "Banking names show constructive momentum."
    assert len(result["top_5_picks"]) == 5
    assert all(isinstance(p["buy_price"], float) for p in result["top_5_picks"])
    assert all(len(p["targets"]) == 3 for p in result["top_5_picks"])

    system_prompt, user_prompt = llm.calls[0]
    assert "JSON" in system_prompt
    assert all(f"STOCK{i}" in user_prompt for i in range(6))  # every mover briefed


def test_fallback_without_llm_key():
    frames = make_frames(8)
    recommender = StockRecommender(FakeLLM(configured=False), fetcher=FakeFetcher(frames), universe=list(frames))
    result = recommender.get_top_recommendations()

    assert len(result["top_5_picks"]) == 5
    assert "top_5_picks" in result and "market_outlook" in result
    for pick in result["top_5_picks"]:
        assert pick["trade_type"] in {"Intraday", "Longterm"}
        assert isinstance(pick["buy_price"], float)
        assert len(pick["targets"]) == 3
        assert pick["sell_price"] < pick["buy_price"]  # long-biased heuristic


def test_llm_failure_falls_back_to_heuristic():
    frames = make_frames(6)
    llm = FakeLLM(response=LLMClientError("API down"))
    recommender = StockRecommender(llm, fetcher=FakeFetcher(frames), universe=list(frames))
    result = recommender.get_top_recommendations()

    assert len(result["top_5_picks"]) == 5


def test_unparseable_llm_response_falls_back():
    frames = make_frames(6)
    llm = FakeLLM(response="My apologies, I cannot comply.")
    recommender = StockRecommender(llm, fetcher=FakeFetcher(frames), universe=list(frames))
    result = recommender.get_top_recommendations()

    assert len(result["top_5_picks"]) == 5
    assert "Heuristic scan" in result["market_outlook"]


def test_insufficient_universe_raises():
    frames = {"ONLY_ONE": make_ohlcv()}
    with pytest.raises(RecommendationError):
        StockRecommender(
            FakeLLM(), fetcher=FakeFetcher(frames), universe=list(frames)
        ).get_top_recommendations()
