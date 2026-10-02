"""Environment-driven settings and app-wide constants."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root = directory containing this package's parent (repo root).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load .env from the project root (no-op if the file does not exist).
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _streamlit_secret(key: str) -> str:
    """Read a value from Streamlit secrets (Community Cloud "Secrets" UI).

    Returns "" when Streamlit secrets aren't available (CLI/tests/local .env
    users), so callers can fall back to environment variables.
    """
    try:
        import streamlit as st

        value = st.secrets[key]
        return str(value) if value is not None else ""
    except Exception:  # noqa: BLE001 - no secrets file / bare mode → treat as unset
        return ""


@dataclass(frozen=True)
class Settings:
    """Central app configuration.

    Values resolve in order: environment variables (local `.env` via
    python-dotenv) → Streamlit secrets (Cloud "Secrets" settings).
    """

    # AI insights
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY") or _streamlit_secret("OPENAI_API_KEY")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.environ.get("OPENAI_BASE_URL") or _streamlit_secret("OPENAI_BASE_URL")
    )
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))

    # App
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # Data
    default_period: str = "1y"          # yfinance history window
    default_interval: str = "1d"        # yfinance bar interval
    exchange_suffix: str = ".NS"        # NSE by default; use ".BO" for BSE


SETTINGS = Settings()

# Default universe for the dashboard's symbol picker (large-cap NIFTY names).
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "SBIN", "BHARTIARTL", "ITC", "LT", "HINDUNILVR",
    "BAJFINANCE", "MARUTI", "SUNPHARMA", "TATAMOTORS", "WIPRO",
)

# Universe scanned by the AI recommender (src/ai/recommender.py):
# the 30 largest NIFTY 50 constituents by market cap.
TOP_30_SYMBOLS: tuple[str, ...] = (
    "RELIANCE", "HDFCBANK", "TCS", "ICICIBANK", "BHARTIARTL",
    "INFY", "SBIN", "LT", "ITC", "HINDUNILVR",
    "BAJFINANCE", "HCLTECH", "MARUTI", "KOTAKBANK", "SUNPHARMA",
    "M&M", "TITAN", "ULTRACEMCO", "BAJAJ-AUTO", "ADANIENT",
    "ADANIPORTS", "ASIANPAINT", "POWERGRID", "COALINDIA", "BAJAJFINSV",
    "WIPRO", "NTPC", "JSWSTEEL", "NESTLEIND", "TATASTEEL",
)

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
