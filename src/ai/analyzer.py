"""AI stock analysis: turns price/indicator data into natural-language insights.

TODO: add a structured output mode (JSON) so the dashboard can render
signal chips (bullish/bearish/neutral) instead of free text only.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.ai.llm_client import LLMClient, LLMClientError
from src.data.indicators import add_all

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an equity research assistant for the Indian stock market (NSE/BSE). "
    "You receive recent price action and technical indicator values and respond "
    "with a concise, balanced analysis in plain English. Cover trend, momentum, "
    "volatility and key levels. Never give personalised investment advice; "
    "include a one-line reminder that this is not investment advice."
)


class StockAnalyzer:
    """Produces AI (or heuristic fallback) insights for a single stock."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def analyze(self, symbol: str, df: pd.DataFrame) -> str:
        """Return a natural-language analysis of ``df`` for ``symbol``.

        Falls back to a deterministic rule-based summary when no LLM is
        configured, so the dashboard stays functional without an API key.
        """
        enriched = add_all(df)
        summary = self._build_data_summary(enriched)

        if self.llm.is_configured:
            try:
                return self.llm.generate(SYSTEM_PROMPT, f"Stock: {symbol}\n\n{summary}")
            except LLMClientError as exc:
                logger.warning("AI insights unavailable (%s); using fallback", exc)

        return self._fallback_summary(symbol, enriched)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_data_summary(enriched: pd.DataFrame) -> str:
        """Condense the indicator frame into a compact text briefing."""
        latest = enriched.dropna().tail(1)
        if latest.empty:
            return "Not enough data to compute indicators."
        row = latest.iloc[-1]
        change_pct = enriched["Close"].pct_change().tail(30).mean() * 100
        return (
            f"Latest close: {row['Close']:.2f} INR\n"
            f"30-day average daily change: {change_pct:+.2f}%\n"
            f"RSI(14): {row.get('RSI_14', float('nan')):.1f}\n"
            f"SMA20: {row.get('SMA_20', float('nan')):.2f}, "
            f"SMA50: {row.get('SMA_50', float('nan')):.2f}\n"
            f"MACD: {row.get('MACD', float('nan')):.3f} vs Signal: {row.get('Signal', float('nan')):.3f}\n"
            f"Bollinger bands: lower {row.get('BB_Lower', float('nan')):.2f} / "
            f"upper {row.get('BB_Upper', float('nan')):.2f}\n"
            f"Data window: {enriched.index[0].date()} to {enriched.index[-1].date()} "
            f"({len(enriched)} sessions)"
        )

    @staticmethod
    def _fallback_summary(symbol: str, enriched: pd.DataFrame) -> str:
        """Rule-based summary used when the LLM is not configured."""
        row = enriched.dropna().tail(1).iloc[0]
        trend = "uptrend" if row["Close"] > row["SMA_50"] else "downtrend"
        rsi_value = float(row["RSI_14"])
        momentum = "overbought" if rsi_value >= 70 else "oversold" if rsi_value <= 30 else "neutral"
        return (
            f"{symbol} is trading in an {trend} (close {row['Close']:.2f} vs "
            f"SMA50 {row['SMA_50']:.2f}). RSI(14) at {rsi_value:.1f} suggests "
            f"{momentum} momentum conditions. "
            "Note: heuristic summary — configure OPENAI_API_KEY for full AI analysis. "
            "This is not investment advice."
        )
