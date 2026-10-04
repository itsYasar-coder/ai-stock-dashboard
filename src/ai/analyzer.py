"""AI stock analysis: turns price/indicator data into natural-language insights."""

from __future__ import annotations

import logging
import re
from typing import Dict, Any

import pandas as pd

from src.ai.llm_client import LLMClient, LLMClientError
from src.data.indicators import add_all

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an elite equity research assistant for the Indian stock market (NSE/BSE). "
    "Analyze the provided technical data and respond with a concise, balanced analysis. "
    "Cover trend, momentum, volatility, and key support/resistance levels. "
    "End your response with a one-line reminder that this is not investment advice."
)


class StockAnalyzer:
    """Produces AI (or heuristic fallback) insights for a single stock."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def analyze(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Return structured insights dict for ``symbol``.
        
        Returns:
            {"analysis": str, "sentiment": str}
        Falls back to deterministic rule-based summary when no LLM is configured.
        """
        enriched = add_all(df)
        summary = self._build_data_summary(enriched)

        # --- AI Path ---
        if self.llm.is_configured:
            try:
                raw_response = self.llm.generate(SYSTEM_PROMPT, f"Stock: {symbol}\n\n{summary}")
                
                # Safe parsing of LLM response
                if isinstance(raw_response, dict):
                    # If LLM returned JSON directly
                    analysis_text = raw_response.get("analysis", raw_response.get("text", str(raw_response)))
                    sentiment = raw_response.get("sentiment", self._detect_sentiment(analysis_text))
                elif isinstance(raw_response, str):
                    # Clean markdown/code blocks if present
                    cleaned = re.sub(r'^```(?:json)?\s*', '', raw_response.strip())
                    cleaned = re.sub(r'\s*```$', '', cleaned)
                    analysis_text = cleaned
                    sentiment = self._detect_sentiment(cleaned)
                else:
                    analysis_text = str(raw_response)
                    sentiment = "Neutral"

                return {
                    "analysis": analysis_text,
                    "sentiment": sentiment
                }

            except LLMClientError as exc:
                logger.warning("AI insights unavailable (%s); using fallback", exc)
            except Exception as e:
                logger.exception("Unexpected error during AI analysis for %s", symbol)
                return {
                    "analysis": f"⚠️ AI processing failed: {str(e)[:100]}",
                    "sentiment": "Error"
                }

        # --- Fallback Path ---
        fallback_text = self._fallback_summary(symbol, enriched)
        return {
            "analysis": fallback_text,
            "sentiment": self._detect_sentiment(fallback_text)
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_sentiment(text: str) -> str:
        """Simple keyword-based sentiment detection from analysis text."""
        lower_text = text.lower()
        bullish_keywords = ["bullish", "buy", "uptrend", "positive", "breakout", "accumulate"]
        bearish_keywords = ["bearish", "sell", "downtrend", "negative", "breakdown", "avoid"]
        
        bull_score = sum(1 for kw in bullish_keywords if kw in lower_text)
        bear_score = sum(1 for kw in bearish_keywords if kw in lower_text)
        
        if bull_score > bear_score:
            return "Bullish"
        elif bear_score > bull_score:
            return "Bearish"
        return "Neutral"

    @staticmethod
    def _build_data_summary(enriched: pd.DataFrame) -> str:
        """Condense the indicator frame into a compact text briefing."""
        latest = enriched.dropna().tail(1)
        if latest.empty:
            return "Not enough data to compute indicators."
            
        row = latest.iloc[-1]
        change_pct = enriched["Close"].pct_change().tail(30).mean() * 100
        
        # Safely get values avoiding NaN formatting issues
        def fmt(val, decimals=2):
            try:
                return f"{float(val):.{decimals}f}"
            except (ValueError, TypeError):
                return "N/A"

        return (
            f"Latest close: {fmt(row['Close'])} INR\n"
            f"30-day average daily change: {change_pct:+.2f}%\n"
            f"RSI(14): {fmt(row.get('RSI_14'), 1)}\n"
            f"SMA20: {fmt(row.get('SMA_20'))}, SMA50: {fmt(row.get('SMA_50'))}\n"
            f"MACD: {fmt(row.get('MACD'), 3)} vs Signal: {fmt(row.get('Signal'), 3)}\n"
            f"Bollinger bands: lower {fmt(row.get('BB_Lower'))} / upper {fmt(row.get('BB_Upper'))}\n"
            f"Data window: {enriched.index[0].date()} to {enriched.index[-1].date()} ({len(enriched)} sessions)"
        )

    @staticmethod
    def _fallback_summary(symbol: str, enriched: pd.DataFrame) -> str:
        """Rule-based summary used when the LLM is not configured."""
        row = enriched.dropna().tail(1).iloc[0]
        trend = "uptrend" if row["Close"] > row["SMA_50"] else "downtrend"
        
        try:
            rsi_value = float(row["RSI_14"])
            if rsi_value >= 70:
                momentum = "overbought"
            elif rsi_value <= 30:
                momentum = "oversold"
            else:
                momentum = "neutral"
        except (ValueError, TypeError):
            momentum = "unknown"
            rsi_value = 0.0

        return (
            f"{symbol} is trading in an {trend} (close {row['Close']:.2f} vs "
            f"SMA50 {row['SMA_50']:.2f}). RSI(14) at {rsi_value:.1f} suggests "
            f"{momentum} momentum conditions. "
            "Note: heuristic summary — configure OPENAI_API_KEY for full AI analysis. "
            "This is not investment advice."
        )