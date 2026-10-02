"""Top-5 stock recommendations from AI analysis of the top Nifty movers.

Pipeline:
    1. Fetch history for the top-30 NIFTY universe (thread-pooled).
    2. Rank symbols by absolute daily % change and keep the top 15 movers.
    3. Enrich the movers with technical indicators and build a text briefing.
    4. Ask the LLM for exactly 5 trade ideas in a strict JSON format.
    5. Parse/validate the JSON (coercing types) — falling back to a
       deterministic rule-based scan when the LLM is unavailable or
       returns unusable output.

The recommendations are research/education output, not investment advice.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from config.settings import TOP_30_SYMBOLS
from src.ai.llm_client import LLMClient, LLMClientError
from src.data.fetcher import StockDataFetcher
from src.data.indicators import add_all
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an AI trading assistant analysing Indian equities listed on NSE (NIFTY 50). "
    "You receive a briefing on the day's biggest movers with technical indicators.\n"
    "Task: pick the 5 most promising setups and return ONLY a JSON object — no markdown, "
    "no commentary — with exactly this shape:\n"
    "{\n"
    '  "market_outlook": "<1-2 sentence overall market sentiment>",\n'
    '  "top_5_picks": [\n'
    "    {\n"
    '      "trade_type": "Intraday" or "Longterm",\n'
    '      "stock_name": "<SYMBOL>",\n'
    '      "buy_price": <entry level as a number>,\n'
    '      "targets": ["<target1>", "<target2>", "<target3>"],\n'
    '      "sell_price": <stop-loss level as a number>\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Rules: prices are in INR anchored to the latest close provided; targets are strings "
    "with 2 decimals ordered by increasing distance from the entry; sell_price is the "
    "protective stop-loss (below the entry for longs, above for shorts); choose trade_type "
    "based on the setup's time horizon. "
    "Your output is for research and education only — it is not investment advice."
)


class RecommendationError(RuntimeError):
    """Raised when the recommender cannot produce valid recommendations."""


class StockRecommender:
    """Scans the NIFTY universe and produces Top-5 AI trade recommendations.

    Args:
        llm_client: Configured (or not) LLM client; when unconfigured the
            heuristic fallback is used.
        fetcher: Anything exposing ``fetch_history(symbol)``; defaults to a
            ``StockDataFetcher`` with a 6-month daily window. Injectable for
            offline testing.
        universe: Symbols to scan; defaults to the top-30 NIFTY list.
        movers: How many top absolute-%-change symbols to brief the AI on.
        num_picks: Number of recommendations to return.
        max_workers: Thread-pool size for the parallel data fetch.
        min_universe: Minimum symbols that must fetch successfully.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        fetcher: Any | None = None,
        universe: Sequence[str] | None = None,
        movers: int = 15,
        num_picks: int = 5,
        max_workers: int = 8,
        min_universe: int = 5,
    ):
        self.llm_client = llm_client
        self.fetcher = fetcher or StockDataFetcher(period="6mo", interval="1d")
        self.universe: tuple[str, ...] = tuple(universe or TOP_30_SYMBOLS)
        self.movers = movers
        self.num_picks = num_picks
        self.max_workers = max_workers
        self.min_universe = min_universe

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_top_recommendations(self) -> dict:
        """Run the full pipeline and return the recommendation payload.

        Returns:
            {
                "market_outlook": "Brief sentiment",
                "top_5_picks": [
                    {
                        "trade_type": "Intraday" | "Longterm",
                        "stock_name": "RELIANCE",
                        "buy_price": 2950.50,
                        "targets": ["3000.00", "3050.00", "3100.00"],
                        "sell_price": 2900.00
                    }, ...
                ]
            }

        Raises:
            RecommendationError: If too few symbols could be fetched.
                (LLM failures degrade to the heuristic fallback instead.)
        """
        universe = self._fetch_universe()
        movers = self._rank_movers(universe)
        logger.info(
            "Top movers: %s",
            ", ".join(f"{m['symbol']} ({m['day_change_pct']:+.2f}%)" for m in movers),
        )

        if self.llm_client.is_configured:
            try:
                raw = self.llm_client.generate(SYSTEM_PROMPT, self._build_user_prompt(movers))
                result = self._parse_recommendations(raw)
                logger.info("Parsed %d AI picks", len(result["top_5_picks"]))
                return result
            except (LLMClientError, RecommendationError) as exc:
                logger.warning("AI recommendations unavailable (%s); using heuristic fallback", exc)

        return self._heuristic_recommendations(movers)

    # ------------------------------------------------------------------ #
    # Pipeline steps
    # ------------------------------------------------------------------ #

    def _fetch_universe(self) -> dict[str, pd.DataFrame]:
        """Fetch history for every universe symbol in parallel, skipping failures."""
        def _load(symbol: str) -> tuple[str, pd.DataFrame | None]:
            try:
                return symbol, self.fetcher.fetch_history(symbol)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the scan
                logger.warning("Skipping %s: %s", symbol, exc)
                return symbol, None

        universe: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for symbol, frame in pool.map(_load, self.universe):
                if frame is not None and len(frame) >= 2:
                    universe[symbol] = frame

        if len(universe) < self.min_universe:
            raise RecommendationError(
                f"Insufficient data: fetched {len(universe)} of {len(self.universe)} symbols "
                f"(need at least {self.min_universe})"
            )
        logger.info("Fetched %d/%d universe symbols", len(universe), len(self.universe))
        return universe

    def _rank_movers(self, universe: dict[str, pd.DataFrame]) -> list[dict]:
        """Keep the top N movers by absolute daily % change, with indicators attached."""
        scored: list[dict] = []
        for symbol, df in universe.items():
            closes = df["Close"].dropna()
            if len(closes) < 2:
                continue
            day_change = (closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0
            scored.append({"symbol": symbol, "history": df, "day_change_pct": day_change})

        scored.sort(key=lambda item: abs(item["day_change_pct"]), reverse=True)
        movers = scored[: self.movers]
        for item in movers:
            item["enriched"] = add_all(item["history"])
        return movers

    def _build_user_prompt(self, movers: list[dict]) -> str:
        blocks = [self._stock_block(item) for item in movers]
        footer = (
            f"\n\nSelect exactly {self.num_picks} picks from the stocks above. "
            "Respond with ONLY the JSON object — no markdown fences, no commentary."
        )
        return "\n\n".join(blocks) + footer

    # ------------------------------------------------------------------ #
    # Response parsing / validation
    # ------------------------------------------------------------------ #

    def _parse_recommendations(self, raw: str) -> dict:
        return self._validate_recommendations(self._extract_json(raw))

    @staticmethod
    def _extract_json(raw: str) -> Any:
        """Pull a JSON object out of an LLM response (direct, fenced, or embedded)."""
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1).strip())
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise RecommendationError(f"Malformed JSON in LLM response: {exc}") from exc

        raise RecommendationError(f"No JSON object found in LLM response: {raw[:200]!r}")

    def _validate_recommendations(self, payload: Any) -> dict:
        """Enforce the strict schema, coercing types and dropping invalid picks."""
        if not isinstance(payload, dict):
            raise RecommendationError("LLM JSON must be an object with 'market_outlook' and 'top_5_picks'")

        outlook = payload.get("market_outlook")
        outlook = outlook.strip() if isinstance(outlook, str) and outlook.strip() else "No market outlook provided."

        raw_picks = payload.get("top_5_picks")
        if not isinstance(raw_picks, list):
            raise RecommendationError("'top_5_picks' must be a list")

        picks: list[dict] = []
        for entry in raw_picks:
            if len(picks) >= self.num_picks:
                break
            if not isinstance(entry, dict) or not str(entry.get("stock_name", "")).strip():
                continue
            buy = self._to_float(entry.get("buy_price"))
            stop = self._to_float(entry.get("sell_price"))
            if buy is None or buy <= 0 or stop is None or stop <= 0:
                continue
            picks.append(
                {
                    "trade_type": str(entry.get("trade_type", "Longterm")).strip().title() or "Longterm",
                    "stock_name": str(entry["stock_name"]).strip().upper(),
                    "buy_price": round(buy, 2),
                    "targets": self._coerce_targets(entry.get("targets"), buy, stop),
                    "sell_price": round(stop, 2),
                }
            )

        if not picks:
            raise RecommendationError("No valid picks found in LLM response")
        return {"market_outlook": outlook, "top_5_picks": picks}

    @staticmethod
    def _coerce_targets(targets: Any, buy_price: float, stop_price: float) -> list[str]:
        """Normalise targets to exactly 3 strings with 2 decimals.

        Short lists are padded by extrapolating the last objective's distance
        from the entry; missing/empty lists are derived from the entry with
        the direction inferred from the stop-loss.
        """
        values: list[float] = []
        if isinstance(targets, list):
            for target in targets:
                value = StockRecommender._to_float(target)
                if value is not None:
                    values.append(value)
        if not values:
            sign = 1.0 if stop_price <= buy_price else -1.0
            values = [buy_price * (1 + sign * step) for step in (0.02, 0.04, 0.06)]

        values = values[:3]
        while len(values) < 3:
            last = values[-1]
            step = abs(last - buy_price) or buy_price * 0.02
            sign = 1.0 if last >= buy_price else -1.0
            values.append(last + sign * step)
        return [f"{value:.2f}" for value in values]

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Best-effort numeric coercion (handles strings like "2,950.50")."""
        try:
            result = float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            return None
        return result if result == result else None  # NaN check

    # ------------------------------------------------------------------ #
    # Briefing & heuristic fallback
    # ------------------------------------------------------------------ #

    def _stock_block(self, item: dict) -> str:
        """Format one mover's indicator snapshot for the AI briefing."""
        df = item["enriched"]
        closes = df["Close"].dropna()

        change_5d = (closes.iloc[-1] / closes.iloc[-6] - 1) * 100 if len(closes) >= 6 else float("nan")
        change_20d = (closes.iloc[-1] / closes.iloc[-21] - 1) * 100 if len(closes) >= 21 else float("nan")

        lines = [
            f"### {item['symbol']} (NSE)",
            f"Latest session: {df.index[-1].date()} | Close: {self._num(closes.iloc[-1])} INR "
            f"| Day change: {item['day_change_pct']:+.2f}%",
            f"5-day change: {self._num(change_5d, '{:+.2f}%')} | "
            f"20-day change: {self._num(change_20d, '{:+.2f}%')}",
        ]

        last = df.dropna(subset=["SMA_20", "RSI_14"]).tail(1)
        if last.empty:
            lines.append("Indicators: not enough data")
            return "\n".join(lines)

        row = last.iloc[0]
        lines.append(
            f"RSI(14): {self._num(row['RSI_14'], '{:.1f}')} | MACD: {self._num(row['MACD'], '{:.3f}')} "
            f"Signal: {self._num(row['Signal'], '{:.3f}')}"
        )
        lines.append(f"SMA20: {self._num(row['SMA_20'])} | SMA50: {self._num(row['SMA_50'])}")
        pct_b = (row["Close"] - row["BB_Lower"]) / (row["BB_Upper"] - row["BB_Lower"])
        lines.append(
            f"Bollinger: lower {self._num(row['BB_Lower'])} / upper {self._num(row['BB_Upper'])} "
            f"(%B {self._num(pct_b, '{:.2f}')})"
        )
        if {"High", "Low"} <= set(df.columns):
            lines.append(
                f"20-day high/low: {self._num(df['High'].tail(20).max())} / {self._num(df['Low'].tail(20).min())}"
            )
        if "Volume" in df:
            avg_volume = df["Volume"].iloc[-21:-1].mean()
            volume_ratio = df["Volume"].iloc[-1] / avg_volume if avg_volume else float("nan")
            lines.append(f"Volume vs 20-day avg: {self._num(volume_ratio, '{:.2f}x')}")
        return "\n".join(lines)

    def _heuristic_recommendations(self, movers: list[dict]) -> dict:
        """Deterministic momentum scan used when the LLM is unavailable."""
        rows: list[tuple[dict, pd.Series | None]] = []
        for item in movers:
            last = item["enriched"].dropna(subset=["SMA_20", "MACD", "Signal"]).tail(1)
            rows.append((item, last.iloc[0] if not last.empty else None))

        def _bullish(pair: tuple[dict, pd.Series | None]) -> bool:
            item, row = pair
            return row is not None and bool(row["Close"] > row["SMA_20"] and row["MACD"] > row["Signal"])

        bullish = sorted(
            (pair for pair in rows if _bullish(pair)),
            key=lambda pair: pair[0]["day_change_pct"],
            reverse=True,
        )
        rest = sorted(
            (pair for pair in rows if not _bullish(pair)),
            key=lambda pair: abs(pair[0]["day_change_pct"]),
            reverse=True,
        )
        picks = [self._heuristic_pick(item, row) for item, row in (bullish + rest)[: self.num_picks]]
        picks = [pick for pick in picks if pick is not None]

        above = sum(1 for _, row in rows if row is not None and row["Close"] > row["SMA_20"])
        tone = "constructive" if above >= len(rows) / 2 else "cautious"
        outlook = (
            f"Heuristic scan: {above}/{len(rows)} of the top movers trade above their 20-day SMA — "
            f"{tone} short-term breadth. Set OPENAI_API_KEY for full AI analysis."
        )
        return {"market_outlook": outlook, "top_5_picks": picks}

    def _heuristic_pick(self, item: dict, row: pd.Series | None) -> dict | None:
        if row is None:
            return None
        close = float(row["Close"])
        trade_type = "Intraday" if abs(item["day_change_pct"]) >= 2.0 else "Longterm"
        if trade_type == "Intraday":
            targets = [close * (1 + step) for step in (0.006, 0.010, 0.014)]
            stop = close * 0.995
        else:
            targets = [close * (1 + step) for step in (0.02, 0.04, 0.06)]
            stop = close * 0.95
        return {
            "trade_type": trade_type,
            "stock_name": item["symbol"].upper(),
            "buy_price": round(close, 2),
            "targets": [f"{target:.2f}" for target in targets],
            "sell_price": round(stop, 2),
        }

    @staticmethod
    def _num(value: Any, spec: str = "{:.2f}") -> str:
        """NaN-safe number formatting for the briefing text."""
        try:
            if pd.isna(value):
                return "n/a"
            return spec.format(float(value))
        except (TypeError, ValueError):
            return "n/a"
