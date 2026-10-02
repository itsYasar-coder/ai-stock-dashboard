"""CLI entry point for the AI-powered Indian Stock Market Dashboard.

Examples:
    python main.py fetch RELIANCE     # download & preview price history
    python main.py analyze TCS        # indicator snapshot + AI insights
    python main.py recommend          # Top-5 AI picks from top Nifty movers

The web dashboard runs separately:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import argparse
import json
import sys

from src.data.fetcher import StockDataFetcher
from src.data.indicators import add_all
from src.utils.logger import get_logger

logger = get_logger("main")


def _cmd_fetch(args: argparse.Namespace) -> int:
    fetcher = StockDataFetcher(period=args.period)
    df = fetcher.fetch_history(args.symbol)
    fetcher.save_history(args.symbol, df)
    print(f"\n{args.symbol}: {len(df)} rows fetched "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(df.tail(5).to_string())
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from src.ai.analyzer import StockAnalyzer

    fetcher = StockDataFetcher(period=args.period)
    df = fetcher.fetch_history(args.symbol)
    enriched = add_all(df)
    latest = enriched.dropna().tail(1)
    if latest.empty:
        print("Not enough data to compute indicators.")
        return 1

    print(f"\n=== {args.symbol} indicator snapshot ===")
    print(latest.iloc[0][["Close", "SMA_20", "SMA_50", "RSI_14", "MACD", "Signal"]].to_string())
    print("\n=== AI analysis ===")
    print(StockAnalyzer().analyze(args.symbol, df))
    return 0


def _cmd_recommend(args: argparse.Namespace) -> int:
    from src.ai.llm_client import LLMClient
    from src.ai.recommender import RecommendationError, StockRecommender

    print(f"Scanning top Nifty universe and ranking movers (top {args.movers})...")
    result = StockRecommender(LLMClient(), movers=args.movers).get_top_recommendations()
    print(json.dumps(result, indent=2))
    print("\nDisclaimer: research/education output only — not investment advice.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indian-stock-dashboard", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download and preview price history")
    fetch.add_argument("symbol", help="Stock symbol, e.g. RELIANCE")
    fetch.add_argument("--period", default="1y", help="History window (default: 1y)")
    fetch.set_defaults(func=_cmd_fetch)

    analyze = sub.add_parser("analyze", help="Indicator snapshot + AI insights")
    analyze.add_argument("symbol", help="Stock symbol, e.g. TCS")
    analyze.add_argument("--period", default="1y", help="History window (default: 1y)")
    analyze.set_defaults(func=_cmd_analyze)

    recommend = sub.add_parser("recommend", help="Top-5 AI picks from the top Nifty movers")
    recommend.add_argument("--movers", type=int, default=15, help="Brief the AI on the top N movers (default: 15)")
    recommend.set_defaults(func=_cmd_recommend)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report and fail cleanly
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
