"""Streamlit entry point for the AI-powered Indian Stock Market Dashboard."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config.settings import DEFAULT_SYMBOLS
from src.ai.analyzer import StockAnalyzer
from src.ai.llm_client import LLMClient
from src.ai.recommender import RecommendationError, StockRecommender
from src.dashboard.components import (
    render_ai_section,
    render_indicator_chart,
    render_metric_row,
    render_price_chart,
    render_recommendation_cards,
)
from src.data.fetcher import StockDataFetcher
from src.data.indicators import add_all
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="AI Stock Terminal — India", page_icon="📈", layout="wide")

CUSTOM_CSS = """
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #e2e8f0; }
#MainMenu { visibility: hidden; } footer { visibility: hidden; } [data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; } h1, h2, h3, h4 { color: #f1f5f9; }
.main-header { background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%); padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25); }
.header-pill { background: rgba(255, 255, 255, 0.15); border: 1px solid rgba(255, 255, 255, 0.25); color: #ffffff; border-radius: 20px; padding: 4px 14px; display: inline-block; font-size: 0.85rem; margin: 12px 4px 0 4px; }
.section-header { color: #e2e8f0; border-left: 4px solid #3b82f6; padding-left: 12px; font-size: 1.2rem; font-weight: 700; margin: 24px 0 12px 0; }
.stock-card { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 20px; border-radius: 15px; border: 1px solid #475569; margin: 10px 0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); transition: transform 0.2s; }
.stock-card:hover { transform: translateY(-5px); box-shadow: 0 8px 12px rgba(0, 0, 0, 0.4); }
.intraday-badge { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; display: inline-block; }
.longterm-badge { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; display: inline-block; }
.target-box { background: rgba(37, 99, 235, 0.2); border: 1px solid #2563eb; padding: 10px; border-radius: 8px; text-align: center; }
.stoploss-box { background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; padding: 15px; border-radius: 8px; text-align: center; }
.metric-container { background: rgba(30, 41, 59, 0.8); padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #475569; }
.metric-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; }
.metric-value { font-size: 1.45rem; font-weight: 800; color: #f1f5f9; }
.metric-delta-up { color: #22c55e; font-weight: 600; } .metric-delta-down { color: #ef4444; font-weight: 600; }
.outlook-banner { background: rgba(37, 99, 235, 0.15); border: 1px solid #2563eb; border-radius: 12px; padding: 14px 18px; margin: 10px 0 18px 0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); }
.sidebar-brand { font-size: 1.25rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%); -webkit-background-clip: text; background-clip: text; color: transparent; padding-bottom: 8px; }
.stButton>button { background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%); color: white; border: none; padding: 10px 30px; border-radius: 25px; font-weight: bold; }
.stButton>button:hover { transform: scale(1.05); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4); }
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

def render_header() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin: 0;">📈 AI Stock Terminal</h1>
            <p style="margin: 8px 0 0 0; opacity: 0.92; font-size: 1.05rem;">AI-powered Indian market analytics &mdash; technicals, insights &amp; trade ideas</p>
            <div>
                <span class="header-pill">NSE / BSE</span>
                <span class="header-pill">Live OHLCV</span>
                <span class="header-pill">AI Top-5 Picks</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar() -> tuple[str, str, bool, bool]:
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">📈 AI Stock Terminal</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="margin-top: 6px;">📊 Stock Analysis</div>', unsafe_allow_html=True)
        symbol = st.selectbox("Symbol (NSE)", DEFAULT_SYMBOLS, index=0)
        period = st.selectbox("History period", ["6mo", "1y", "2y", "5y"], index=1)
        load_clicked = st.button("Load data", type="primary", width="stretch")

        st.markdown('<div class="section-header" style="margin-top: 10px;">🤖 AI Market Scan</div>', unsafe_allow_html=True)
        scan_clicked = st.button("Get Top 5 Picks", width="stretch")
        st.caption("Scans the top-30 NIFTY universe, ranks the biggest movers and asks the AI for 5 trade ideas.")
        st.divider()
        st.caption("Data: Yahoo Finance (delayed). For research & education only — not investment advice.")
    return symbol, period, load_clicked, scan_clicked

def main() -> None:
    render_header()
    symbol, period, load_clicked, scan_clicked = render_sidebar()

    if scan_clicked:
        with st.spinner("Scanning 30 NIFTY stocks · ranking movers · asking the AI…"):
            try:
                st.session_state["recs"] = StockRecommender(LLMClient()).get_top_recommendations()
            except RecommendationError as exc:
                logger.exception("Recommendation scan failed")
                st.error(f"Recommendation scan failed: {exc}")
    render_recommendation_cards(st.session_state.get("recs"))

    st.divider()
    st.markdown('<div class="section-header">📊 Stock Analysis</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=600, show_spinner=False)
    def _load(ticker: str, history_period: str):
        return StockDataFetcher(period=history_period).fetch_history(ticker)

    if load_clicked or "df" not in st.session_state or st.session_state.get("symbol") != symbol:
        try:
            with st.spinner(f"Fetching {symbol}…"):
                st.session_state["df"] = _load(symbol, period)
            st.session_state["symbol"] = symbol
        except Exception as exc:
            logger.exception("Failed to load %s", symbol)
            st.error(f"Could not load data for {symbol}: {exc}")
            st.stop()

    df = st.session_state["df"]
    enriched = add_all(df)

    render_metric_row(enriched)
    st.markdown('<div class="section-header">Price &amp; Indicators</div>', unsafe_allow_html=True)
    render_price_chart(enriched)
    render_indicator_chart(enriched)

    # ------------------------- TRACEBACK DEBUGGER SECTION ---------------- #
    st.markdown('<div class="section-header">🤖 AI Analysis</div>', unsafe_allow_html=True)
    
    if st.button("Generate AI insights", key="ai_insights_btn"):
        with st.spinner("Analyzing market data & generating insights..."):
            try:
                analyzer = StockAnalyzer()
                raw_response = analyzer.analyze(symbol, df)
                
                if raw_response is None or (isinstance(raw_response, dict) and not raw_response):
                    st.session_state["insights"] = {
                        "analysis": "⚠️ AI Agent se koi valid response nahi mila. Kripya API Key check karein.",
                        "sentiment": "Neutral"
                    }
                else:
                    st.session_state["insights"] = raw_response
                    
            except Exception as exc:
                # YEH LINE STREAMLIT LOGS MEIN EXACT ERROR DIKHAEGI!
                error_trace = traceback.format_exc()
                print("="*70)
                print("FULL ERROR TRACEBACK (YEH mujhe bhejna hai):")
                print(error_trace)
                print("="*70)
                
                st.session_state["insights"] = {
                    "analysis": f"❌ AI Analysis Failed. Exact error Streamlit Logs mein print ho gaya hai.",
                    "sentiment": "Error"
                }
                
    render_ai_section(st.session_state.get("insights"))

    st.markdown(
        '<div class="data-footer" style="color:#64748b;font-size:0.8rem;text-align:center;'
        'margin-top:30px;">Data: Yahoo Finance (delayed) · For research &amp; education only · '
        'Not investment advice</div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()