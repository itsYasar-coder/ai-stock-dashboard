"""Reusable Streamlit/Plotly UI components for the dark trading-terminal theme.

Kept separate from app.py so charts and cards can be unit-tested and reused
in notebooks or reports.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Slate palette shared by all charts (matches the app's dark CSS).
_TEXT = "#e2e8f0"
_GRID = "#334155"
_UP = "#22c55e"
_DOWN = "#ef4444"
_ACCENT_BLUE = "#3b82f6"
_AMBER = "#f59e0b"
_MUTED = "#64748b"

_DARK_LAYOUT: dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": _TEXT, "family": "Inter, 'Segoe UI', sans-serif"},
    "xaxis": {"gridcolor": _GRID, "zerolinecolor": "#475569"},
    "yaxis": {"gridcolor": _GRID, "zerolinecolor": "#475569"},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "font": {"color": "#cbd5e1"}},
    "margin": {"l": 10, "r": 10, "t": 10, "b": 10},
    "hoverlabel": {"bgcolor": "#1e293b", "bordercolor": "#475569", "font": {"color": _TEXT}},
}


def _dark(fig: go.Figure, **overrides: Any) -> go.Figure:
    fig.update_layout(**{**_DARK_LAYOUT, **overrides})
    return fig


def render_price_chart(df: pd.DataFrame) -> None:
    """Dark candlestick chart with SMA20/SMA50 and Bollinger band overlays."""
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color=_UP,
            increasing_fillcolor=_UP,
            decreasing_line_color=_DOWN,
            decreasing_fillcolor=_DOWN,
        )
    )
    for column, color in (("SMA_20", _AMBER), ("SMA_50", "#60a5fa")):
        if column in df:
            fig.add_trace(go.Scatter(x=df.index, y=df[column], name=column, line=dict(color=color, width=1.6)))
    if {"BB_Upper", "BB_Lower"} <= set(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_Upper"], name="BB upper",
                line=dict(width=1, dash="dot", color="#94a3b8"), opacity=0.6,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_Lower"], name="BB lower",
                line=dict(width=1, dash="dot", color="#94a3b8"), opacity=0.6,
            )
        )

    _dark(fig, height=460, xaxis_rangeslider_visible=False, yaxis_title="Price (INR)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_indicator_chart(df: pd.DataFrame) -> None:
    """Dark two-row panel: RSI on top, MACD below (histogram colored by sign)."""
    fig = go.Figure().set_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)

    if "RSI_14" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI(14)", line=dict(color="#a78bfa", width=1.6)), row=1, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=_DOWN, opacity=0.6, row=1, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=_UP, opacity=0.6, row=1, col=1)

    if {"MACD", "Signal"} <= set(df.columns):
        if "Histogram" in df:
            hist_colors = [_UP if value >= 0 else _DOWN for value in df["Histogram"].fillna(0)]
            fig.add_trace(go.Bar(x=df.index, y=df["Histogram"], name="Histogram", marker_color=hist_colors, opacity=0.55), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#60a5fa", width=1.6)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal", line=dict(color=_AMBER, width=1.4)), row=2, col=1)

    _dark(fig, height=400, legend=dict(orientation="h", y=1.06, font={"color": "#cbd5e1"}))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# --------------------------------------------------------------------------- #
# Metric row (KPI cards under the header)
# --------------------------------------------------------------------------- #

def _fmt_indian_number(value: float) -> str:
    """Indian market conventions: crore / lakh suffixes."""
    if value >= 1e7:
        return f"{value / 1e7:,.2f} Cr"
    if value >= 1e5:
        return f"{value / 1e5:,.2f} L"
    return f"{value:,.0f}"


def _metric_html(label: str, value: str, delta: str = "", delta_class: str = "") -> str:
    delta_html = f'<div class="{delta_class}">{html.escape(delta)}</div>' if delta else ""
    return (
        f'<div class="metric-container">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f"{delta_html}</div>"
    )


def render_metric_row(df: pd.DataFrame) -> None:
    """KPI strip: LTP, RSI(14), 20-day range and volume."""
    closes = df["Close"].dropna()
    if closes.empty:
        return
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) >= 2 else last
    day_change = (last / prev - 1.0) * 100.0 if prev else 0.0

    rsi_row = df.dropna(subset=["RSI_14"]).tail(1)
    rsi_value = float(rsi_row.iloc[0]["RSI_14"]) if not rsi_row.empty else None
    rsi_label = ""
    if rsi_value is not None:
        rsi_label = "Overbought" if rsi_value >= 70 else "Oversold" if rsi_value <= 30 else "Neutral"

    has_range = {"High", "Low"} <= set(df.columns)
    high_20 = float(df["High"].tail(20).max()) if has_range else None
    low_20 = float(df["Low"].tail(20).min()) if has_range else None
    volume = float(df["Volume"].iloc[-1]) if "Volume" in df and pd.notna(df["Volume"].iloc[-1]) else None

    col_ltp, col_rsi, col_range, col_vol = st.columns(4)
    with col_ltp:
        st.markdown(
            _metric_html(
                "Last Traded Price",
                f"₹{last:,.2f}",
                f"{day_change:+.2f}% today",
                "metric-delta-up" if day_change >= 0 else "metric-delta-down",
            ),
            unsafe_allow_html=True,
        )
    with col_rsi:
        st.markdown(
            _metric_html(
                "RSI (14)",
                f"{rsi_value:.1f}" if rsi_value is not None else "n/a",
                rsi_label,
                "metric-delta-down" if rsi_value is not None and rsi_value >= 70
                else "metric-delta-up" if rsi_value is not None and rsi_value <= 30
                else "metric-label",
            ),
            unsafe_allow_html=True,
        )
    with col_range:
        st.markdown(
            _metric_html(
                "20-Day Range",
                f"₹{low_20:,.0f} – ₹{high_20:,.0f}" if has_range else "n/a",
            ),
            unsafe_allow_html=True,
        )
    with col_vol:
        st.markdown(
            _metric_html("Volume", _fmt_indian_number(volume) if volume is not None else "n/a"),
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# AI analysis & Top-5 recommendation cards
# --------------------------------------------------------------------------- #

def render_ai_section(insights: str | None) -> None:
    """Render the AI analysis block (or a hint when not yet generated)."""
    if insights:
        st.markdown(insights)
    else:
        st.info("Click **Generate AI insights** to produce an analysis of the selected stock.")


def _fmt_money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"₹{number:,.2f}"


def _pct(value: float, base: float) -> str:
    if not base:
        return ""
    return f"{(value / base - 1.0) * 100:+.1f}%"


def _pick_card_html(pick: dict) -> str:
    """One premium stock card: header + badge, entry, targets, stop-loss, R:R."""
    name = html.escape(str(pick.get("stock_name", "")))
    trade_type = str(pick.get("trade_type", "Longterm"))
    badge_class = "intraday-badge" if trade_type.lower() == "intraday" else "longterm-badge"

    try:
        buy = float(pick.get("buy_price"))
        stop = float(pick.get("sell_price"))
    except (TypeError, ValueError):
        buy, stop = 0.0, 0.0

    targets = [t for t in pick.get("targets", [])][:3]
    target_values = []
    for target in targets:
        try:
            target_values.append(float(target))
        except (TypeError, ValueError):
            pass

    target_boxes = "".join(
        f'<div class="target-box" style="flex:1;">'
        f'<div class="metric-label">T{i + 1}</div>'
        f'<div style="font-weight:700;color:#93c5fd;">{_fmt_money(value)}</div>'
        f'<div style="font-size:0.72rem;color:#94a3b8;">{_pct(value, buy)}</div>'
        f"</div>"
        for i, value in enumerate(target_values)
    )

    risk = abs(buy - stop)
    reward = abs(target_values[-1] - buy) if target_values else 0.0
    rr = reward / risk if risk else 0.0

    return (
        f'<div class="stock-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f"<div>"
        f'<div style="font-size:1.45rem;font-weight:800;letter-spacing:0.02em;">{name}</div>'
        f'<span class="{badge_class}">{html.escape(trade_type.upper())}</span>'
        f"</div>"
        f'<div style="text-align:right;">'
        f'<div class="metric-label">Buy @</div>'
        f'<div style="font-size:1.55rem;font-weight:800;color:{_UP};">{_fmt_money(buy)}</div>'
        f"</div></div>"
        f'<div style="display:flex;gap:8px;margin-top:14px;">{target_boxes}</div>'
        f'<div style="display:flex;gap:8px;margin-top:10px;">'
        f'<div class="stoploss-box" style="flex:1;">'
        f'<div class="metric-label">Stop Loss</div>'
        f'<div style="font-weight:700;color:#fca5a5;">{_fmt_money(stop)} '
        f'<span style="font-size:0.75rem;">{_pct(stop, buy)}</span></div>'
        f"</div>"
        f'<div class="metric-container" style="flex:1;">'
        f'<div class="metric-label">Risk : Reward</div>'
        f'<div style="font-weight:700;">1 : {rr:.1f}</div>'
        f"</div></div></div>"
    )


def render_recommendation_cards(recs: dict | None) -> None:
    """Render the Top-5 AI picks as premium cards in a two-column grid."""
    if not recs:
        st.info("Click **Get Top 5 Picks** in the sidebar to run the AI market scan.")
        return

    outlook = recs.get("market_outlook")
    if outlook:
        st.markdown(
            f'<div class="outlook-banner">🧭 <b>Market Outlook:</b> {html.escape(str(outlook))}</div>',
            unsafe_allow_html=True,
        )

    picks = recs.get("top_5_picks", [])
    if not picks:
        st.warning("The scan returned no picks.")
        return

    columns = st.columns(2, gap="medium")
    for index, pick in enumerate(picks):
        with columns[index % 2]:
            st.markdown(_pick_card_html(pick), unsafe_allow_html=True)

    st.caption("Research/education output only — not investment advice.")
