"""
Live Portfolio Dashboard
==========================
Real-time monitoring dashboard built with Dash/Plotly.

Pages:
  1. Overview: portfolio PnL, Greeks, regime, strategy allocation
  2. Risk: vega utilization, drawdown, margin, scenario results
  3. Signals: strategy signals, regime classifier output, entry/exit
  4. Execution: order status, TCA, phased build-up progress

Usage:
    python -m dashboard.app                     # Serve on port 8050
    python -m dashboard.app --port 8080         # Custom port
    python -m dashboard.app --data backtest     # Load from backtest results
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

try:
    import dash
    from dash import dcc, html, dash_table, callback_context
    from dash.dependencies import Input, Output
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    logger.warning("Dash not installed. Run: pip install dash plotly")


def load_backtest_data() -> pd.DataFrame | None:
    """Load the most recent backtest results."""
    path = Path(__file__).resolve().parent.parent / "backtest" / "output" / "backtest_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def create_app(data: pd.DataFrame | None = None) -> "dash.Dash":
    """Create the Dash application."""
    if not HAS_DASH:
        raise ImportError("Dash is required: pip install dash plotly")

    app = dash.Dash(
        __name__,
        title="Vol Portfolio Dashboard",
        suppress_callback_exceptions=True,
    )

    # Color scheme matching the HTML pages
    COLORS = {
        "bg": "#0d1117",
        "card": "#1a2332",
        "border": "#2a3e5c",
        "text": "#e2e8f0",
        "text_dim": "#8899aa",
        "teal": "#3ecfb4",
        "green": "#48b884",
        "red": "#e06060",
        "gold": "#d8a843",
        "purple": "#9b7ad8",
    }

    card_style = {
        "backgroundColor": COLORS["card"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "8px",
        "padding": "20px",
        "marginBottom": "15px",
    }

    metric_style = {
        "textAlign": "center",
        "padding": "15px",
        "backgroundColor": COLORS["card"],
        "borderRadius": "8px",
        "border": f"1px solid {COLORS['border']}",
    }

    if data is None:
        data = load_backtest_data()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    app.layout = html.Div(
        style={"backgroundColor": COLORS["bg"], "color": COLORS["text"],
               "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
               "minHeight": "100vh", "padding": "20px"},
        children=[
            # Header
            html.Div([
                html.H1("Systematic Volatility Portfolio",
                         style={"color": COLORS["teal"], "marginBottom": "5px", "fontSize": "24px"}),
                html.P("Real-Time Portfolio Dashboard",
                       style={"color": COLORS["text_dim"], "fontSize": "14px"}),
            ], style={"marginBottom": "25px"}),

            # Auto-refresh
            dcc.Interval(id="refresh", interval=30000, n_intervals=0),

            # Key metrics bar
            html.Div(id="metrics-bar", children=_build_metrics_bar(data, COLORS, metric_style)),

            # Charts row 1: Cumulative PnL + Daily PnL
            html.Div([
                html.Div([
                    html.H3("Cumulative PnL", style={"fontSize": "16px", "marginBottom": "10px"}),
                    dcc.Graph(id="cum-pnl-chart", figure=_build_cum_pnl_chart(data, COLORS)),
                ], style={**card_style, "width": "65%", "display": "inline-block", "verticalAlign": "top"}),

                html.Div([
                    html.H3("Regime Distribution", style={"fontSize": "16px", "marginBottom": "10px"}),
                    dcc.Graph(id="regime-chart", figure=_build_regime_chart(data, COLORS)),
                ], style={**card_style, "width": "33%", "display": "inline-block",
                          "verticalAlign": "top", "marginLeft": "2%"}),
            ]),

            # Charts row 2: Greeks + Strategy contribution
            html.Div([
                html.Div([
                    html.H3("Portfolio Greeks Over Time", style={"fontSize": "16px", "marginBottom": "10px"}),
                    dcc.Graph(id="greeks-chart", figure=_build_greeks_chart(data, COLORS)),
                ], style={**card_style, "width": "65%", "display": "inline-block", "verticalAlign": "top"}),

                html.Div([
                    html.H3("Strategy Contribution", style={"fontSize": "16px", "marginBottom": "10px"}),
                    dcc.Graph(id="strategy-chart", figure=_build_strategy_chart(data, COLORS)),
                ], style={**card_style, "width": "33%", "display": "inline-block",
                          "verticalAlign": "top", "marginLeft": "2%"}),
            ]),

            # Charts row 3: Drawdown + VIX
            html.Div([
                html.Div([
                    html.H3("Drawdown", style={"fontSize": "16px", "marginBottom": "10px"}),
                    dcc.Graph(id="drawdown-chart", figure=_build_drawdown_chart(data, COLORS)),
                ], style={**card_style, "width": "48%", "display": "inline-block", "verticalAlign": "top"}),

                html.Div([
                    html.H3("VIX Level", style={"fontSize": "16px", "marginBottom": "10px"}),
                    dcc.Graph(id="vix-chart", figure=_build_vix_chart(data, COLORS)),
                ], style={**card_style, "width": "48%", "display": "inline-block",
                          "verticalAlign": "top", "marginLeft": "4%"}),
            ]),

            # Footer
            html.Div([
                html.P("Systematic Volatility Portfolio - Dashboard",
                       style={"color": COLORS["text_dim"], "fontSize": "12px",
                              "textAlign": "center", "marginTop": "30px"}),
            ]),
        ],
    )

    return app


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _build_metrics_bar(data, colors, style):
    if data is None:
        return html.Div("No data loaded", style={"color": colors["text_dim"]})

    total_pnl = data["cumulative_pnl"].iloc[-1] if "cumulative_pnl" in data else 0
    max_dd = data["drawdown_pct"].max() * 100 if "drawdown_pct" in data else 0
    latest_vix = data["vix"].iloc[-1] if "vix" in data else 0
    latest_regime = data["regime"].iloc[-1] if "regime" in data else "N/A"
    win_rate = (data["total_pnl"] > 0).mean() * 100 if "total_pnl" in data else 0

    regime_color = {"LOW_VOL_HARVESTING": colors["green"],
                    "TRANSITIONAL": colors["gold"],
                    "CRISIS": colors["red"]}.get(latest_regime, colors["text"])

    metrics = [
        ("Total PnL", f"${total_pnl / 1e6:.1f}M", colors["green"] if total_pnl > 0 else colors["red"]),
        ("Max Drawdown", f"{max_dd:.1f}%", colors["red"] if max_dd > 10 else colors["gold"]),
        ("Win Rate", f"{win_rate:.0f}%", colors["teal"]),
        ("VIX", f"{latest_vix:.1f}", colors["text"]),
        ("Regime", latest_regime.replace("_", " "), regime_color),
        ("Days", f"{len(data)}", colors["text_dim"]),
    ]

    return html.Div([
        html.Div([
            html.Div(label, style={"fontSize": "11px", "color": colors["text_dim"],
                                    "letterSpacing": "0.05em", "marginBottom": "4px"}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "700", "color": color}),
        ], style={**style, "flex": "1", "margin": "0 5px"})
        for label, value, color in metrics
    ], style={"display": "flex", "marginBottom": "20px"})


def _chart_layout(colors, height=300):
    return dict(
        plot_bgcolor=colors["card"],
        paper_bgcolor=colors["card"],
        font=dict(color=colors["text_dim"], size=11),
        margin=dict(l=50, r=20, t=10, b=40),
        height=height,
        xaxis=dict(gridcolor=colors["border"], showgrid=True),
        yaxis=dict(gridcolor=colors["border"], showgrid=True),
    )


def _build_cum_pnl_chart(data, colors):
    fig = go.Figure()
    if data is not None and "cumulative_pnl" in data:
        fig.add_trace(go.Scatter(
            x=data["date"], y=data["cumulative_pnl"] / 1e6,
            mode="lines", line=dict(color=colors["teal"], width=2),
            fill="tozeroy", fillcolor="rgba(62,207,180,0.1)",
            name="Cumulative PnL",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color=colors["text_dim"], line_width=0.5)
    fig.update_layout(**_chart_layout(colors), yaxis_title="PnL ($M)")
    return fig


def _build_regime_chart(data, colors):
    fig = go.Figure()
    if data is not None and "regime" in data:
        counts = data["regime"].value_counts()
        regime_colors = {
            "LOW_VOL_HARVESTING": colors["green"],
            "TRANSITIONAL": colors["gold"],
            "CRISIS": colors["red"],
        }
        fig.add_trace(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            marker=dict(colors=[regime_colors.get(r, colors["text"]) for r in counts.index]),
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
    fig.update_layout(
        plot_bgcolor=colors["card"], paper_bgcolor=colors["card"],
        font=dict(color=colors["text_dim"], size=11),
        margin=dict(l=10, r=10, t=10, b=10), height=300,
        showlegend=False,
    )
    return fig


def _build_greeks_chart(data, colors):
    fig = go.Figure()
    if data is not None:
        for greek, color in [("net_vega", colors["teal"]), ("net_gamma", colors["gold"]),
                              ("net_theta", colors["red"]), ("net_delta", colors["purple"])]:
            if greek in data:
                fig.add_trace(go.Scatter(
                    x=data["date"], y=data[greek] / 1e6,
                    mode="lines", name=greek.replace("net_", "").title(),
                    line=dict(color=color, width=1.5),
                ))
    fig.update_layout(**_chart_layout(colors), yaxis_title="Greeks ($M)")
    return fig


def _build_strategy_chart(data, colors):
    fig = go.Figure()
    if data is not None and "strategy_pnl" in data:
        try:
            strat_pnls = pd.DataFrame(data["strategy_pnl"].apply(
                lambda x: x if isinstance(x, dict) else eval(x) if isinstance(x, str) else {}
            ).tolist())
            totals = strat_pnls.sum().sort_values()
            bar_colors = [colors["green"] if v > 0 else colors["red"] for v in totals.values]
            fig.add_trace(go.Bar(
                y=totals.index.tolist(),
                x=(totals.values / 1e6).tolist(),
                orientation="h",
                marker=dict(color=bar_colors),
            ))
        except Exception:
            pass
    fig.update_layout(**_chart_layout(colors, 300), xaxis_title="PnL ($M)")
    return fig


def _build_drawdown_chart(data, colors):
    fig = go.Figure()
    if data is not None and "drawdown_pct" in data:
        fig.add_trace(go.Scatter(
            x=data["date"], y=-data["drawdown_pct"] * 100,
            mode="lines", fill="tozeroy",
            line=dict(color=colors["red"], width=1.5),
            fillcolor="rgba(224,96,96,0.2)",
            name="Drawdown",
        ))
        fig.add_hline(y=-15, line_dash="dash", line_color=colors["red"],
                      line_width=1, annotation_text="15% Stop")
    fig.update_layout(**_chart_layout(colors), yaxis_title="Drawdown (%)")
    return fig


def _build_vix_chart(data, colors):
    fig = go.Figure()
    if data is not None and "vix" in data:
        regime_colors = data["regime"].map({
            "LOW_VOL_HARVESTING": colors["green"],
            "TRANSITIONAL": colors["gold"],
            "CRISIS": colors["red"],
        }).fillna(colors["text_dim"])

        fig.add_trace(go.Scatter(
            x=data["date"], y=data["vix"],
            mode="markers+lines",
            marker=dict(color=regime_colors.tolist(), size=3),
            line=dict(color=colors["text_dim"], width=0.5),
            name="VIX",
        ))
    fig.update_layout(**_chart_layout(colors), yaxis_title="VIX Level")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vol Portfolio Dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not HAS_DASH:
        print("Dash not installed. Run: pip install dash plotly")
        sys.exit(1)

    app = create_app()
    print(f"Dashboard running at http://localhost:{args.port}")
    app.run(debug=args.debug, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
