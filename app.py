from dash import Dash, html, dcc, Input, Output, State, ALL, ctx, no_update
import dash_bootstrap_components as dbc
from pathlib import Path
import json
import pandas as pd
from dash import dash_table
import plotly.graph_objects as go
import numpy as np
from dash.dash_table.Format import Format, Group, Scheme
from optimizer_backend import update_data_from_ui, run_optimizer_for_ui
import io
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots


# =================================================
# Load input data
# =================================================
BASE_DIR = Path(__file__).parent
with open(BASE_DIR / "optimizer_input.json", "r") as f:
    data = json.load(f)

base_investment = {k: float(v) for k, v in data["spends"].items()}
channels = list(base_investment.keys())


INCREMENTAL_CHANNELS = data.get("incremental_channels", {})
OPTIMIZED_CHANNELS = set(data["s_curve_params"].keys())


# =================================================
# Helpers
# =================================================
def parse_currency(val):
    if val in (None, ""):
        return None
    if isinstance(val, (int, float, np.number)):
        return float(val)
    return float(str(val).replace(",", "").strip())


def format_currency(val):
    return f"{int(round(val)):,}"


def fmt_money_short(x):
    x = float(x)
    if abs(x) >= 1e6:
        return f"${x/1e6:.1f}M"
    elif abs(x) >= 1e3:
        return f"${x/1e3:.1f}K"
    else:
        return f"${x:,.0f}"


def fmt_money_full(x):
    return f"${x:,.0f}"

# =================================================
# Visualization Helpers (Pack A)
# =================================================
def _to_num(x):
    """Coerce to float for both already-numeric and currency-like strings."""
    v = parse_currency(x)
    return float(v) if v is not None else np.nan


def _clean_results_df(results, optimized_only=True):
    df = pd.DataFrame(results).copy()

    if df.empty or "Channel" not in df.columns:
        return pd.DataFrame()

    # Ensure Channel Type exists
    if "Channel Type" not in df.columns:
        df["Channel Type"] = np.where(
            df["Channel"].isin(OPTIMIZED_CHANNELS),
            "Optimized",
            "Incremental (Linear)"
        )

    df = df[df["Channel"].notna()]

    if optimized_only:
        df = df[df["Channel Type"] == "Optimized"].copy()

    return df



def build_total_waterfall(results):
    """
    Waterfall story:
    Current Total Revenue (incl baseline) -> Optimized Total Revenue (incl baseline)
    Green bar shows delta only.
    """
    df = _clean_results_df(results, optimized_only=False)
    baseline = float(get_baseline(data))

    df2 = df[df["Channel"].astype(str).str.upper() != "TOTAL"].copy()
    if df2.empty:
        return go.Figure()

    actual_spend = df2["Actual/Input Spend"].apply(_to_num).sum()
    opt_spend = df2["Optimized Spend"].apply(_to_num).sum()
    actual_incr = df2["Actual Response Metric"].apply(_to_num).sum()
    opt_incr = df2["Optimized Response Metric"].apply(_to_num).sum()

    actual_total = actual_incr + baseline
    opt_total = opt_incr + baseline

    delta_total = opt_total - actual_total
    delta_spend = opt_spend - actual_spend

    actual_roas = (actual_incr / actual_spend) if actual_spend else 0.0
    opt_roas = (opt_incr / opt_spend) if opt_spend else 0.0

    # Per-bar hover templates (CRITICAL)
    hover_templates = [
        # Current
        (
            "<b>Current Total (Incl. Baseline)</b><br><br>"
            f"Total Revenue: {fmt_money_full(actual_total)}<br>"
            f"Total Spend: {fmt_money_full(actual_spend)}<br>"
            f"Incremental Revenue: {fmt_money_full(actual_incr)}<br>"
            f"Incremental ROAS: {actual_roas:.2f}"
            "<extra></extra>"
        ),
        # Delta
        (
            "<b>Δ Total Revenue</b><br><br>"
            f"Incremental Change: {fmt_money_full(delta_total)}<br>"
            f"Δ Total Spend: {fmt_money_full(delta_spend)}<br>"
            "From media reallocation only"
            "<extra></extra>"
        ),
        # Optimized
        (
            "<b>Optimized Total (Incl. Baseline)</b><br><br>"
            f"Total Revenue: {fmt_money_full(opt_total)}<br>"
            f"Total Spend: {fmt_money_full(opt_spend)}<br>"
            f"Incremental Revenue: {fmt_money_full(opt_incr)}<br>"
            f"Incremental ROAS: {opt_roas:.2f}"
            "<extra></extra>"
        ),
    ]

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=[
                "Current Total (Incl. Baseline)",
                "Δ Total Revenue",
                "Optimized Total (Incl. Baseline)",
            ],
            y=[actual_total, delta_total, opt_total],
            connector={"line": {"color": "rgba(17, 24, 39, 0.25)"}},
            increasing={"marker": {"color": GREEN_POSITIVE}},
            decreasing={"marker": {"color": BLUE_SECONDARY}},
            totals={"marker": {"color": NAVY_PRIMARY}},
            customdata=[
                [actual_total, actual_spend, actual_incr, actual_roas],
                [delta_total, delta_spend, None, None],
                [opt_total, opt_spend, opt_incr, opt_roas],
            ],
            hovertemplate=(
                "<b>%{x}</b><br><br>"
                "Total Revenue: %{customdata[0]:$,.0f}<br>"
                "Total Spend: %{customdata[1]:$,.0f}<br>"
                "Incremental Revenue: %{customdata[2]:$,.0f}<br>"
                "Incremental ROAS: %{customdata[3]:.2f}"
                "<extra></extra>"
            ),  # ✅ correct usage
        )
    )

    fig.update_layout(
        template="simple_white",
        height=420,
        margin=dict(l=40, r=20, t=70, b=40),
        title=dict(
            text="<b>Total Revenue Story (Incl. Baseline)</b>",
            x=0.01,
            xanchor="left",
        ),
        yaxis=dict(
            title="Annual Revenue / Response (USD)",
            gridcolor="rgba(0,0,0,0.06)",
        ),
        xaxis=dict(title=""),
        font=dict(family="Arial, system-ui", size=12, color=UI_TEXT),
    )

    return fig


def build_spend_dumbbell(results):
    df = _clean_results_df(results)
    d = df[df["Channel"].astype(str).str.upper() != "TOTAL"].copy()
    if d.empty:
        return go.Figure()

    d["Actual/Input Spend"] = d["Actual/Input Spend"].apply(_to_num)
    d["Optimized Spend"] = d["Optimized Spend"].apply(_to_num)
    d["Δ Spend (Abs)"] = d["Δ Spend (Abs)"].apply(_to_num)

    d = d.sort_values("Δ Spend (Abs)", key=lambda x: abs(x), ascending=False)

    fig = go.Figure()

    # connector lines
    for _, r in d.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[r["Actual/Input Spend"], r["Optimized Spend"]],
                y=[r["Channel"], r["Channel"]],
                mode="lines",
                line=dict(color="rgba(17, 24, 39, 0.20)", width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=d["Actual/Input Spend"],
            y=d["Channel"],
            mode="markers",
            marker=dict(size=10, color=NAVY_PRIMARY),
            name="Current",
            hovertemplate="<b>%{y}</b><br>Current: $%{x:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["Optimized Spend"],
            y=d["Channel"],
            mode="markers",
            marker=dict(size=10, color=GREEN_POSITIVE),
            name="Optimized",
            hovertemplate="<b>%{y}</b><br>Optimized: $%{x:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="simple_white",
        height=max(360, 22 * len(d) + 160),
        margin=dict(l=160, r=20, t=70, b=40),
        title=dict(text="<b>Spend Reallocation (Current vs Optimized)</b>", x=0.01, xanchor="left"),
        xaxis=dict(title="Annual Investment (USD)", gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(title="", automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        font=dict(family="Arial, system-ui", size=12, color=UI_TEXT),
    )

    return fig


def build_delta_spend_bar(results):
    df = _clean_results_df(results)
    d = df[df["Channel"].astype(str).str.upper() != "TOTAL"].copy()
    if d.empty:
        return go.Figure()

    d["Δ Spend (Abs)"] = d["Δ Spend (Abs)"].apply(_to_num)
    d = d.sort_values("Δ Spend (Abs)", ascending=True)

    colors = np.where(
        d["Δ Spend (Abs)"] >= 0,
        SKY_INTERACT,  # Fresh sky (accent-3)
        BLUE_SECONDARY,  # muted slate
    )

    fig = go.Figure(
        go.Bar(
            x=d["Δ Spend (Abs)"],
            y=d["Channel"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Δ Spend: $%{x:,.0f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="rgba(17,24,39,0.35)")

    fig.update_layout(
        template="simple_white",
        height=max(340, 20 * len(d) + 160),
        margin=dict(l=160, r=20, t=70, b=40),
        title=dict(text="<b>Change in Annual Investment by Channel</b>", x=0.01, xanchor="left"),
        xaxis=dict(title="Change in Annual Investment (USD)", gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(title="", automargin=True),
        font=dict(family="Arial, system-ui", size=12, color=UI_TEXT),
    )

    return fig

# def build_channel_actions(results):
#     df = pd.DataFrame(results)
#     if "Channel" not in df.columns:
#         df = df.reset_index().rename(columns={"index": "Channel"})

#     df = df[df["Channel"].astype(str).str.upper() != "TOTAL"].copy()
#     if df.empty:
#         return None

#     df["Δ Spend (%)"] = pd.to_numeric(df["Δ Spend (%)"], errors="coerce")
#     df["Optimized ROI"] = pd.to_numeric(df["Optimized ROI"], errors="coerce")

#     roas_median = df["Optimized ROI"].median()

#     # buckets = {
#     #     "Increase": df[(df["Δ Spend (%)"] > 0) & (df["Optimized ROI"] >= roas_median)],
#     #     "Optimize": df[(df["Δ Spend (%)"] < 0) & (df["Optimized ROI"] >= roas_median)],
#     #     "Maintain": df[df["Δ Spend (%)"].abs() <= 5],
#     #     "Deprioritize": df[(df["Δ Spend (%)"] < 0) & (df["Optimized ROI"] < roas_median)],
#     # }

#     cards = []
#     for title, subset in buckets.items():
#         if subset.empty:
#             continue
#         cards.append(
#             dbc.Col(
#                 html.Div(
#                     [
#                         html.Div(title, className="section-title"),
#                         html.Ul([html.Li(ch) for ch in subset["Channel"]]),
#                     ],
#                     className="chart-card",
#                 ),
#                 md=3,
#             )
#         )

#     return dbc.Row(cards, className="g-3")



def build_efficiency_quadrant(results):
    df = _clean_results_df(results)
    d = df[df["Channel"].astype(str).str.upper() != "TOTAL"].copy()
    if d.empty:
        return go.Figure()

    d["Optimized Spend"] = d["Optimized Spend"].apply(_to_num)
    d["Optimized ROI"] = pd.to_numeric(d["Optimized ROI"], errors="coerce")
    d["Optimized Response Metric"] = d["Optimized Response Metric"].apply(_to_num)

    x_med = float(np.nanmedian(d["Optimized Spend"])) if len(d) else 0.0
    y_med = float(np.nanmedian(d["Optimized ROI"])) if len(d) else 0.0

    size = d["Optimized Response Metric"].fillna(0.0).astype(float)
    size_scaled = 12 + 42 * (size / (size.max() if size.max() > 0 else 1.0))

    fig = go.Figure(
        go.Scatter(
            x=d["Optimized Spend"],
            y=d["Optimized ROI"],
            mode="markers+text",
            text=d["Channel"],
            textposition="top center",
            marker=dict(
                size=size_scaled,
                color=NAVY_PRIMARY,
                line=dict(color=NAVY_PRIMARY, width=2),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Optimized Spend: $%{x:,.0f}<br>"
                "Optimized ROAS: %{y:.2f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=x_med, line_width=1, line_dash="dash", line_color="rgba(17,24,39,0.35)")
    fig.add_hline(y=y_med, line_width=1, line_dash="dash", line_color="rgba(17,24,39,0.35)")

    fig.update_layout(
        template="simple_white",
        height=520,
        margin=dict(l=50, r=20, t=70, b=50),
        title=dict(text="<b>Relative Channel Efficiency vs Optimized Investment</b>", x=0.01, xanchor="left"),
        xaxis=dict(title="Optimized Annual Investment (USD)", gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(title="Optimized ROAS", gridcolor="rgba(0,0,0,0.06)", rangemode="tozero"),
        font=dict(family="Arial, system-ui", size=12, color=UI_TEXT),
    )

    return fig
# =================================================
# Visualization Helpers (Pack A)
# =================================================

def build_marginal_roas_rank(results):
    df = pd.DataFrame(results)
    df = df[df["Channel Type"] == "Optimized"]
    if "Channel" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Channel"})

    df = df[df["Channel"].astype(str).str.upper() != "TOTAL"].copy()
    if df.empty:
        return go.Figure()

    df["Optimized Spend"] = df["Optimized Spend"].apply(parse_currency)
    df["Δ Spend (%)"] = pd.to_numeric(df["Δ Spend (%)"], errors="coerce")

    df = df[
        df["Channel"].isin(data["s_curve_params"].keys())
        & df["Optimized Spend"].notna()
    ].copy()


    df["Marginal ROAS"] = df.apply(
        lambda r: marginal_roas_10pct(
            r["Channel"],
            r["Optimized Spend"],
            data["s_curve_params"],
        ),
        axis=1,
    )

    df = df.sort_values("Marginal ROAS", ascending=True)

    colors = np.where(
        df["Δ Spend (%)"] >= 0,
        NAVY_PRIMARY,   # Imperial Blue = increase
        BLUE_SECONDARY # Muted = decrease
    )

    fig = go.Figure(
        go.Bar(
            x=df["Marginal ROAS"],
            y=df["Channel"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Marginal ROAS: %{x:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="simple_white",
        height=max(340, 22 * len(df) + 160),
        margin=dict(l=160, r=20, t=60, b=40),
        title=dict(
            text="<b>Channel Efficiency at the Margin (Optimized)</b>",
            x=0.01,
            xanchor="left",
        ),
        xaxis=dict(title="Marginal ROAS (+10% Spend)"),
        yaxis=dict(title=""),
        font=dict(family="Arial, system-ui", size=12, color=UI_TEXT),
    )

    return fig


def get_baseline(d):
    """Return baseline as float whether constant is a number, list/tuple/array, or dict."""
    if not isinstance(d, dict) or "constant" not in d or d["constant"] is None:
        return 0.0

    c = d["constant"]

    if isinstance(c, (int, float, np.number)):
        return float(c)

    if isinstance(c, dict):
        return float(sum(v for v in c.values() if v is not None))

    if isinstance(c, (list, tuple, np.ndarray)):
        return float(sum(v for v in c if v is not None))

    try:
        return float(c)
    except Exception:
        return 0.0


def compute_kpis(df):

    def parse(x):
        return parse_currency(x) if isinstance(x, str) else float(x)

    total_actual_spend = df["Actual/Input Spend"].apply(parse).sum()
    total_opt_spend = df["Optimized Spend"].apply(parse).sum()
    total_actual_rev = df["Actual Response Metric"].apply(parse).sum()
    total_opt_rev = df["Optimized Response Metric"].apply(parse).sum()
    actual_roas = total_actual_rev / total_actual_spend if total_actual_spend > 0 else 0
    opt_roas = total_opt_rev / total_opt_spend if total_opt_spend > 0 else 0

    return {
        "actual_spend": total_actual_spend,
        "opt_spend": total_opt_spend,
        "actual_rev": total_actual_rev,
        "opt_rev": total_opt_rev,
        "incr_rev": total_opt_rev - total_actual_rev,
        "actual_roas": actual_roas,
        "opt_roas": opt_roas,
    }


# =================================================
# Brand tokens (Reynolds-led per PDFs)
# =================================================
# =================================================
# Approved Color Palette (STRICT)
# =================================================
NAVY_PRIMARY   = "#0e2961"
BLUE_SECONDARY = "#004e9d"
GOLD_HIGHLIGHT = "#e3a002"
SKY_INTERACT   = "#00aeea"
GREEN_POSITIVE = "#55ad43"
PURPLE_ACCENT  = "#583088"
PINK_ALERT     = "#e22380"

UI_BG = "#f3f4f6"
UI_BORDER = "#e5e7eb"
UI_TEXT = "#111827"
UI_MUTED = "#6b7280"


APP_TITLE = "MMM OPTIMIZER"
APP_SUBTITLE = "Marketing Mix Optimization - Demo"

# =================================================
# App
# =================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
server = app.server
app.title = "MMM Optimizer"

# =================================================
# UI Components
# =================================================
def kpi_card(title, main, sub=None, tooltip=None):
    return html.Div(
        className="kpi",
        children=[
            html.Div(title, className="kpi-label"),
            html.Div(main, className="kpi-value", title=tooltip),
            html.Div(sub, className="kpi-sub") if sub else None,
        ],
    )


def app_header():
    return html.Div(
        className="app-header",
        children=[
            html.Div(
                className="header-left",
                children=[
                    html.Div(APP_TITLE, className="app-title"),
                    html.Div(APP_SUBTITLE, className="app-subtitle"),
                ],
            ),
            # html.Img(
            #     src=app.get_asset_url("reynolds-logo.png"),
            #     className="reynolds-logo",
            #     alt="Reynolds",
            # ),
        ],
    )


def action_bar():
    # Sticky action bar: exports only
    return html.Div(
        className="action-bar",
        children=[
            html.Div(
                className="action-bar-right",
                children=[
                    dbc.Button(
                        "Download CSV",
                        id="download-csv-btn",
                        color="secondary",
                        size="sm",
                        className="btn-secondary",
                    ),
                    dbc.Button(
                        "Download XLSX",
                        id="download-btn",
                        color="secondary",
                        size="sm",
                        className="btn-secondary",
                        disabled=True,
                    ),
                ],
            ),
        ],
    )


def app_footer():
    return html.Footer(
        className="app-footer",
        children=[
            html.Div(
                "© 2025 Marketing Mix Optimization - Confidential",
                className="footer-left",
            ),
            html.Div("Created by Blend360", className="footer-right"),
        ],
    )


def channel_rows_optimize():
    rows = []
    for ch in channels:
        base = base_investment[ch]
        lb, ub = data["bounds_dict"][ch]
        rows.append(
            html.Tr(
                [
                    html.Td(ch, style={"fontWeight": 700, "textAlign": "left", "verticalAlign": "middle"}),
                    html.Td(
                        dbc.Switch(
                            id={"type": "include", "ch": ch},
                            value=True,
                            className="table-toggle",
                        ),
                        className="table-center",
                        style={"textAlign": "center", "verticalAlign": "middle"},
                    ),
                    html.Td(
                        dbc.Checkbox(
                            id={"type": "lock", "ch": ch},
                            value=False,
                            className="table-checkbox",
                        ),
                        className="table-center",
                        style={"textAlign": "center", "verticalAlign": "middle"},
                    ),
                    html.Td(
                        dcc.Input(
                            type="text",
                            value=format_currency(base),
                            disabled=True,
                            style={
                                "width": "100%",
                                "background": "transparent",
                                "border": "none",
                                "textAlign": "right",
                                "fontWeight": "700",
                                "color": NAVY_PRIMARY,
                                "cursor": "default",
                            },
                        ),
                        style={"textAlign": "right", "verticalAlign": "middle"},
                    ),
                    html.Td(
                        dcc.Input(
                            id={"type": "lb", "ch": ch},
                            type="number",
                            value=lb,
                            step=1,
                            style={"width": "100%", "textAlign": "center"},
                        ),
                        style={"textAlign": "center", "verticalAlign": "middle"},
                    ),
                    html.Td(
                        dcc.Input(
                            id={"type": "ub", "ch": ch},
                            type="number",
                            value=ub,
                            step=1,
                            style={"width": "100%", "textAlign": "center"},
                        ),
                        style={"textAlign": "center", "verticalAlign": "middle"},
                    ),
                    html.Td(
                        dcc.Input(
                            id={"type": "min", "ch": ch},
                            type="text",
                            value=format_currency(base * (1 + lb / 100)),
                            style={"width": "100%", "textAlign": "right"},
                        ),
                        style={"textAlign": "right", "verticalAlign": "middle"},
                    ),
                    html.Td(
                        dcc.Input(
                            id={"type": "max", "ch": ch},
                            type="text",
                            value=format_currency(base * (1 + ub / 100)),
                            style={"width": "100%", "textAlign": "right"},
                        ),
                        style={"textAlign": "right", "verticalAlign": "middle"},
                    ),

                ]
            )
        )
    return rows

def channel_rows_simulate():
    rows = []

    for ch in channels:
        base = base_investment[ch]

        rows.append(
            html.Tr(
                [
                    html.Td(ch, style={"fontWeight": 700, "textAlign": "left", "verticalAlign": "middle"}),

                    html.Td(
                        dbc.Switch(
                            id={"type": "sim-include", "ch": ch},
                            value=True,
                        ),
                        className="table-center",
                        style={"textAlign": "center", "verticalAlign": "middle"},
                    ),

                    html.Td(
                        fmt_money_full(base),
                        style={"textAlign": "right", "verticalAlign": "middle", "fontVariantNumeric": "tabular-nums"},
                        id={"type": "sim-current", "ch": ch},
                    ),

                    html.Td(
                        dcc.Slider(
                            id={"type": "tilt", "ch": ch},
                            min=-300,
                            max=300,
                            step=5,
                            value=0,
                            marks={
                                -300: "-300%",
                                -150: "-150%",
                                0: "0%",
                                150: "+150%",
                                300: "+300%",
                            },
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        style={"width": "260px", "verticalAlign": "middle"},
                    ),

                    html.Td(
                        fmt_money_full(base),
                        style={"fontWeight": 700, "textAlign": "right", "verticalAlign": "middle", "fontVariantNumeric": "tabular-nums"},
                        id={"type": "sim-new", "ch": ch},
                    ),

                    html.Td(
                        "$0",
                        style={"textAlign": "right", "verticalAlign": "middle", "fontVariantNumeric": "tabular-nums"},
                        id={"type": "sim-delta", "ch": ch},
                    ),
                ]
            )
        )

    return rows

@app.callback(
    Output({"type": "sim-new", "ch": ALL}, "children"),
    Output({"type": "sim-delta", "ch": ALL}, "children"),
    Input({"type": "tilt", "ch": ALL}, "value"),
    Input({"type": "sim-include", "ch": ALL}, "value"),
    State("run-mode", "value"),
)
def update_simulation_values(tilts, includes, run_mode):

    if run_mode != "simulate":
        raise PreventUpdate

    if tilts is None or includes is None:
        raise PreventUpdate


    new_vals = []
    delta_vals = []

    for ch, tilt, include in zip(channels, tilts, includes):
        base = base_investment[ch]

        if not include:
            new_vals.append(fmt_money_full(base))
            delta_vals.append("$0")
            continue

        pct = float(tilt or 0)
        new_spend = base * (1 + pct / 100)
        delta = new_spend - base

        new_vals.append(fmt_money_full(new_spend))
        delta_vals.append(
            f"{'+' if delta >= 0 else ''}{fmt_money_full(delta)}"
        )

    return new_vals, delta_vals



def incremental_channel_rows():
    rows = []

    for ch in INCREMENTAL_CHANNELS.keys():
        rows.append(
            html.Tr(
                [
                    # Channel
                    html.Td(
                        ch,
                        style={
                            "fontWeight": 600,
                            "textAlign": "left",
                            "verticalAlign": "middle",
                            "width": "45%",
                        },
                    ),

                    # Include toggle
                    html.Td(
                        html.Div(
                            dbc.Switch(
                                id={"type": "inc-include", "ch": ch},
                                value=False,
                            ),
                            style={
                                "display": "flex",
                                "justifyContent": "center",
                                "alignItems": "center",
                            },
                        ),
                        style={
                            "textAlign": "center",
                            "verticalAlign": "middle",
                            "width": "15%",
                        },
                    ),


                    # Scenario Investment
                    html.Td(
                        dcc.Input(
                            id={"type": "inc-spend", "ch": ch},
                            type="number",
                            placeholder="Enter spend",
                            step=1000,
                            disabled=True,
                            style={
                                "width": "160px",
                                "textAlign": "right",
                            },
                        ),
                        style={
                            "textAlign": "right",
                            "verticalAlign": "middle",
                            "width": "40%",
                        },
                    ),
                ]
            )
        )

    return rows



# =================================================
# Response Curves (Marginal ROAS +10%)
# =================================================
def response_annual_from_params(params, annual_spend):
    x_weekly = annual_spend / 52.0
    y_weekly = (
        params["L"] * (x_weekly ** params["alpha"])
        / (params["theta"] ** params["alpha"] + x_weekly ** params["alpha"] + 1e-8)
    )
    return y_weekly * 52.0


def marginal_roas_10pct(channel, annual_spend, s_curve_params):
    """
    Marginal ROAS with a +10% increase:
    (R(1.1S) - R(S)) / (0.1S)
    """
    if channel not in s_curve_params:
        return np.nan
    
    if annual_spend is None or annual_spend <= 0:
        return 0.0
    
    params = s_curve_params[channel]
    s0 = float(annual_spend)
    s1 = 1.1 * s0
    r0 = response_annual_from_params(params, s0)
    r1 = response_annual_from_params(params, s1)
    return (r1 - r0) / (s1 - s0)


def build_response_curve_fig(channel, s_curve_params, base_spend, optimized_spend, lb_pct=None, ub_pct=None):
    params = s_curve_params[channel]
    max_spend = max(base_spend, optimized_spend) * 1.35
    x_annual = np.linspace(0, max_spend, 120)

    y_annual = np.array([response_annual_from_params(params, s) for s in x_annual], dtype=float)
    roas = np.divide(y_annual, x_annual, out=np.zeros_like(y_annual), where=x_annual > 0)

    opt_resp_annual = response_annual_from_params(params, optimized_spend)
    opt_roas = (opt_resp_annual / optimized_spend) if optimized_spend > 0 else 0
    opt_mroas_10 = marginal_roas_10pct(channel, optimized_spend, s_curve_params)

    fig = go.Figure()
    # Feasible range shading based on bounds (optional)
    if (lb_pct is not None) and (ub_pct is not None):
        min_allowed = base_spend * (1 + float(lb_pct) / 100)
        max_allowed = base_spend * (1 + float(ub_pct) / 100)
        fig.add_vrect(
            x0=min_allowed,
            x1=max_allowed,
            fillcolor="rgba(0,174,234,0.18)",  # fresh-sky tint
            line_width=0,
            layer="below",
        )


    fig.add_trace(
        go.Scatter(
            x=x_annual,
            y=y_annual,
            mode="lines",
            line=dict(color="rgba(31, 41, 55, 0.75)", width=2),
            name="Annual Response",
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_annual,
            y=roas,
            mode="lines",
            line=dict(color=GOLD_HIGHLIGHT, width=2, dash="dot"),
            name="ROAS",
            yaxis="y2",
        )
    )

    fig.add_vline(x=base_spend, line_width=2, line_dash="dash", line_color=NAVY_PRIMARY)
    fig.add_vline(x=optimized_spend, line_width=4, line_color=GREEN_POSITIVE,layer="above",)

    fig.add_annotation(
        x=base_spend,
        y=0.96,
        xref="x",
        yref="paper",
        text="<b>Current</b>",
        showarrow=False,
        xanchor="right",
        font=dict(size=11, color=NAVY_PRIMARY),
    )

    fig.add_annotation(
        x=optimized_spend,
        y=0.7,
        xref="x",
        yref="paper",
        text="<b>Optimized</b>",
        showarrow=False,
        xanchor="left",
        font=dict(size=11, color=GREEN_POSITIVE),
    )

    fig.add_trace(
        go.Scatter(
            x=[optimized_spend],
            y=[opt_roas],
            mode="markers",
            marker=dict(size=9, color=GOLD_HIGHLIGHT),
            yaxis="y2",
            showlegend=False,
        )
    )

    fig.add_annotation(
        x=optimized_spend,
        y=opt_roas,
        xref="x",
        yref="y2",
        text=f"<b>Marginal ROAS (+10% Spend) {opt_mroas_10:.2f}</b>",
        showarrow=True,
        arrowhead=2,
        ax=110,
        ay=-45,
        font=dict(size=11, color=UI_TEXT),
        arrowcolor=GOLD_HIGHLIGHT,
    )

    fig.update_layout(
        title=dict(text=f"<b>{channel}</b>", x=0.01, xanchor="left"),
        height=330,
        margin=dict(l=60, r=60, t=75, b=45),
        template="simple_white",
        showlegend=False,
        xaxis=dict(
            title="Annual Investment",
            gridcolor="rgba(0, 0, 0, 0.06)",
            linecolor="rgba(0, 0, 0, 0.18)",
            tickcolor="rgba(0, 0, 0, 0.18)",
        ),
        yaxis=dict(
            title="Annual Response",
            gridcolor="rgba(0, 0, 0, 0.06)",
            linecolor="rgba(0, 0, 0, 0.18)",
            tickcolor="rgba(0, 0, 0, 0.18)",
        ),
        yaxis2=dict(
            title="ROAS",
            overlaying="y",
            side="right",
            rangemode="tozero",
            showgrid=False,
            linecolor="rgba(0, 0, 0, 0.18)",
            tickcolor="rgba(0, 0, 0, 0.18)",
        ),
        font=dict(family="Arial, system-ui", size=12, color=UI_TEXT),
    )
    return fig


response_curve_legend = html.Div(
    className="section",
    children=[
        html.Div("How to read the response curves", className="section-title"),
        html.Div(
            [
                html.Div(
                    [
                        html.Span("━", style={"color": "#1f2937", "fontWeight": 800}),
                        html.Span(" Response (left axis)", style={"marginLeft": "8px"}),
                    ],
                    className="mb-1",
                ),
                html.Div(
                    [
                        html.Span("⋯", style={"color": "#f97316", "fontWeight": 800}),
                        html.Span(" ROAS (right axis)", style={"marginLeft": "8px"}),
                    ],
                    className="mb-1",
                ),
                html.Div(
                    [
                        html.Span("│", style={"color": NAVY_PRIMARY, "fontWeight": 800}),
                        html.Span(" Dashed line = Current spend", style={"marginLeft": "8px"}),
                    ],
                    className="mb-1",
                ),
                html.Div(
                    [
                        html.Span("│", style={"color": GREEN_POSITIVE, "fontWeight": 800}),
                        html.Span(" Solid line = Optimized spend", style={"marginLeft": "8px"}),
                    ],
                    className="mb-1",
                ),
                html.Div(
                    [
                        html.Span("●", style={"color": "#f97316", "fontWeight": 800}),
                        html.Span(" Marginal ROAS (+10% Spend) at optimized spend", style={"marginLeft": "8px"}),
                    ]
                ),
            ],
            style={"marginTop": "6px", "color": UI_TEXT, "fontSize": "12px"},
        ),
    ],
)

# =================================================
# Layout
# =================================================
app.layout = html.Div(
    className="app-shell",
    children=[
        app_header(),
        action_bar(),

        # ---- STORES ----
        dcc.Store(id="optimizer-results"),
        dcc.Store(id="mmm-budget-value"),  # ✅ REQUIRED (numeric MMM budget)

        # =================================================
        # 1. MODE & OPTIMIZATION SETTINGS
        # =================================================
        html.Div(
            className="section",
            children=[
                html.Div("1. MODE & OPTIMIZATION OBJECTIVE", className="section-title"),
                html.Div(
                    "Define what you want the model to do before entering budgets or scenarios.",
                    className="comment",
                ),

                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label(
                                    "Run Mode",
                                    style={"fontSize": "12px", "fontWeight": 700},
                                ),
                                dbc.RadioItems(
                                    id="run-mode",
                                    options=[
                                        {"label": "Optimize", "value": "optimize"},
                                        {"label": "Simulate", "value": "simulate"},
                                    ],
                                    value="optimize",
                                    inline=True,
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.Label(
                                    "Optimization Objective",
                                    style={"fontSize": "12px", "fontWeight": 700},
                                ),
                                dbc.Select(
                                    id="goal",
                                    options=[
                                        {"label": "Maximize Response", "value": "forward"},
                                        {"label": "Minimize Investment", "value": "backward"},
                                    ],
                                    value=data.get("optimization_goal", "forward"),
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="g-2",
                ),
            ],
        ),

        # =================================================
        # 2. TOTAL MARKETING BUDGET & SCENARIOS
        # =================================================
        html.Div(
            id="section-budget",
            className="section",
            children=[
                html.Div("2. TOTAL MARKETING BUDGET & SCENARIOS", className="section-title"),
                html.Div(
                    id="budget-section-description",
                    className="comment",
                ),

                # ---------- TOTAL BUDGET (HIDDEN IN SIMULATION) ----------
                html.Div(
                    id="total-budget-block",
                    children=[
                        # hidden stubs — kept for callbacks
                        html.Div(
                            dcc.Input(id="budget-pct", type="number", value=0, step=1, style={"display": "none"}),
                            id="budget-pct-col",
                            style={"display": "none"},
                        ),

                        html.Label(
                            "Total Marketing Budget (USD)",
                            id="total-target-label",
                            style={"fontSize": "12px", "fontWeight": 700, "display": "block", "marginBottom": "6px"},
                        ),
                        # Inline: [−][Input][+]  ←→  Slider
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Button(
                                            "−", id="budget-minus", n_clicks=0,
                                            style={
                                                "width": "34px", "height": "36px", "flexShrink": 0,
                                                "border": "1px solid #ced4da", "borderRight": "none",
                                                "borderRadius": "4px 0 0 4px", "background": "#f8f9fa",
                                                "cursor": "pointer", "fontSize": "18px", "lineHeight": "1",
                                            },
                                        ),
                                        dcc.Input(
                                            id="total-target",
                                            type="text",
                                            debounce=True,
                                            value=format_currency(data.get("total_target", 0)),
                                            placeholder="e.g., 2,500,000",
                                            style={
                                                "width": "160px", "height": "36px", "flexShrink": 0,
                                                "border": "1px solid #ced4da", "borderRadius": "0",
                                                "padding": "4px 8px", "textAlign": "right",
                                            },
                                        ),
                                        html.Button(
                                            "+", id="budget-plus", n_clicks=0,
                                            style={
                                                "width": "34px", "height": "36px", "flexShrink": 0,
                                                "border": "1px solid #ced4da", "borderLeft": "none",
                                                "borderRadius": "0 4px 4px 0", "background": "#f8f9fa",
                                                "cursor": "pointer", "fontSize": "18px", "lineHeight": "1",
                                            },
                                        ),
                                    ],
                                    style={"display": "flex", "alignItems": "center", "flexShrink": 0},
                                ),
                                html.Div(
                                    dcc.Slider(
                                        id="budget-slider",
                                        min=-50, max=50, step=5, value=0,
                                        marks={i: f"{i:+d}%" if i != 0 else "0%" for i in range(-50, 51, 25)},
                                        tooltip={"placement": "bottom", "always_visible": False, "transform": "budgetSliderFmt"},
                                        className="budget-slider",
                                    ),
                                    style={"flex": "1", "minWidth": "180px", "paddingLeft": "20px", "paddingTop": "6px"},
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"},
                        ),
                        html.Hr(style={"margin": "14px 0 10px 0", "borderColor": "#e5e7eb"}),
                    ],
                ),

                # ---------- TWO-COLUMN: SCENARIO TABLE (left) + BUDGET SUMMARY (right) ----------
                html.Div(
                    id="scenario-block",
                    className="subsection",
                    children=[
                        html.Div(
                            "Incremental / Scenario Channels (Not Optimized)",
                            className="section-subtitle",
                        ),
                        html.Div(
                            "Scenario-only channels. These investments are simulated linearly "
                            "and applied on top of MMM results.",
                            className="comment",
                            style={"fontSize": "11px"},
                        ),
                    ],
                ),
                dbc.Row(
                    [
                        # Left column — table
                        dbc.Col(
                            [
                                html.Div(
                                    dbc.Table(
                                        [
                                            html.Thead(
                                                html.Tr(
                                                    [
                                                        html.Th("Channel", style={"textAlign": "left"}),
                                                        html.Th("Include", style={"textAlign": "center"}),
                                                        html.Th(
                                                            "Scenario Investment (USD)",
                                                            style={"textAlign": "right"},
                                                        ),
                                                    ]
                                                )
                                            ),
                                            html.Tbody(incremental_channel_rows()),
                                        ],
                                        bordered=False,
                                        size="sm",
                                        className="mt-2",
                                    ),
                                    style={
                                        "backgroundColor": "rgba(100,116,139,0.04)",
                                        "border": "1px dashed #cbd5e1",
                                        "borderRadius": "6px",
                                        "padding": "6px 8px",
                                    },
                                ),
                                html.Div(
                                    "Tip: These channels are scenario levers and are never optimized.",
                                    className="comment",
                                    style={"fontSize": "10.5px", "marginTop": "6px"},
                                ),
                            ],
                            md=8,
                            style={"paddingRight": "16px"},
                        ),

                        # Right column — Unified Budget Summary panel
                        dbc.Col(
                            html.Div(
                                id="mmm-budget-display",
                                style={"display": "none"},
                            ),
                            md=4,
                        ),
                    ],
                    className="g-0 mt-1",
                    align="start",
                ),

                # hidden stub — kept for callback compatibility
                html.Div(id="budget-comparison", style={"display": "none"}),
            ],
        ),

        # =================================================
        # 3. CHANNEL INVESTMENT SETTINGS
        # =================================================
        html.Div(
            className="section",
            children=[
                html.Div("3. CHANNEL INVESTMENT SETTINGS", className="section-title"),
                html.Div(
                    "Set bounds when optimizing, or simulate spend changes by channel.",
                    className="comment",
                ),

                # ---------- OPTIMIZE ----------
                html.Div(
                    id="bounds-table-optimize",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Global Lower Bound %", style={"fontSize": "12px", "fontWeight": 700, "display": "block", "marginBottom": "4px"}),
                                        dcc.Input(id="global-lb", type="number", value=-20, style={"width": "100%", "textAlign": "center"}),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Global Upper Bound %", style={"fontSize": "12px", "fontWeight": 700, "display": "block", "marginBottom": "4px"}),
                                        dcc.Input(id="global-ub", type="number", value=20, style={"width": "100%", "textAlign": "center"}),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "APPLY GLOBAL BOUNDS",
                                        id="apply-global-bounds",
                                        color="secondary",
                                        className="btn-secondary",
                                        style={"height": "38px", "width": "100%"},
                                    ),
                                    md=4,
                                    className="d-flex align-items-end",
                                ),
                            ],
                            className="g-2 mt-1",
                        ),

                        html.Div(id="global-bounds-msg", className="mt-2"),

                        dbc.Table(
                            [
                                html.Thead(
                                    html.Tr(
                                        [
                                            html.Th("Channel", style={"textAlign": "left"}),
                                            html.Th("Include", style={"textAlign": "center"}),
                                            html.Th("Lock", style={"textAlign": "center"}),
                                            html.Th("Current Investment (USD)", style={"textAlign": "right"}),
                                            html.Th("Lower Bound %", style={"textAlign": "center"}),
                                            html.Th("Upper Bound %", style={"textAlign": "center"}),
                                            html.Th("Min Investment (USD)", style={"textAlign": "right"}),
                                            html.Th("Max Investment (USD)", style={"textAlign": "right"}),
                                        ]
                                    )
                                ),
                                html.Tbody(channel_rows_optimize()),
                            ],
                            bordered=True,
                            size="sm",
                            className="mt-2",
                        ),

                        html.Div(
                            f"Total current investment: {fmt_money_full(sum(base_investment.values()))}",
                            style={
                                "textAlign": "right",
                                "fontSize": "12px",
                                "fontWeight": 700,
                                "color": NAVY_PRIMARY,
                                "marginTop": "4px",
                            },
                        ),

                        html.Div(id="bounds-feasibility", className="mt-2"),
                    ],
                ),

                # ---------- SIMULATE ----------
                html.Div(
                    id="bounds-table-simulate",
                    style={"display": "none"},
                    children=[
                        html.Div(
                            "Simulate % changes to current spend. No optimization or bounds are applied.",
                            className="comment",
                            style={"fontSize": "11px"},
                        ),
                        dbc.Table(
                            [
                                html.Thead(
                                    html.Tr(
                                        [
                                            html.Th("Channel", style={"textAlign": "left"}),
                                            html.Th("Include", style={"textAlign": "center"}),
                                            html.Th("Current Investment", style={"textAlign": "right"}),
                                            html.Th("Simulation Shift %", style={"textAlign": "center"}),
                                            html.Th("New Investment", style={"textAlign": "right"}),
                                            html.Th("Δ Investment", style={"textAlign": "right"}),
                                        ]
                                    )
                                ),
                                html.Tbody(channel_rows_simulate()),
                            ],
                            bordered=True,
                            size="sm",
                            className="mt-2",
                        ),
                    ],
                ),

            ],
        ),
        # =================================================
        # 4. RESULTS
        # =================================================
        html.Div(
            className="section",
            children=[
                html.Div("4. RESULTS", className="section-title"),
                html.Div(
                    "Review KPIs, reallocation summary, and detailed diagnostics.",
                    className="comment",
                ),

                # Status message
                html.Div(id="msg", className="mt-2"),

                # KPI cards
                html.Div(id="kpis", className="mt-3"),

                # Reallocation summary text
                html.Div(id="realloc-summary", className="mt-2"),

                # -----------------------------
                # Visual Insights
                # -----------------------------
                html.Div(
                    className="viz-wrap",
                    children=[
                        html.Div("VISUAL INSIGHTS", className="section-title"),
                        html.Div(
                            "Executive story + channel diagnostics.",
                            className="comment",
                        ),

                        dbc.Tabs(
                            children=[
                                dbc.Tab(
                                    label="Story",
                                    children=[
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div(id="viz-waterfall"), md=6),
                                                dbc.Col(html.Div(id="viz-dumbbell"), md=6),
                                            ],
                                            className="g-3 mt-2",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div(id="viz-mroas"), md=6),
                                            ],
                                            className="g-3 mt-2",
                                        ),
                                    ],
                                ),

                                dbc.Tab(
                                    label="Channel Shifts",
                                    children=[
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div(id="viz-delta"), md=6),
                                                dbc.Col(html.Div(id="viz-quadrant"), md=6),
                                            ],
                                            className="g-3 mt-2",
                                        ),
                                    ],
                                ),

                                dbc.Tab(
                                    label="Table",
                                    children=[
                                        html.Div(id="table", className="mt-2"),
                                    ],
                                ),

                                dbc.Tab(
                                    label="Response Curves",
                                    children=[
                                        html.Div(id="response-curves", className="mt-2"),
                                    ],
                                ),
                            ]
                        ),
                    ],
                ),

                # Downloads (targets only)
                dcc.Download(id="download-results"),
                dcc.Download(id="download-results-csv"),
            ],
        ),

        # ---- STICKY RUN BAR ----
        html.Div(
            className="sticky-run-bar",
            children=[
                # Left: budget context
                html.Div(
                    id="sticky-bar-budget-info",
                    className="sticky-bar-left",
                ),
                # Centre: optimization result status
                html.Div(
                    id="run-status",
                    className="sticky-bar-status",
                ),
                # Right: reset + primary action
                html.Div(
                    [
                        dbc.Button(
                            "RESET",
                            id="reset-btn-sticky",
                            color="secondary",
                            className="btn-secondary",
                            n_clicks=0,
                            title="Reset all inputs to defaults",
                            style={"height": "40px", "minWidth": "90px", "fontSize": "13px", "fontWeight": 700},
                        ),
                        dbc.Button(
                            "RUN OPTIMIZATION",
                            id="run",
                            color="primary",
                            className="btn-primary sticky-run-btn",
                            n_clicks=0,
                            title="Run optimization with current inputs",
                        ),
                    ],
                    style={"display": "flex", "gap": "10px", "flexShrink": 0},
                ),
            ],
        ),

        app_footer(),
    ],
)

@app.callback(
    Output("budget-section-description", "children"),
    Input("run-mode", "value"),
    Input("goal", "value"),
)
def update_budget_description(run_mode, goal):

    if run_mode == "simulate":
        return (
            "Simulation mode: adjust MMM and scenario channel spends to explore outcomes. "
            "No optimization or total budget constraints are applied."
        )

    if (goal or "").lower() == "backward":
        return (
            "Define the target total revenue or response level. "
            "The model will minimize total investment while meeting this target."
        )

    return (
        "Define the total marketing budget and allocate any scenario / incremental investments. "
        "The remaining budget will be used for MMM optimization."
    )


@app.callback(
    Output("total-budget-block", "style"),
    Input("run-mode", "value"),
)
def toggle_total_budget(run_mode):

    # 🚫 Simulation → no total budget
    if run_mode == "simulate":
        return {"display": "none"}

    return {"display": "block"}

# Callback to enable/disable spend input
@app.callback(
    Output({"type": "inc-spend", "ch": ALL}, "disabled"),
    Output({"type": "inc-spend", "ch": ALL}, "value"),
    Input({"type": "inc-include", "ch": ALL}, "value"),
    State({"type": "inc-spend", "ch": ALL}, "value"),
)
def toggle_incremental_spend_inputs(include_vals, spend_vals):
    if not include_vals:
        raise PreventUpdate

    disabled = []
    values = []

    for include, current_val in zip(include_vals, spend_vals):
        if include:
            disabled.append(False)
            values.append(current_val or 0)  # keep user-entered value
        else:
            disabled.append(True)
            values.append(None)                 # 🔑 force reset when excluded

    return disabled, values


@app.callback(
    Output("total-target-label", "children"),
    Output("total-target", "placeholder"),
    Output("total-target", "value", allow_duplicate=True),
    Input("goal", "value"),
    State("total-target", "value"),
    prevent_initial_call="initial_duplicate",
)
def update_total_target_label(goal, current_value):
    default_spend    = format_currency(data.get("total_target", 0))
    default_response = format_currency(
        sum(data["constant"])
        + sum(
            sum(
                (data["s_curve_params"][ch]["L"]
                 * (s * p) ** data["s_curve_params"][ch]["alpha"]
                 / ((s * p) ** data["s_curve_params"][ch]["alpha"]
                    + data["s_curve_params"][ch]["theta"] ** data["s_curve_params"][ch]["alpha"]
                    + 1e-8))
                for p in data["proportion"][ch]
            )
            for ch, s in data["spends"].items()
            if ch in data["s_curve_params"]
        )
    )

    if goal == "backward":
        return (
            "Target Total Revenue / Response",
            "e.g., 148,000,000",
            default_response,
        )
    return (
        "Total Marketing Budget (USD)",
        "e.g., 2,500,000",
        default_spend,
    )

# =================================================
# Budget % ↔ USD sync (forward mode only)
# =================================================
_BASE_SPEND = float(sum(data["spends"].values()))

@app.callback(
    Output("total-target",  "value", allow_duplicate=True),
    Output("budget-pct",    "value", allow_duplicate=True),
    Output("budget-slider", "value", allow_duplicate=True),
    Input("total-target", "value"),
    Input("budget-pct", "value"),
    State("goal", "value"),
    prevent_initial_call=True,
)
def sync_budget_pct(dollar_val, pct_val, goal):
    if goal == "backward":
        raise PreventUpdate

    trigger = ctx.triggered_id

    if trigger == "budget-pct" and pct_val is not None:
        new_dollar = _BASE_SPEND * (1 + float(pct_val) / 100)
        slider_val = max(-50, min(50, round(float(pct_val) / 5) * 5))
        return format_currency(new_dollar), pct_val, slider_val

    if trigger == "total-target" and dollar_val not in (None, ""):
        parsed = parse_currency(dollar_val)
        if parsed is not None and _BASE_SPEND > 0:
            # Clamp to ±50% of base spend
            parsed = max(_BASE_SPEND * 0.5, min(_BASE_SPEND * 1.5, parsed))
            new_pct = (parsed / _BASE_SPEND - 1) * 100
            # Leave slider where it is — slider only moves when dragged
            return format_currency(parsed), round(new_pct, 1), no_update

    raise PreventUpdate


@app.callback(
    Output("total-target",   "value", allow_duplicate=True),
    Output("budget-pct",     "value", allow_duplicate=True),
    Output("budget-slider",  "value", allow_duplicate=True),
    Input("budget-minus", "n_clicks"),
    Input("budget-plus",  "n_clicks"),
    State("total-target", "value"),
    prevent_initial_call=True,
)
def budget_step_buttons(_minus, _plus, current_val):
    trigger = ctx.triggered_id
    parsed = parse_currency(current_val) if current_val else _BASE_SPEND
    if parsed is None:
        parsed = _BASE_SPEND
    delta = -1000 if trigger == "budget-minus" else 1000
    # Clamp to ±50% of base spend
    new_val = max(_BASE_SPEND * 0.5, min(_BASE_SPEND * 1.5, parsed + delta))
    new_pct = (new_val / _BASE_SPEND - 1) * 100 if _BASE_SPEND > 0 else 0
    # Leave slider where it is — slider only moves when dragged
    return format_currency(new_val), round(new_pct, 1), no_update


@app.callback(
    Output("total-target",  "value", allow_duplicate=True),
    Output("budget-pct",    "value", allow_duplicate=True),
    Input("budget-slider",  "value"),
    prevent_initial_call=True,
)
def sync_budget_slider(slider_val):
    if slider_val is None:
        raise PreventUpdate
    new_dollar = _BASE_SPEND * (1 + float(slider_val) / 100)
    return format_currency(new_dollar), slider_val


@app.callback(
    Output("budget-pct-col", "style"),
    Input("goal", "value"),
)
def toggle_budget_pct_col(goal):
    return {"display": "none"}


@app.callback(
    Output("scenario-block", "style"),
    Input("run-mode", "value"),
    Input("goal", "value"),
)
def toggle_incremental_channels(run_mode, goal):

    # 🚫 Backward optimization → no scenarios
    if run_mode == "optimize" and (goal or "").lower() == "backward":
        return {"display": "none"}

    # ✅ Show in BOTH optimize-forward and simulate
    return {"display": "block"}


@app.callback(
    Output("section-budget", "style"),
    Input("goal", "value"),
)
def toggle_budget_section(goal):
    return {"display": "block"}


_ORIGINAL_MMM_BUDGET = float(data.get("total_target", 0))

@app.callback(
    Output("mmm-budget-display",    "children"),
    Output("mmm-budget-value",      "data"),
    Output("budget-comparison",     "children"),
    Output("budget-comparison",     "style"),
    Output("sticky-bar-budget-info","children"),
    Input("goal",       "value"),
    Input("run-mode",   "value"),
    Input("total-target", "value"),
    Input({"type": "inc-include", "ch": ALL}, "value"),
    Input({"type": "inc-spend",   "ch": ALL}, "value"),
)
def update_remaining_mmm_budget(goal, run_mode, total_target, inc_include, inc_spend):

    hidden = {"display": "none"}

    # No budget logic in simulate OR backward mode
    if run_mode == "simulate" or (goal or "").lower() == "backward":
        return None, None, None, hidden, None

    total_budget = parse_currency(total_target) or 0.0

    incremental_total = 0.0
    for include, spend in zip(inc_include, inc_spend):
        if include and spend not in (None, ""):
            v = parse_currency(spend)
            if v is not None:
                incremental_total += v

    mmm_budget = total_budget - incremental_total
    over_budget = incremental_total > total_budget

    # Clamp to 0 for downstream budget-value usage
    remaining = max(mmm_budget, 0.0)

    def brow(label, value, label_style=None, value_style=None):
        ls = {"textAlign": "left", "padding": "4px 16px 4px 0",
              "color": "#374151", "whiteSpace": "nowrap", "fontSize": "13px"}
        vs = {"textAlign": "right", "padding": "4px 0",
              "fontWeight": 700, "whiteSpace": "nowrap", "fontSize": "13px"}
        if label_style:
            ls.update(label_style)
        if value_style:
            vs.update(value_style)
        return html.Tr([html.Td(label, style=ls), html.Td(value, style=vs)])

    breakdown_rows = [
        brow(
            "Total Budget:",
            fmt_money_full(total_budget),
            label_style={"fontWeight": 600},
        ),
        brow(
            "− Incremental Channels:",
            f"− {fmt_money_full(incremental_total)}" if incremental_total > 0 else fmt_money_full(0),
            value_style={"color": "#dc2626" if incremental_total > 0 else "#374151"},
        ),
    ]

    separator_row = html.Tr([
        html.Td(
            colSpan=2,
            style={
                "borderTop": "2px solid #94a3b8",
                "padding": "2px 0",
            },
        )
    ])

    mmm_value_color = "#dc2626" if over_budget else "#16a34a"
    mmm_label_style = {"fontWeight": 700, "color": "#1e293b", "fontSize": "13px"}
    mmm_value_style = {"fontWeight": 700, "fontSize": "13px", "color": mmm_value_color}

    breakdown_rows.append(separator_row)
    breakdown_rows.append(
        brow(
            "MMM Optimization Budget:",
            fmt_money_full(mmm_budget),
            label_style=mmm_label_style,
            value_style=mmm_value_style,
        )
    )

    warning_block = html.Div(
        "⚠ Incremental spend exceeds total budget",
        style={
            "color": "#dc2626", "fontWeight": 600, "fontSize": "12px",
            "marginTop": "6px", "padding": "6px 10px",
            "background": "#fef2f2", "border": "1px solid #fca5a5",
            "borderRadius": "4px",
        },
    ) if over_budget else None

    # Budget vs model baseline
    orig = _ORIGINAL_MMM_BUDGET
    delta = remaining - orig
    delta_pct = (delta / orig * 100) if orig else 0.0
    sign = "+" if delta >= 0 else ""
    delta_color = "#16a34a" if delta >= 0 else "#dc2626"

    def srow(label, value_el, label_extra=None, value_extra=None):
        ls = {"textAlign": "left", "padding": "4px 12px 4px 0",
              "color": "#374151", "whiteSpace": "nowrap", "fontSize": "12px"}
        vs = {"textAlign": "right", "padding": "4px 0",
              "fontWeight": 700, "whiteSpace": "nowrap", "fontSize": "12px"}
        if label_extra:
            ls.update(label_extra)
        if value_extra:
            vs.update(value_extra)
        return html.Tr([html.Td(label, style=ls), html.Td(value_el, style=vs)])

    unified_panel = html.Div(
        [
            # Panel title
            html.Div("Budget Summary", style={
                "fontSize": "11px", "fontWeight": 700, "color": "#6b7280",
                "textTransform": "uppercase", "letterSpacing": "0.05em",
                "marginBottom": "10px",
            }),

            # ── Section 1: Budget Breakdown ──
            html.Div("Budget Breakdown", style={
                "fontSize": "10px", "fontWeight": 700, "color": "#94a3b8",
                "textTransform": "uppercase", "letterSpacing": "0.04em",
                "marginBottom": "4px",
            }),
            html.Table(
                [
                    srow(
                        "Total Budget:",
                        fmt_money_full(total_budget),
                        label_extra={"fontWeight": 600},
                    ),
                    srow(
                        "− Incremental Channels:",
                        f"− {fmt_money_full(incremental_total)}" if incremental_total > 0 else fmt_money_full(0),
                        value_extra={"color": "#dc2626" if incremental_total > 0 else "#64748b"},
                    ),
                    # Separator row
                    html.Tr([html.Td(
                        colSpan=2,
                        style={"borderTop": "2px solid #94a3b8", "padding": "2px 0"},
                    )]),
                    srow(
                        "MMM Optimization Budget:",
                        fmt_money_full(mmm_budget),
                        label_extra={"fontWeight": 700, "color": "#1e293b"},
                        value_extra={"color": "#dc2626" if over_budget else "#16a34a", "fontSize": "13px"},
                    ),
                ],
                style={"borderCollapse": "collapse", "width": "100%"},
            ),
            # Over-budget warning
            html.Div(
                "⚠ Incremental spend exceeds total budget",
                style={
                    "color": "#dc2626", "fontWeight": 600, "fontSize": "11px",
                    "marginTop": "5px", "padding": "5px 8px",
                    "background": "#fef2f2", "border": "1px solid #fca5a5",
                    "borderRadius": "4px",
                },
            ) if over_budget else None,

            # Divider between sections
            html.Hr(style={"margin": "10px 0", "borderColor": "#e2e8f0"}),

            # ── Section 2: vs Model Baseline ──
            html.Div("vs. Model Baseline", style={
                "fontSize": "10px", "fontWeight": 700, "color": "#94a3b8",
                "textTransform": "uppercase", "letterSpacing": "0.04em",
                "marginBottom": "4px",
            }),
            html.Table(
                [
                    srow("Original MMM Budget:", html.Span(fmt_money_full(orig))),
                    srow("Current MMM Budget:", html.Span(fmt_money_full(remaining))),
                    srow(
                        "Δ vs Model:",
                        html.Span(
                            f"{sign}{fmt_money_full(delta)}  ({sign}{delta_pct:.1f}%)",
                            style={"color": delta_color},
                        ),
                    ),
                ],
                style={"borderCollapse": "collapse", "width": "100%"},
            ),
        ],
        style={
            "background": "#f8fafc",
            "border": "1px solid #e2e8f0",
            "borderRadius": "6px",
            "padding": "12px 14px",
        },
    )

    # Sticky bar left-side content
    sticky_info = html.Div(
        [
            html.Div(
                [
                    html.Span("MMM Budget  ", className="sticky-bar-label"),
                    html.Span(
                        fmt_money_full(mmm_budget),
                        className="sticky-bar-value" + (" sticky-bar-value--warn" if over_budget else ""),
                    ),
                ],
                className="sticky-bar-stat",
            ),
            html.Div(className="sticky-bar-divider"),
            html.Div(
                [
                    html.Span("Δ vs Model  ", className="sticky-bar-label"),
                    html.Span(
                        f"{sign}{fmt_money_full(delta)}  ({sign}{delta_pct:.1f}%)",
                        className="sticky-bar-delta",
                        style={"color": delta_color},
                    ),
                ],
                className="sticky-bar-stat",
            ),
        ],
        style={"display": "flex", "alignItems": "center", "gap": "0"},
    )

    return (
        unified_panel,
        remaining,
        None,               # budget-comparison stub — no longer used
        {"display": "none"},
        sticky_info,
    )

@app.callback(
    Output("goal", "disabled"),
    Input("run-mode", "value"),
)
def disable_goal_in_simulation(run_mode):
    return run_mode == "simulate"


@app.callback(
    Output("run", "children"),
    Output("run", "disabled"),
    Input("run-mode", "value"),
)
def update_run_label(run_mode):
    if run_mode == "simulate":
        return "SIMULATION MODE", True
    return "RUN OPTIMIZATION", False


# =================================================
# Sync LB/UB ↔ Min/Max + Apply Global Bounds
# =================================================
@app.callback(
    Output({"type": "lb", "ch": ALL}, "value"),
    Output({"type": "ub", "ch": ALL}, "value"),
    Output({"type": "min", "ch": ALL}, "value"),
    Output({"type": "max", "ch": ALL}, "value"),
    Output("global-bounds-msg", "children"),
    Input({"type": "lb", "ch": ALL}, "value"),
    Input({"type": "ub", "ch": ALL}, "value"),
    Input({"type": "min", "ch": ALL}, "value"),
    Input({"type": "max", "ch": ALL}, "value"),
    Input("apply-global-bounds", "n_clicks"),
    State("global-lb", "value"),
    State("global-ub", "value"),
)
def sync_bounds(lb_vals, ub_vals, min_vals, max_vals, n_apply, global_lb, global_ub):
    trigger = ctx.triggered_id or {}

    if trigger == "apply-global-bounds":
        new_lb = [float(global_lb)] * len(channels) if global_lb is not None else [float(v) for v in lb_vals]
        new_ub = [float(global_ub)] * len(channels) if global_ub is not None else [float(v) for v in ub_vals]

        new_min, new_max = [], []
        for i, ch in enumerate(channels):
            base = base_investment[ch]
            min_v = base * (1 + new_lb[i] / 100)
            max_v = base * (1 + new_ub[i] / 100)
            new_min.append(format_currency(min_v))
            new_max.append(format_currency(max_v))

        msg = html.Div("Applied global bounds to all channels.", className="status-success")
        return new_lb, new_ub, new_min, new_max, msg

    new_lb, new_ub, new_min, new_max = [], [], [], []
    for i, ch in enumerate(channels):
        base = base_investment[ch]
        lb_raw = lb_vals[i]
        ub_raw = ub_vals[i]

        lb = float(lb_raw) if lb_raw not in [None, ""] else 0.0
        ub = float(ub_raw) if ub_raw not in [None, ""] else 0.0


        min_v = parse_currency(min_vals[i])
        max_v = parse_currency(max_vals[i])

        if isinstance(trigger, dict) and trigger.get("type") == "min" and min_v is not None:
            lb = (min_v / base - 1) * 100
        elif isinstance(trigger, dict) and trigger.get("type") == "max" and max_v is not None:
            ub = (max_v / base - 1) * 100

        min_calc = base * (1 + lb / 100)
        max_calc = base * (1 + ub / 100)



        new_lb.append(round(lb, 2))
        new_ub.append(round(ub, 2))
        new_min.append(format_currency(min_calc))
        new_max.append(format_currency(max_calc))

    return new_lb, new_ub, new_min, new_max, None

@app.callback(
    Output("bounds-feasibility", "children"),
    Input("run-mode", "value"),
    Input("goal", "value"),
    Input("mmm-budget-value", "data"),
    Input({"type": "include", "ch": ALL}, "value"),   # ✅ ADD
    Input({"type": "lock", "ch": ALL}, "value"),      # ✅ ADD
    Input({"type": "lb", "ch": ALL}, "value"),
    Input({"type": "ub", "ch": ALL}, "value"),
)
def show_bounds_feasibility(run_mode, goal, mmm_budget, include_vals, lock_vals, lb_vals, ub_vals):

    if run_mode == "simulate":
        return None

    # Bounds feasibility only applies to forward optimization
    if (goal or "").lower() == "backward":
        return None

    if mmm_budget is None:
        raise PreventUpdate

    mmm_budget = float(mmm_budget)

    min_total = 0.0
    max_total = 0.0

    for ch, include, lock, lb, ub in zip(
        channels, include_vals, lock_vals, lb_vals, ub_vals
    ):
        # ❌ Excluded channels do NOT count
        if not include:
            continue

        base = base_investment[ch]

        # 🔒 Locked → fixed spend
        if lock:
            min_total += base
            max_total += base
            continue

        lb = float(lb) if lb is not None else 0.0
        ub = float(ub) if ub is not None else 0.0

        min_total += base * (1 + lb / 100)
        max_total += base * (1 + ub / 100)

    if min_total <= mmm_budget <= max_total:
        return html.Div(
            f"Bounds feasible ✔  MMM budget {fmt_money_full(mmm_budget)} "
            f"is within [{fmt_money_full(min_total)} – {fmt_money_full(max_total)}]",
            className="status-success",
        )

    return html.Div(
        f"⚠ Budget outside bounds. Feasible range: "
        f"{fmt_money_full(min_total)} – {fmt_money_full(max_total)}. "
        f"MMM budget: {fmt_money_full(mmm_budget)}.",
        className="status-danger",
    )


# =================================================
# Run optimizer (bottom button only)
# =================================================
# =================================================
# Reset callback
# =================================================
@app.callback(
    Output("run-mode",                        "value"),
    Output("goal",                            "value"),
    Output("total-target",                    "value",    allow_duplicate=True),
    Output("budget-pct",                      "value",    allow_duplicate=True),
    Output("budget-slider",                   "value",    allow_duplicate=True),
    Output("global-lb",                       "value"),
    Output("global-ub",                       "value"),
    Output({"type": "include",    "ch": ALL}, "value",    allow_duplicate=True),
    Output({"type": "lock",       "ch": ALL}, "value",    allow_duplicate=True),
    Output({"type": "lb",         "ch": ALL}, "value",    allow_duplicate=True),
    Output({"type": "ub",         "ch": ALL}, "value",    allow_duplicate=True),
    Output({"type": "min",        "ch": ALL}, "value",    allow_duplicate=True),
    Output({"type": "max",        "ch": ALL}, "value",    allow_duplicate=True),
    Output({"type": "tilt",       "ch": ALL}, "value",    allow_duplicate=True),
    Output({"type": "inc-include","ch": ALL}, "value",    allow_duplicate=True),
    Output({"type": "inc-spend",  "ch": ALL}, "value",    allow_duplicate=True),
    Output("optimizer-results",               "data",     allow_duplicate=True),
    Output("msg",                             "children", allow_duplicate=True),
    Output("run-status",                      "children", allow_duplicate=True),
    Input("reset-btn-sticky", "n_clicks"),
    prevent_initial_call=True,
)
def reset_all(_):
    n_ch     = len(channels)
    n_inc    = len(INCREMENTAL_CHANNELS)
    def_lb   = -20.0
    def_ub   = 20.0
    def_mins = [format_currency(base_investment[ch] * (1 + def_lb / 100)) for ch in channels]
    def_maxs = [format_currency(base_investment[ch] * (1 + def_ub / 100)) for ch in channels]
    return (
        "optimize",
        "forward",
        format_currency(data.get("total_target", 0)),
        0,
        0,
        def_lb,
        def_ub,
        [True]  * n_ch,
        [False] * n_ch,
        [def_lb] * n_ch,
        [def_ub] * n_ch,
        def_mins,
        def_maxs,
        [0] * n_ch,
        [False] * n_inc,   # inc-include → all off
        [None]  * n_inc,   # inc-spend   → cleared
        None,              # optimizer-results
        None,              # msg
        None,              # run-status
    )


@app.callback(
    Output("optimizer-results", "data"),
    Output("msg", "children"),
    Output("download-btn", "disabled"),
    Output("run-status", "children"),
    Input("run", "n_clicks"),
    State("goal", "value"),
    State("total-target", "value"),
    State("run-mode", "value"),
    State({"type": "include", "ch": ALL}, "value"),
    State({"type": "lock", "ch": ALL}, "value"),
    State({"type": "lb", "ch": ALL}, "value"),
    State({"type": "ub", "ch": ALL}, "value"),
    State({"type": "inc-include", "ch": ALL}, "value"),   # ✅ ADD
    State({"type": "inc-spend", "ch": ALL}, "value"),     # ✅ ADD
    prevent_initial_call=True,
)
def run_optimizer(n_clicks, goal, total_target, run_mode, include_vals, lock_vals, lb_vals, ub_vals, inc_include_vals, inc_spend_vals, ):
    if run_mode == "simulate":
        return (
            None,
            html.Div(
                "Simulation mode is active. Adjust sliders to explore spend changes. "
                "No optimization or budget constraints are applied.",
                className="status-info",
            ),
            True,
            None,
        )


    total_target_val = parse_currency(total_target)
    # --- NEW: total marketing budget logic ---
    total_marketing_budget = total_target_val or 0.0

    incremental_spends = {}
    inc_channels = list(INCREMENTAL_CHANNELS.keys())

    if goal == "backward":
        incremental_spends = {}


    for i, ch in enumerate(inc_channels):
        include = bool(inc_include_vals[i]) if inc_include_vals else False
        spend_val = parse_currency(inc_spend_vals[i]) if inc_spend_vals else None

        if include and spend_val is not None:
            incremental_spends[ch] = float(spend_val)

    total_incremental = sum(incremental_spends.values())
    if goal == "forward":
        mmm_budget = max(total_marketing_budget - total_incremental, 0.0)
    else:
        mmm_budget = total_marketing_budget  # interpreted as response target



    bounds = {}
    invalid_rows = []
    for i, ch in enumerate(channels):
        lb_raw = lb_vals[i]
        ub_raw = ub_vals[i]
        lb = float(lb_raw) if lb_raw not in [None, ""] else -50.0
        ub = float(ub_raw) if ub_raw not in [None, ""] else 50.0

        include = bool(include_vals[i]) if include_vals[i] is not None else True
        locked = bool(lock_vals[i]) if lock_vals[i] is not None else False

        if not include:
            lb, ub = -100.0, -100.0
        elif locked:
            lb, ub = 0.0, 0.0

        if lb > ub:
            invalid_rows.append(ch)

        bounds[ch] = [lb, ub]

    if invalid_rows:
        msg = html.Div(
            f"Invalid bounds: Lower Bound > Upper Bound for {', '.join(invalid_rows)}.",
            className="status-danger",
        )
        err_status = html.Span("⚠ Invalid bounds", className="sticky-status sticky-status--error")
        return None, msg, True, err_status
    

    if run_mode == "optimize":
        updated_data = update_data_from_ui(
            data=data,
            optimization_goal=goal,
            total_target=mmm_budget,
            channel_spends=base_investment,
            bounds_dict=bounds,
        )
        updated_data["incremental_channels"] = INCREMENTAL_CHANNELS
        updated_data["incremental_spends"] = incremental_spends

        out = run_optimizer_for_ui(updated_data)
    else:
        return (
            None,
            html.Div(
                "Simulation mode is selected, but simulation logic is not enabled yet.",
                className="status-warning",
            ),
            True,
            None,
        )



    if not out.success:
        err_status = html.Span("⚠ Optimization failed", className="sticky-status sticky-status--error")
        return None, html.Div(out.error or "Optimization failed.", className="status-danger"), True, err_status

    results_df = out.results
    if results_df is None or results_df.empty:
        err_status = html.Span("⚠ No results returned", className="sticky-status sticky-status--error")
        return None, html.Div("Optimization returned no results.", className="status-danger"), True, err_status

    df = results_df.copy()

    if "Channel" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Channel"})


    k = compute_kpis(df)
    incr_pct = ((k["opt_rev"] / k["actual_rev"]) - 1) * 100 if k["actual_rev"] else 0.0
    invest_delta = k["opt_spend"] - k["actual_spend"]

    invest_change_pct = ((k["opt_spend"] / k["actual_spend"]) - 1) * 100 if k["actual_spend"] else 0.0

    # -----------------------------
    # Build user message (CORRECT)
    # -----------------------------

    if incremental_spends:
        overlay_spend = sum(incremental_spends.values())

        # overlay revenue = optimized - MMM-only optimized
        overlay_df = df[df["Channel Type"] == "Incremental (Linear)"]
        overlay_rev = overlay_df["Optimized Response Metric"].sum()


        msg_text = (
            f"Scenario overlay applied: "
            f"+{fmt_money_short(overlay_rev)} revenue from "
            f"{fmt_money_short(overlay_spend)} additional investment "
            f"(linear assumption, not optimized)."
        )
    else:
        msg_text = (
            f"Optimized MMM plan: "
            f"{incr_pct:+.1f}% higher response "
            f"with {invest_change_pct:+.1f}% change in total investment. "
            f"ROAS {k['actual_roas']:.2f} → {k['opt_roas']:.2f}."
        )


    df["_ui_bounds"] = df["Channel"].map(bounds)

    success_status = html.Span(
        [html.Span("✓ ", style={"fontWeight": 900}), f"Optimized  {incr_pct:+.1f}% response  ·  ROAS {k['actual_roas']:.2f} → {k['opt_roas']:.2f}"],
        className="sticky-status sticky-status--success",
    )

    return df.to_dict("records"), html.Div(msg_text, className="status-success"), False, success_status



# =================================================
# Response curves
# =================================================
@app.callback(
    Output("response-curves", "children"),
    Input("optimizer-results", "data"),
)
def render_response_curves(results):
    if not results:
        return None

    df = pd.DataFrame(results)
    df["Channel"] = df["Channel"].fillna("Incremental (Simulated)")
    if "Channel" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Channel"})

    curves = []
    for _, row in df[df["Channel Type"] == "Optimized"].iterrows():
        ch = row["Channel"]

        if ch not in base_investment:
            continue

        optimized_spend = parse_currency(row["Optimized Spend"])

        bounds_ui = row.get("_ui_bounds")

        lb = bounds_ui[0] if isinstance(bounds_ui, (list, tuple)) and len(bounds_ui) > 0 else None
        ub = bounds_ui[1] if isinstance(bounds_ui, (list, tuple)) and len(bounds_ui) > 1 else None



        fig = build_response_curve_fig(
            channel=ch,
            s_curve_params=data["s_curve_params"],
            base_spend=base_investment[ch],
            optimized_spend=optimized_spend,
            lb_pct=lb,
            ub_pct=ub,
        )

        curves.append(
            dbc.Col(
                dcc.Graph(
                    figure=fig,
                    config={"displayModeBar": False, "displaylogo": False},
                ),
                md=6,
            )
        )

    if not curves:
        return html.Div("No response curves available.", style={"color": UI_MUTED})

    return html.Div(
        [
            html.Div("4. RESPONSE CURVES", className="section-title", style={"marginTop": "10px"}),
            html.Div(
                "Channel-level response and ROAS curves illustrating diminishing returns and optimal investment levels.",
                className="comment",
            ),
            response_curve_legend,
            dbc.Row(curves, className="g-3"),
        ]
    )


# =================================================
# KPI cards + reallocation summary
# =================================================
def render_kpis(results):
    baseline = get_baseline(data)
    df = pd.DataFrame(results)

    def parse(x):
        return parse_currency(x) if isinstance(x, str) else float(x)

    # ==============================
    # 1️⃣ Split MMM vs Scenario
    # ==============================
    mmm_df = df[df["Channel Type"] == "Optimized"].copy()
    scenario_df = df[df["Channel Type"] == "Incremental (Linear)"].copy()

    # ==============================
    # 2️⃣ MMM-only core metrics
    # ==============================
    mmm_actual_spend = mmm_df["Actual/Input Spend"].apply(parse).sum()
    mmm_opt_spend = mmm_df["Optimized Spend"].apply(parse).sum()

    mmm_actual_incr_rev = mmm_df["Actual Response Metric"].apply(parse).sum()
    mmm_opt_incr_rev = mmm_df["Optimized Response Metric"].apply(parse).sum()

    # MMM totals
    mmm_actual_total_rev = mmm_actual_incr_rev + baseline
    mmm_opt_total_rev = mmm_opt_incr_rev + baseline

    # Changes
    invest_change_pct = (
        (mmm_opt_spend / mmm_actual_spend - 1) * 100
        if mmm_actual_spend > 0 else 0.0
    )

    incr_rev = mmm_opt_incr_rev - mmm_actual_incr_rev

    total_rev_change_pct = (
        (mmm_opt_total_rev / mmm_actual_total_rev - 1) * 100
        if mmm_actual_total_rev > 0 else 0.0
    )

    # ROAS (incremental only, no baseline)
    incr_roas_actual = (
        mmm_actual_incr_rev / mmm_actual_spend
        if mmm_actual_spend > 0 else 0.0
    )
    incr_roas_opt = (
        mmm_opt_incr_rev / mmm_opt_spend
        if mmm_opt_spend > 0 else 0.0
    )

    incr_roas_change_pct = (
        (incr_roas_opt / incr_roas_actual - 1) * 100
        if incr_roas_actual > 0 else 0.0
    )

    # ==============================
    # 3️⃣ Render KPI cards
    # ==============================
    return html.Div(
        className="kpi-row",
        children=[
            kpi_card(
                "MMM Optimized Investment",
                fmt_money_short(mmm_opt_spend),
                f"was {fmt_money_short(mmm_actual_spend)} ({invest_change_pct:+.1f}%)",
                tooltip=fmt_money_full(mmm_opt_spend),
            ),
            kpi_card(
                "MMM Incremental Revenue (Excl. Baseline)",
                fmt_money_short(incr_rev),
                "From optimized reallocation only",
                tooltip=fmt_money_full(incr_rev),
            ),
            kpi_card(
                "Total Revenue (Incl. Baseline)",
                fmt_money_short(mmm_opt_total_rev),
                f"was {fmt_money_short(mmm_actual_total_rev)} ({total_rev_change_pct:+.1f}%)",
                tooltip=fmt_money_full(mmm_opt_total_rev),
            ),
            kpi_card(
                "MMM Incremental ROAS",
                f"{incr_roas_opt:.2f}",
                f"was {incr_roas_actual:.2f} ({incr_roas_change_pct:+.1f}%)",
                tooltip=f"Optimized: {incr_roas_opt:.4f} | Current: {incr_roas_actual:.4f}",
            ),
        ],
    )

@app.callback(
    Output({"type": "tilt", "ch": ALL}, "disabled"),
    Input({"type": "sim-include", "ch": ALL}, "value"),
    Input("run-mode", "value"),
)
def toggle_sim_sliders(includes, run_mode):

    if not includes:
        raise PreventUpdate

    if run_mode != "simulate":
        return [True] * len(includes)

    return [not inc for inc in includes]



@app.callback(
    Output("kpis", "children"),
    Input("optimizer-results", "data"),
)
def render_kpi_cards(results):
    if not results:
        return None

    df = pd.DataFrame(results)

    # Split scenario channels
    scenario_df = df[df["Channel Type"] == "Incremental (Linear)"].copy()

    # KPI cards (MMM only)
    kpi_block = render_kpis(results)

    # Optional scenario note
    scenario_note = None
    if not scenario_df.empty:
        def parse(x):
            return parse_currency(x) if isinstance(x, str) else float(x)

        scenario_spend = scenario_df["Optimized Spend"].apply(parse).sum()
        scenario_rev = scenario_df["Optimized Response Metric"].apply(parse).sum()

        if scenario_spend > 0:
            scenario_note = html.Div(
                f"Scenario overlay applied: +{fmt_money_short(scenario_rev)} revenue "
                f"from {fmt_money_short(scenario_spend)} additional investment "
                f"(linear assumption, not optimized).",
                className="comment",
                style={
                    "marginTop": "6px",
                    "fontSize": "11px",
                    "color": UI_MUTED,
                },
            )

    return html.Div(
        [
            kpi_block,
            scenario_note,
        ]
    )

@app.callback(
    Output("bounds-table-optimize", "style"),
    Output("bounds-table-simulate", "style"),
    Input("run-mode", "value"),
)
def toggle_optimize_sim_tables(run_mode):
    if run_mode == "simulate":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("realloc-summary", "children"),
    Input("optimizer-results", "data"),
)
def render_realloc_summary(results):
    if not results:
        return None

    df = pd.DataFrame(results)
    if "Channel" not in df.columns:
        df = df.reset_index(drop=True)
    if "Δ Spend (Abs)" not in df.columns:
        return None

    df2 = df[df["Channel"].astype(str).str.upper() != "TOTAL"].copy()
    if df2.empty:
        return None

    df2["Δ Spend (Abs)"] = df2["Δ Spend (Abs)"].apply(parse_currency)
    inc = df2.loc[df2["Δ Spend (Abs)"].idxmax()] if (df2["Δ Spend (Abs)"].max() is not None) else None
    dec = df2.loc[df2["Δ Spend (Abs)"].idxmin()] if (df2["Δ Spend (Abs)"].min() is not None) else None

    parts = []
    if inc is not None and float(inc["Δ Spend (Abs)"]) > 0:
        parts.append(f"Largest increase: {inc['Channel']} (+{fmt_money_short(inc['Δ Spend (Abs)'])})")
    if dec is not None and float(dec["Δ Spend (Abs)"]) < 0:
        parts.append(f"Largest decrease: {dec['Channel']} ({fmt_money_short(dec['Δ Spend (Abs)'])})")

    if not parts:
        return None

    return html.Div(" • ".join(parts), className="realloc-summary")


# =================================================
# Render results table
# =================================================
@app.callback(
    Output("table", "children"),
    Input("optimizer-results", "data"),
)
def render_table(results):
    if not results:
        return html.Div("No results yet.", style={"color": UI_MUTED, "fontSize": "12px"})

    df = pd.DataFrame(results)

    if "Channel" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Channel"})

    # <<< FIX 2A: GUARANTEE Channel Type EXISTS >>>
    if "Channel Type" not in df.columns:
        df["Channel Type"] = np.where(
            df["Channel"].isin(OPTIMIZED_CHANNELS),
            "Optimized",
            "Incremental (Linear)",
        )

    preferred_order = [
        "Channel",
        "Actual/Input Spend",
        "Optimized Spend",
        "Δ Spend (Abs)",
        "Δ Spend (%)",
        "Actual Response Metric",
        "Optimized Response Metric",
        "Actual ROI",
        "Optimized ROI",
        "Channel Type",  # keep internally
    ]
    cols = [c for c in preferred_order if c in df.columns]
    df = df[cols]

    df["__direction__"] = np.where(
        df["Δ Spend (%)"] > 0,
        "up",
        np.where(df["Δ Spend (%)"] < 0, "down", "flat"),
    )
    df.loc[df["Channel"].astype(str).str.upper() == "TOTAL", "__direction__"] = "flat"

    arrow = {"up": "▲", "down": "▼", "flat": "●"}

    df["Channel Display"] = (
        np.where(
            df["Channel Type"] == "Incremental (Linear)",
            "Ⓢ ",
            "Ⓞ ",
        )
        + df["__direction__"].map(arrow).fillna("●")
        + " "
        + df["Channel"].astype(str)
    )

    df["Row Type"] = np.where(
        df["Channel Type"] == "Incremental (Linear)",
        "Simulated",
        "Optimized",
    )

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[
            {"name": "Channel", "id": "Channel Display"},
            {
                "name": "Actual/Input Spend",
                "id": "Actual/Input Spend",
                "type": "numeric",
                "format": Format(scheme=Scheme.fixed, group=Group.yes, precision=0),
            },
            {
                "name": "Optimized Spend",
                "id": "Optimized Spend",
                "type": "numeric",
                "format": Format(scheme=Scheme.fixed, group=Group.yes, precision=0),
            },
            {
                "name": "Δ Spend (Abs)",
                "id": "Δ Spend (Abs)",
                "type": "numeric",
                "format": Format(scheme=Scheme.fixed, group=Group.yes, precision=0),
            },
            {
                "name": "Δ Spend (%)",
                "id": "Δ Spend (%)",
                "type": "numeric",
                "format": Format(precision=1, scheme=Scheme.fixed, symbol_suffix="%"),
            },
            {
                "name": "Actual Response",
                "id": "Actual Response Metric",
                "type": "numeric",
                "format": Format(scheme=Scheme.fixed, group=Group.yes, precision=0),
            },
            {
                "name": "Optimized Response",
                "id": "Optimized Response Metric",
                "type": "numeric",
                "format": Format(scheme=Scheme.fixed, group=Group.yes, precision=0),
            },
            {"name": "Actual ROAS", "id": "Actual ROI", "type": "numeric", "format": Format(precision=2)},
            {"name": "Optimized ROAS", "id": "Optimized ROI", "type": "numeric", "format": Format(precision=2)},
        ],
        fixed_columns={"headers": True, "data": 1},
        sort_action="native",
        sort_by=[{"column_id": "Δ Spend (%)", "direction": "desc"}],
        style_table={"overflowX": "auto", "width": "100%", "minWidth": "100%"},
        style_cell={
            "padding": "8px 12px",
            "fontFamily": "Arial, system-ui",
            "fontSize": "12px",
            "whiteSpace": "nowrap",
            "textAlign": "right",
            "fontVariantNumeric": "tabular-nums",
            "border": f"1px solid {UI_BORDER}",
            "backgroundColor": "white",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Channel Display"}, "textAlign": "left", "fontWeight": "800", "paddingLeft": "12px"},
            {"if": {"column_id": "Δ Spend (%)"}, "textAlign": "center"},
            {"if": {"column_id": "Actual ROI"}, "textAlign": "right"},
            {"if": {"column_id": "Optimized ROI"}, "textAlign": "right"},
        ],
        style_header={
            "backgroundColor": NAVY_PRIMARY,
            "fontWeight": "800",
            "color": "white",
            "border": "none",
            "textAlign": "right",
        },
        style_header_conditional=[
            {"if": {"column_id": "Channel Display"}, "textAlign": "left"},
            {"if": {"column_id": "Δ Spend (%)"}, "textAlign": "center"},
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgba(14,41,97,0.04)"},
            {
                "if": {"state": "active"},
                "backgroundColor": SKY_INTERACT,
                "border": "1px solid SKY_INTERACT",
            },
            {
                "if": {"filter_query": "{Δ Spend (%)} > 0", "column_id": "Δ Spend (%)"},
                "color": GREEN_POSITIVE,
                "fontWeight": "800",
            },
            {
                "if": {"filter_query": "{Δ Spend (%)} < 0", "column_id": "Δ Spend (%)"},
                "color": "#64748b",
                "fontWeight": "800",
            },
            {
                "if": {"filter_query": "{Row Type} = 'Simulated'"},
                "backgroundColor": "rgba(100,116,139,0.08)",
                "fontStyle": "italic",
                "color": "#475569",
            },
            {
                "if": {"column_id": "Optimized Spend"},
                "fontWeight": "900",
                "backgroundColor": "rgba(0, 78, 157, 0.06)",
            },
        ],
    )



# =================================================
# Pack A Visualizations (Waterfall, Dumbbell, Delta, Quadrant)
# =================================================
@app.callback(
    Output("viz-waterfall", "children"),
    Input("optimizer-results", "data"),
)
def render_viz_waterfall(results):
    if not results:
        return None
    fig = build_total_waterfall(results)
    return dcc.Graph(figure=fig, config={"displayModeBar": False, "displaylogo": False, "responsive": True})


@app.callback(
    Output("viz-dumbbell", "children"),
    Input("optimizer-results", "data"),
)
def render_viz_dumbbell(results):
    if not results:
        return None
    fig = build_spend_dumbbell(results)
    return dcc.Graph(figure=fig, config={"displayModeBar": False, "displaylogo": False, "responsive": True})


@app.callback(
    Output("viz-delta", "children"),
    Input("optimizer-results", "data"),
)
def render_viz_delta(results):
    if not results:
        return None
    fig = build_delta_spend_bar(results)
    return dcc.Graph(figure=fig, config={"displayModeBar": False, "displaylogo": False, "responsive": True})


@app.callback(
    Output("viz-quadrant", "children"),
    Input("optimizer-results", "data"),
)
def render_viz_quadrant(results):
    if not results:
        return None
    fig = build_efficiency_quadrant(results)
    return dcc.Graph(figure=fig, config={"displayModeBar": False, "displaylogo": False, "responsive": True})

@app.callback(
    Output("viz-mroas", "children"),
    Input("optimizer-results", "data"),
)
def render_viz_mroas(results):
    if not results:
        return None
    fig = build_marginal_roas_rank(results)
    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": False, "displaylogo": False, "responsive": True},
    )




# =================================================
# Download results (XLSX)
# =================================================
@app.callback(
    Output("download-results", "data"),
    Input("download-btn", "n_clicks"),
    State("optimizer-results", "data"),
    State("goal", "value"),
    prevent_initial_call=True,
)
def download_results(n, results, goal):
    if not results:
        raise PreventUpdate

    df = pd.DataFrame(results).copy()
    baseline = get_baseline(data)

    if "Channel" not in df.columns:
        df = df.reset_index(drop=True)

    if "Δ Spend (%)" in df.columns:
        df["Δ Spend (%)"] = df["Δ Spend (%)"] / 100

    if "Actual Response Metric" in df.columns:
        df["Actual Response"] = df["Actual Response Metric"]
    if "Optimized Response Metric" in df.columns:
        df["Optimized Response"] = df["Optimized Response Metric"]

    df["Actual Total Revenue (Incl. Baseline)"] = df["Actual Response"]
    df["Optimized Total Revenue (Incl. Baseline)"] = df["Optimized Response"]

    total_mask = df["Channel"].astype(str).str.upper() == "TOTAL"
    if total_mask.any():
        df.loc[total_mask, "Actual Total Revenue (Incl. Baseline)"] = (
            df.loc[total_mask, "Actual Total Revenue (Incl. Baseline)"] + baseline
        )
        df.loc[total_mask, "Optimized Total Revenue (Incl. Baseline)"] = (
            df.loc[total_mask, "Optimized Total Revenue (Incl. Baseline)"] + baseline
        )

    df["Marginal ROAS (+10% Spend)"] = df.apply(
        lambda r: marginal_roas_10pct(
            str(r["Channel"]),
            parse_currency(r["Optimized Spend"]),
            data["s_curve_params"],
        )
        if str(r["Channel"]).upper() != "TOTAL"
        else np.nan,
        axis=1,
    )

    kpis = compute_kpis(pd.DataFrame(results))
    actual_spend = float(kpis["actual_spend"] or 0.0)
    opt_spend = float(kpis["opt_spend"] or 0.0)

    actual_incr_resp = float(kpis["actual_rev"] or 0.0)
    opt_incr_resp = float(kpis["opt_rev"] or 0.0)

    actual_total_rev = actual_incr_resp + float(baseline)
    opt_total_rev = opt_incr_resp + float(baseline)

    actual_incr_roas = (actual_incr_resp / actual_spend) if actual_spend else np.nan
    opt_incr_roas = (opt_incr_resp / opt_spend) if opt_spend else np.nan

    actual_total_roas = (actual_total_rev / actual_spend) if actual_spend else np.nan
    opt_total_roas = (opt_total_rev / opt_spend) if opt_spend else np.nan

    df_kpis = pd.DataFrame(
        {
            "Metric": [
                "Total Spend",
                "Incremental Response (Excl. Baseline)",
                "Total Revenue (Incl. Baseline)",
                "Incremental ROAS (Excl. Baseline)",
                "Total ROAS (Incl. Baseline)",
            ],
            "Actual": [
                actual_spend,
                actual_incr_resp,
                actual_total_rev,
                actual_incr_roas,
                actual_total_roas,
            ],
            "Optimized": [
                opt_spend,
                opt_incr_resp,
                opt_total_rev,
                opt_incr_roas,
                opt_total_roas,
            ],
        }
    )
    df_kpis["% Change"] = (df_kpis["Optimized"] / df_kpis["Actual"] - 1).replace([np.inf, -np.inf], np.nan)

    df_meta = pd.DataFrame(
        {
            "Field": ["Date", "Optimization Goal"],
            "Value": [
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                goal,
            ],
        }
    )

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book

        df.to_excel(writer, index=False, sheet_name="Results")
        df_kpis.to_excel(writer, index=False, sheet_name="KPI Summary")
        df_meta.to_excel(writer, index=False, sheet_name="Metadata")

        dollar_fmt = workbook.add_format({"num_format": "$#,##0"})
        pct_fmt = workbook.add_format({"num_format": "0.00%"})
        ratio_fmt = workbook.add_format({"num_format": "0.00"})
        bold_fmt = workbook.add_format({"bold": True})

        ws = writer.sheets["Results"]

        dollar_cols = [
            "Actual/Input Spend",
            "Optimized Spend",
            "Δ Spend (Abs)",
            "Actual Response",
            "Optimized Response",
            "Actual Total Revenue (Incl. Baseline)",
            "Optimized Total Revenue (Incl. Baseline)",
        ]
        for col in dollar_cols:
            if col in df.columns:
                c = df.columns.get_loc(col)
                ws.set_column(c, c, 22, dollar_fmt)

        if "Δ Spend (%)" in df.columns:
            c = df.columns.get_loc("Δ Spend (%)")
            ws.set_column(c, c, 14, pct_fmt)

        for col in ["Actual ROI", "Optimized ROI", "Marginal ROAS (+10% Spend)"]:
            if col in df.columns:
                c = df.columns.get_loc(col)
                ws.set_column(c, c, 22 if "Marginal" in col else 14, ratio_fmt)

        total_row_idx = df.index[df["Channel"].astype(str).str.upper() == "TOTAL"]
        if len(total_row_idx) > 0:
            r = int(total_row_idx[0]) + 1
            ws.set_row(r, None, bold_fmt)

        ws_kpi = writer.sheets["KPI Summary"]
        ws_kpi.set_column(0, 0, 42, bold_fmt)
        ws_kpi.set_column(1, 2, 22, dollar_fmt)
        ws_kpi.set_column(3, 3, 14, pct_fmt)

        writer.sheets["Metadata"].set_column(0, 1, 30)

    output.seek(0)
    return dcc.send_bytes(
        output.read(),
        filename=f"mmm_optimizer_results_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
    )


# =================================================
# Download results (CSV)
# =================================================
@app.callback(
    Output("download-results-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    State("optimizer-results", "data"),
    prevent_initial_call=True,
)
def download_results_csv(n, results):
    if not results:
        raise PreventUpdate

    df = pd.DataFrame(results).copy()
    baseline = get_baseline(data)

    if "Channel" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Channel"})

    if "Actual Response Metric" in df.columns and "Actual Response" not in df.columns:
        df["Actual Response"] = df["Actual Response Metric"]
    if "Optimized Response Metric" in df.columns and "Optimized Response" not in df.columns:
        df["Optimized Response"] = df["Optimized Response Metric"]

    if "Actual Response" in df.columns:
        df["Actual Total Revenue (Incl. Baseline)"] = df["Actual Response"]
    if "Optimized Response" in df.columns:
        df["Optimized Total Revenue (Incl. Baseline)"] = df["Optimized Response"]

    total_mask = df["Channel"].astype(str).str.upper() == "TOTAL"
    if total_mask.any():
        if "Actual Total Revenue (Incl. Baseline)" in df.columns:
            df.loc[total_mask, "Actual Total Revenue (Incl. Baseline)"] = (
                df.loc[total_mask, "Actual Total Revenue (Incl. Baseline)"] + float(baseline)
            )
        if "Optimized Total Revenue (Incl. Baseline)" in df.columns:
            df.loc[total_mask, "Optimized Total Revenue (Incl. Baseline)"] = (
                df.loc[total_mask, "Optimized Total Revenue (Incl. Baseline)"] + float(baseline)
            )

    df["Marginal ROAS (+10% Spend)"] = df.apply(
        lambda r: marginal_roas_10pct(
            str(r["Channel"]),
            parse_currency(r["Optimized Spend"]),
            data["s_curve_params"],
        )
        if str(r["Channel"]).upper() != "TOTAL"
        else np.nan,
        axis=1,
    )

    preferred = [
        "Channel",
        "Actual/Input Spend",
        "Optimized Spend",
        "Δ Spend (Abs)",
        "Δ Spend (%)",
        "Actual Response",
        "Optimized Response",
        "Actual Total Revenue (Incl. Baseline)",
        "Optimized Total Revenue (Incl. Baseline)",
        "Actual ROI",
        "Optimized ROI",
        "Marginal ROAS (+10% Spend)",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return dcc.send_string(
        buf.getvalue(),
        filename=f"mmm_optimizer_results_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
    )

@app.callback(
    Output("mmm-budget-display", "style"),
    Input("run-mode", "value"),
    Input("goal", "value"),
)
def toggle_mmm_budget_display(run_mode, goal):
    if run_mode == "simulate" or (goal or "").lower() == "backward":
        return {"display": "none"}
    return {"display": "block"}


# =================================================
# Run app
# =================================================
if __name__ == "__main__":
    import os
    debug = os.environ.get("DASH_DEBUG", "true").lower() == "true"
    port  = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=debug)