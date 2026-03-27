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

# Budget hierarchy groups (D2C vs IPA & Activation)
D2C_CHANNELS = [ch for ch in ["D2C Reach", "D2C Rewards"] if ch in INCREMENTAL_CHANNELS]
IPA_CHANNELS = [ch for ch in ["IPA - Strat Cities", "IPA", "IPA - TCE", "Sponsorships"] if ch in INCREMENTAL_CHANNELS]

_D2C_DEFAULT  = sum(float(INCREMENTAL_CHANNELS[ch]["historical_spend"]) for ch in D2C_CHANNELS)
# DOM order = layout order: D2C channels first, then IPA channels
INC_DOM_ORDER = D2C_CHANNELS + IPA_CHANNELS
_IPA_DEFAULT  = sum(float(INCREMENTAL_CHANNELS[ch]["historical_spend"]) for ch in IPA_CHANNELS)
_MEDIA_DEFAULT = float(sum({k: float(v) for k, v in data["spends"].items()}.values()))
_GRAND_TOTAL  = _D2C_DEFAULT + _IPA_DEFAULT + _MEDIA_DEFAULT



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
        rev = compute_response_for_spend(ch, base)
        roi = (rev / base) if base > 0 else 0.0
        rows.append(
            html.Tr(
                [
                    html.Td(ch, style={"fontWeight": 700, "textAlign": "left", "verticalAlign": "middle"}),
                    html.Td(
                        html.Div(
                            dbc.Switch(
                                id={"type": "include", "ch": ch},
                                value=True,
                                className="table-toggle",
                            ),
                            style={"display": "flex", "justifyContent": "center", "alignItems": "center"},
                        ),
                        style={"verticalAlign": "middle"},
                    ),
                    html.Td(
                        html.Div(
                            dbc.Checkbox(
                                id={"type": "lock", "ch": ch},
                                value=False,
                                className="table-checkbox",
                            ),
                            style={"display": "flex", "justifyContent": "center", "alignItems": "center"},
                        ),
                        style={"verticalAlign": "middle"},
                    ),
                    html.Td(
                        html.Div([
                            dcc.Input(
                                id={"type": "spend", "ch": ch},
                                type="text",
                                debounce=True,
                                value=format_currency(base),
                                style={"width": "100%", "textAlign": "right"},
                            ),
                            html.Div(
                                f"Min: {fmt_money_short(base * (1 + lb / 100))}  Max: {fmt_money_short(base * (1 + ub / 100))}",
                                id={"type": "minmax-display", "ch": ch},
                                style={"fontSize": "11px", "color": "#64748B", "textAlign": "right", "marginTop": "3px"},
                            ),
                        ]),
                        style={"textAlign": "right", "verticalAlign": "middle"},
                    ),
                    html.Td(
                        fmt_money_short(rev) if rev > 0 else "—",
                        id={"type": "revenue", "ch": ch},
                        style={
                            "textAlign": "right", "verticalAlign": "middle",
                            "fontVariantNumeric": "tabular-nums",
                            "fontWeight": 700, "color": GREEN_POSITIVE,
                            "whiteSpace": "nowrap",
                            "background": "rgba(78,174,70,0.06)",
                        },
                    ),
                    html.Td(
                        f"{roi:.2f}" if roi > 0 else "—",
                        id={"type": "roi", "ch": ch},
                        style={
                            "textAlign": "right", "verticalAlign": "middle",
                            "fontVariantNumeric": "tabular-nums",
                            "background": "rgba(78,174,70,0.06)",
                        },
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
                ]
            )
        )
    return rows




def incremental_channel_rows():
    """Build inc-channel rows with historical_spend pre-populated (still used for DOM IDs)."""
    rows = []
    for ch in INCREMENTAL_CHANNELS.keys():
        hist_spend = float(INCREMENTAL_CHANNELS[ch].get("historical_spend", 0))
        rows.append(
            html.Tr([
                html.Td(ch, style={"fontWeight": 600, "textAlign": "left", "verticalAlign": "middle"}),
                html.Td(
                    html.Div(
                        dbc.Switch(id={"type": "inc-include", "ch": ch}, value=True),
                        style={"display": "flex", "justifyContent": "center", "alignItems": "center"},
                    ),
                    style={"textAlign": "center", "verticalAlign": "middle"},
                ),
                html.Td(
                    dcc.Input(
                        id={"type": "inc-spend", "ch": ch},
                        type="text",
                        debounce=True,
                        value=format_currency(hist_spend),
                        style={"width": "140px", "textAlign": "right"},
                    ),
                    style={"textAlign": "right", "verticalAlign": "middle"},
                ),
                html.Td(
                    "",
                    id={"type": "budget-pct-inc", "ch": ch},
                    style={"textAlign": "right", "verticalAlign": "middle", "fontSize": "12px", "color": "#64748B"},
                ),
            ])
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


def compute_response_for_spend(ch, annual_spend):
    """Compute annual revenue for a channel at a given spend using its hill curve."""
    if ch not in data["s_curve_params"]:
        return 0.0
    params = data["s_curve_params"][ch]
    return response_annual_from_params(params, float(annual_spend))


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
                html.Div("1. OPTIMIZATION OBJECTIVE", className="section-title"),
                html.Div(
                    "Define what you want the model to do before entering budgets or scenarios.",
                    className="comment",
                ),

                dbc.Row(
                    [
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
                html.Div(id="budget-section-description", className="comment"),

                # ---------- HIERARCHICAL BUDGET TABLE ----------
                html.Div([
                    dbc.Table(
                        [
                            html.Thead(
                                html.Tr([
                                    html.Th("Category / Channel", style={"textAlign":"left",   "fontSize":"12px"}),
                                    html.Th("Include",            style={"textAlign":"center", "fontSize":"12px"}),
                                    html.Th("Current Budget ($)", style={"textAlign":"right",  "fontSize":"12px","whiteSpace":"nowrap"}),
                                    html.Th("New Budget ($)",     style={"textAlign":"right",  "fontSize":"12px","whiteSpace":"nowrap"}),
                                    html.Th("Δ (%)",          style={"textAlign":"right",  "fontSize":"12px","whiteSpace":"nowrap"}),
                                    html.Th("% of Total",         style={"textAlign":"right",  "fontSize":"12px","whiteSpace":"nowrap"}),
                                    html.Th("Revenue ($)",        style={"textAlign":"right",  "fontSize":"12px","color":"#4EAE46","whiteSpace":"nowrap"}),
                                    html.Th("ROI",                style={"textAlign":"right",  "fontSize":"12px","color":"#4EAE46"}),
                                ])
                            ),
                            html.Tbody(
                                # D2C category header row
                                [
                                    html.Tr([
                                        html.Td(html.Strong("D2C"), style={"textAlign":"left","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td("",                  style={"textAlign":"center","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span(format_currency(_D2C_DEFAULT), style={"fontWeight":700,"fontSize":"13px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span(format_currency(_D2C_DEFAULT), id="budget-d2c-display", style={"fontWeight":700,"fontSize":"13px"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span("0.0%", id="budget-d2c-delta", style={"fontWeight":700,"fontSize":"12px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span("", id="budget-pct-d2c", style={"fontSize":"12px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span("", id="budget-d2c-revenue", style={"fontWeight":700,"fontSize":"13px","color":"#4EAE46"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span("", id="budget-d2c-roi", style={"fontWeight":700,"fontSize":"13px","color":"#4EAE46"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                    ]),
                                ] + [
                                    html.Tr([
                                        html.Td(html.Span("D2C Reach", style={"paddingLeft":"20px","fontSize":"13px"}),
                                                style={"textAlign":"left","verticalAlign":"middle"}),
                                        html.Td(
                                            html.Div(dbc.Switch(id={"type":"inc-include","ch":"D2C Reach"}, value=True),
                                                     style={"display":"flex","justifyContent":"center","alignItems":"center"}),
                                            style={"textAlign":"center","verticalAlign":"middle"},
                                        ),
                                        html.Td(html.Span(format_currency(float(INCREMENTAL_CHANNELS["D2C Reach"]["historical_spend"])),
                                                          style={"fontSize":"12px","color":"#64748B","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(
                                            html.Div([
                                        html.Button("−", id={"type":"inc-minus","ch":"D2C Reach"}, n_clicks=0, style={"width":"22px","height":"28px","flexShrink":0,"border":"1px solid #ced4da","borderRight":"none","borderRadius":"4px 0 0 4px","background":"#f8f9fa","cursor":"pointer","fontSize":"14px","lineHeight":"1","padding":"0"}),
                                                dcc.Input(id={"type":"inc-spend","ch":"D2C Reach"}, type="text", debounce=True,
                                                          value=format_currency(float(INCREMENTAL_CHANNELS["D2C Reach"]["historical_spend"])),
                                                          style={"flex":"1","textAlign":"right","minWidth":"80px",
                                                                 "border":"1px solid #ced4da","borderRadius":"0",
                                                                 "height":"28px","padding":"0 4px","fontSize":"12px"}),
                                        html.Button("+", id={"type":"inc-plus","ch":"D2C Reach"}, n_clicks=0, style={"width":"22px","height":"28px","flexShrink":0,"border":"1px solid #ced4da","borderLeft":"none","borderRadius":"0 4px 4px 0","background":"#f8f9fa","cursor":"pointer","fontSize":"14px","lineHeight":"1","padding":"0"}),
                                            ], style={"display":"flex","alignItems":"center"}),
                                            style={"textAlign":"right","verticalAlign":"middle"},
                                        ),
                                        html.Td(html.Span("0.0%", id={"type":"inc-delta","ch":"D2C Reach"},
                                                          style={"fontWeight":700,"fontSize":"12px","color":"#64748B","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(html.Span("", id={"type":"budget-pct-inc","ch":"D2C Reach"}, style={"fontSize":"12px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(html.Span("", id={"type":"inc-revenue","ch":"D2C Reach"},
                                                          style={"fontWeight":700,"color":"#4EAE46","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(78,174,70,0.06)"}),
                                        html.Td(html.Span("", id={"type":"inc-roi","ch":"D2C Reach"},
                                                          style={"color":"#4EAE46","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(78,174,70,0.06)"}),
                                    ]),
                                    html.Tr([
                                        html.Td(html.Span("D2C Rewards", style={"paddingLeft":"20px","fontSize":"13px"}),
                                                style={"textAlign":"left","verticalAlign":"middle"}),
                                        html.Td(
                                            html.Div(dbc.Switch(id={"type":"inc-include","ch":"D2C Rewards"}, value=True),
                                                     style={"display":"flex","justifyContent":"center","alignItems":"center"}),
                                            style={"textAlign":"center","verticalAlign":"middle"},
                                        ),
                                        html.Td(html.Span(format_currency(float(INCREMENTAL_CHANNELS["D2C Rewards"]["historical_spend"])),
                                                          style={"fontSize":"12px","color":"#64748B","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(
                                            html.Div([
                                        html.Button("−", id={"type":"inc-minus","ch":"D2C Rewards"}, n_clicks=0, style={"width":"22px","height":"28px","flexShrink":0,"border":"1px solid #ced4da","borderRight":"none","borderRadius":"4px 0 0 4px","background":"#f8f9fa","cursor":"pointer","fontSize":"14px","lineHeight":"1","padding":"0"}),
                                                dcc.Input(id={"type":"inc-spend","ch":"D2C Rewards"}, type="text", debounce=True,
                                                          value=format_currency(float(INCREMENTAL_CHANNELS["D2C Rewards"]["historical_spend"])),
                                                          style={"flex":"1","textAlign":"right","minWidth":"80px",
                                                                 "border":"1px solid #ced4da","borderRadius":"0",
                                                                 "height":"28px","padding":"0 4px","fontSize":"12px"}),
                                        html.Button("+", id={"type":"inc-plus","ch":"D2C Rewards"}, n_clicks=0, style={"width":"22px","height":"28px","flexShrink":0,"border":"1px solid #ced4da","borderLeft":"none","borderRadius":"0 4px 4px 0","background":"#f8f9fa","cursor":"pointer","fontSize":"14px","lineHeight":"1","padding":"0"}),
                                            ], style={"display":"flex","alignItems":"center"}),
                                            style={"textAlign":"right","verticalAlign":"middle"},
                                        ),
                                        html.Td(html.Span("0.0%", id={"type":"inc-delta","ch":"D2C Rewards"},
                                                          style={"fontWeight":700,"fontSize":"12px","color":"#64748B","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(html.Span("", id={"type":"budget-pct-inc","ch":"D2C Rewards"}, style={"fontSize":"12px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(html.Span("", id={"type":"inc-revenue","ch":"D2C Rewards"},
                                                          style={"fontWeight":700,"color":"#4EAE46","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(78,174,70,0.06)"}),
                                        html.Td(html.Span("", id={"type":"inc-roi","ch":"D2C Rewards"},
                                                          style={"color":"#4EAE46","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(78,174,70,0.06)"}),
                                    ]),
                                ] +
                                # IPA & Activation category header row
                                [
                                    html.Tr([
                                        html.Td(html.Strong("IPA & Activation"), style={"textAlign":"left","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td("",                               style={"textAlign":"center","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span(format_currency(_IPA_DEFAULT), style={"fontWeight":700,"fontSize":"13px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span(format_currency(_IPA_DEFAULT), id="budget-ipa-display", style={"fontWeight":700,"fontSize":"13px"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span("0.0%", id="budget-ipa-delta", style={"fontWeight":700,"fontSize":"12px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span("", id="budget-pct-ipa", style={"fontSize":"12px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span("", id="budget-ipa-revenue", style={"fontWeight":700,"fontSize":"13px","color":"#4EAE46"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span("", id="budget-ipa-roi", style={"fontWeight":700,"fontSize":"13px","color":"#4EAE46"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                    ]),
                                ] + [
                                    html.Tr([
                                        html.Td(html.Span("IPA - Strat Cities", style={"paddingLeft":"20px","fontSize":"13px"}),
                                                style={"textAlign":"left","verticalAlign":"middle"}),
                                        html.Td(
                                            html.Div(dbc.Switch(id={"type":"inc-include","ch":"IPA - Strat Cities"}, value=True),
                                                     style={"display":"flex","justifyContent":"center","alignItems":"center"}),
                                            style={"textAlign":"center","verticalAlign":"middle"},
                                        ),
                                        html.Td(html.Span(format_currency(float(INCREMENTAL_CHANNELS["IPA - Strat Cities"]["historical_spend"])),
                                                          style={"fontSize":"12px","color":"#64748B","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(
                                            html.Div([
                                        html.Button("−", id={"type":"inc-minus","ch":"IPA - Strat Cities"}, n_clicks=0, style={"width":"22px","height":"28px","flexShrink":0,"border":"1px solid #ced4da","borderRight":"none","borderRadius":"4px 0 0 4px","background":"#f8f9fa","cursor":"pointer","fontSize":"14px","lineHeight":"1","padding":"0"}),
                                                dcc.Input(id={"type":"inc-spend","ch":"IPA - Strat Cities"}, type="text", debounce=True,
                                                          value=format_currency(float(INCREMENTAL_CHANNELS["IPA - Strat Cities"]["historical_spend"])),
                                                          style={"flex":"1","textAlign":"right","minWidth":"80px",
                                                                 "border":"1px solid #ced4da","borderRadius":"0",
                                                                 "height":"28px","padding":"0 4px","fontSize":"12px"}),
                                        html.Button("+", id={"type":"inc-plus","ch":"IPA - Strat Cities"}, n_clicks=0, style={"width":"22px","height":"28px","flexShrink":0,"border":"1px solid #ced4da","borderLeft":"none","borderRadius":"0 4px 4px 0","background":"#f8f9fa","cursor":"pointer","fontSize":"14px","lineHeight":"1","padding":"0"}),
                                            ], style={"display":"flex","alignItems":"center"}),
                                            style={"textAlign":"right","verticalAlign":"middle"},
                                        ),
                                        html.Td(html.Span("0.0%", id={"type":"inc-delta","ch":"IPA - Strat Cities"},
                                                          style={"fontWeight":700,"fontSize":"12px","color":"#64748B","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(html.Span("", id={"type":"budget-pct-inc","ch":"IPA - Strat Cities"}, style={"fontSize":"12px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(html.Span("", id={"type":"inc-revenue","ch":"IPA - Strat Cities"},
                                                          style={"fontWeight":700,"color":"#4EAE46","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(78,174,70,0.06)"}),
                                        html.Td(html.Span("", id={"type":"inc-roi","ch":"IPA - Strat Cities"},
                                                          style={"color":"#4EAE46","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(78,174,70,0.06)"}),
                                    ]),
                                    html.Tr([
                                        html.Td(html.Span("IPA", style={"paddingLeft":"20px","fontSize":"13px"}),
                                                style={"textAlign":"left","verticalAlign":"middle"}),
                                        html.Td(
                                            html.Div(dbc.Switch(id={"type":"inc-include","ch":"IPA"}, value=True),
                                                     style={"display":"flex","justifyContent":"center","alignItems":"center"}),
                                            style={"textAlign":"center","verticalAlign":"middle"},
                                        ),
                                        html.Td(html.Span(format_currency(float(INCREMENTAL_CHANNELS["IPA"]["historical_spend"])),
                                                          style={"fontSize":"12px","color":"#64748B","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(
                                            html.Div([
                                        html.Button("−", id={"type":"inc-minus","ch":"IPA"}, n_clicks=0, style={"width":"22px","height":"28px","flexShrink":0,"border":"1px solid #ced4da","borderRight":"none","borderRadius":"4px 0 0 4px","background":"#f8f9fa","cursor":"pointer","fontSize":"14px","lineHeight":"1","padding":"0"}),
                                                dcc.Input(id={"type":"inc-spend","ch":"IPA"}, type="text", debounce=True,
                                                          value=format_currency(float(INCREMENTAL_CHANNELS["IPA"]["historical_spend"])),
                                                          style={"flex":"1","textAlign":"right","minWidth":"80px",
                                                                 "border":"1px solid #ced4da","borderRadius":"0",
                                                                 "height":"28px","padding":"0 4px","fontSize":"12px"}),
                                        html.Button("+", id={"type":"inc-plus","ch":"IPA"}, n_clicks=0, style={"width":"22px","height":"28px","flexShrink":0,"border":"1px solid #ced4da","borderLeft":"none","borderRadius":"0 4px 4px 0","background":"#f8f9fa","cursor":"pointer","fontSize":"14px","lineHeight":"1","padding":"0"}),
                                            ], style={"display":"flex","alignItems":"center"}),
                                            style={"textAlign":"right","verticalAlign":"middle"},
                                        ),
                                        html.Td(html.Span("0.0%", id={"type":"inc-delta","ch":"IPA"},
                                                          style={"fontWeight":700,"fontSize":"12px","color":"#64748B","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(html.Span("", id={"type":"budget-pct-inc","ch":"IPA"}, style={"fontSize":"12px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(html.Span("", id={"type":"inc-revenue","ch":"IPA"},
                                                          style={"fontWeight":700,"color":"#4EAE46","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(78,174,70,0.06)"}),
                                        html.Td(html.Span("", id={"type":"inc-roi","ch":"IPA"},
                                                          style={"color":"#4EAE46","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(78,174,70,0.06)"}),
                                    ]),
                                    html.Tr([
                                        html.Td(html.Span("IPA - TCE", style={"paddingLeft":"20px","fontSize":"13px"}),
                                                style={"textAlign":"left","verticalAlign":"middle"}),
                                        html.Td(
                                            html.Div(dbc.Switch(id={"type":"inc-include","ch":"IPA - TCE"}, value=True),
                                                     style={"display":"flex","justifyContent":"center","alignItems":"center"}),
                                            style={"textAlign":"center","verticalAlign":"middle"},
                                        ),
                                        html.Td(html.Span(format_currency(float(INCREMENTAL_CHANNELS["IPA - TCE"]["historical_spend"])),
                                                          style={"fontSize":"12px","color":"#64748B","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(
                                            html.Div([
                                        html.Button("−", id={"type":"inc-minus","ch":"IPA - TCE"}, n_clicks=0, style={"width":"22px","height":"28px","flexShrink":0,"border":"1px solid #ced4da","borderRight":"none","borderRadius":"4px 0 0 4px","background":"#f8f9fa","cursor":"pointer","fontSize":"14px","lineHeight":"1","padding":"0"}),
                                                dcc.Input(id={"type":"inc-spend","ch":"IPA - TCE"}, type="text", debounce=True,
                                                          value=format_currency(float(INCREMENTAL_CHANNELS["IPA - TCE"]["historical_spend"])),
                                                          style={"flex":"1","textAlign":"right","minWidth":"80px",
                                                                 "border":"1px solid #ced4da","borderRadius":"0",
                                                                 "height":"28px","padding":"0 4px","fontSize":"12px"}),
                                        html.Button("+", id={"type":"inc-plus","ch":"IPA - TCE"}, n_clicks=0, style={"width":"22px","height":"28px","flexShrink":0,"border":"1px solid #ced4da","borderLeft":"none","borderRadius":"0 4px 4px 0","background":"#f8f9fa","cursor":"pointer","fontSize":"14px","lineHeight":"1","padding":"0"}),
                                            ], style={"display":"flex","alignItems":"center"}),
                                            style={"textAlign":"right","verticalAlign":"middle"},
                                        ),
                                        html.Td(html.Span("0.0%", id={"type":"inc-delta","ch":"IPA - TCE"},
                                                          style={"fontWeight":700,"fontSize":"12px","color":"#64748B","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(html.Span("", id={"type":"budget-pct-inc","ch":"IPA - TCE"}, style={"fontSize":"12px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(html.Span("", id={"type":"inc-revenue","ch":"IPA - TCE"},
                                                          style={"fontWeight":700,"color":"#4EAE46","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(78,174,70,0.06)"}),
                                        html.Td(html.Span("", id={"type":"inc-roi","ch":"IPA - TCE"},
                                                          style={"color":"#4EAE46","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(78,174,70,0.06)"}),
                                    ]),
                                    html.Tr([
                                        html.Td(html.Span("Sponsorships", style={"paddingLeft":"20px","fontSize":"13px"}),
                                                style={"textAlign":"left","verticalAlign":"middle"}),
                                        html.Td(
                                            html.Div(dbc.Switch(id={"type":"inc-include","ch":"Sponsorships"}, value=True),
                                                     style={"display":"flex","justifyContent":"center","alignItems":"center"}),
                                            style={"textAlign":"center","verticalAlign":"middle"},
                                        ),
                                        html.Td(html.Span(format_currency(float(INCREMENTAL_CHANNELS["Sponsorships"]["historical_spend"])),
                                                          style={"fontSize":"12px","color":"#64748B","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(
                                            html.Div([
                                        html.Button("−", id={"type":"inc-minus","ch":"Sponsorships"}, n_clicks=0, style={"width":"22px","height":"28px","flexShrink":0,"border":"1px solid #ced4da","borderRight":"none","borderRadius":"4px 0 0 4px","background":"#f8f9fa","cursor":"pointer","fontSize":"14px","lineHeight":"1","padding":"0"}),
                                                dcc.Input(id={"type":"inc-spend","ch":"Sponsorships"}, type="text", debounce=True,
                                                          value=format_currency(float(INCREMENTAL_CHANNELS["Sponsorships"]["historical_spend"])),
                                                          style={"flex":"1","textAlign":"right","minWidth":"80px",
                                                                 "border":"1px solid #ced4da","borderRadius":"0",
                                                                 "height":"28px","padding":"0 4px","fontSize":"12px"}),
                                        html.Button("+", id={"type":"inc-plus","ch":"Sponsorships"}, n_clicks=0, style={"width":"22px","height":"28px","flexShrink":0,"border":"1px solid #ced4da","borderLeft":"none","borderRadius":"0 4px 4px 0","background":"#f8f9fa","cursor":"pointer","fontSize":"14px","lineHeight":"1","padding":"0"}),
                                            ], style={"display":"flex","alignItems":"center"}),
                                            style={"textAlign":"right","verticalAlign":"middle"},
                                        ),
                                        html.Td(html.Span("0.0%", id={"type":"inc-delta","ch":"Sponsorships"},
                                                          style={"fontWeight":700,"fontSize":"12px","color":"#64748B","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(html.Span("", id={"type":"budget-pct-inc","ch":"Sponsorships"}, style={"fontSize":"12px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle"}),
                                        html.Td(html.Span("", id={"type":"inc-revenue","ch":"Sponsorships"},
                                                          style={"fontWeight":700,"color":"#4EAE46","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(78,174,70,0.06)"}),
                                        html.Td(html.Span("", id={"type":"inc-roi","ch":"Sponsorships"},
                                                          style={"color":"#4EAE46","fontVariantNumeric":"tabular-nums"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(78,174,70,0.06)"}),
                                    ]),
                                ] +
                                # Media (MMM) row
                                [
                                    html.Tr([
                                        html.Td(html.Strong("Media (MMM)"), style={"textAlign":"left","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td("",                          style={"textAlign":"center","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span(format_currency(_MEDIA_DEFAULT), style={"fontWeight":700,"fontSize":"13px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span(format_currency(_MEDIA_DEFAULT), id="budget-media-display", style={"fontWeight":700,"fontSize":"13px"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span("0.0%", id="budget-media-delta", style={"fontWeight":700,"fontSize":"12px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span("", id="budget-pct-media", style={"fontSize":"12px","color":"#64748B"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span("—", id="budget-media-revenue-display", style={"fontWeight":700,"fontSize":"13px","color":"#4EAE46"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                        html.Td(html.Span("—", id="budget-media-roi-display", style={"fontWeight":700,"fontSize":"13px","color":"#4EAE46"}),
                                                style={"textAlign":"right","verticalAlign":"middle","background":"rgba(14,41,97,0.04)"}),
                                    ]),
                                ]
                            ),
                        ],
                        bordered=True, size="sm", className="mt-0",
                    ),
                    html.Div(id="budget-hierarchy-msg", className="mt-2"),
                ]),

                # Derived totals summary strip
                html.Div([
                    html.Span("Total Non-MMM:", style={"fontSize": "12px", "fontWeight": 700, "color": "#374151"}),
                    html.Span("", id="nonmmm-total-display",
                              style={"fontSize": "13px", "fontWeight": 900, "color": "#0E2961",
                                     "marginLeft": "6px", "fontVariantNumeric": "tabular-nums"}),
                    html.Span(style={"width": "1px", "height": "16px", "background": "#cbd5e1",
                                     "margin": "0 14px", "display": "inline-block", "verticalAlign": "middle"}),
                    html.Span("Total Marketing Budget:", style={"fontSize": "12px", "fontWeight": 700, "color": "#374151"}),
                    html.Span("", id="total-mktg-display",
                              style={"fontSize": "13px", "fontWeight": 900, "color": "#0E2961",
                                     "marginLeft": "6px", "fontVariantNumeric": "tabular-nums"}),
                ], style={"display": "flex", "alignItems": "center", "padding": "8px 2px 2px 2px"}),

                # Stubs kept for callback compatibility
                html.Div(id="mmm-budget-display", style={"display": "none"}),
                html.Div(id="budget-comparison", style={"display": "none"}),
            ],
        ),


        # =================================================
        # 3. CHANNEL BUDGET PLANNING
        # =================================================
        html.Div(
            className="section",
            children=[
                html.Div("3. CHANNEL BUDGET PLANNING", className="section-title"),
                html.Div(
                    "Adjust budgets for each channel and see the expected impact on revenue. Freeze channels to keep their budgets fixed while optimizing the rest.",
                    className="comment",
                ),

                # ---------- MMM BUDGET CONTROL ----------
                html.Div(
                    id="total-budget-block",
                    children=[
                        html.Div(
                            dcc.Input(id="budget-pct", type="number", value=0, step=1, style={"display": "none"}),
                            id="budget-pct-col",
                            style={"display": "none"},
                        ),
                        html.Label(
                            "MMM Budget (USD)",
                            id="total-target-label",
                            style={"fontSize": "12px", "fontWeight": 700, "display": "block", "marginBottom": "6px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Button(
                                            "−", id="budget-minus", n_clicks=0,
                                            style={"width": "34px", "height": "36px", "flexShrink": 0,
                                                   "border": "1px solid #ced4da", "borderRight": "none",
                                                   "borderRadius": "4px 0 0 4px", "background": "#f8f9fa",
                                                   "cursor": "pointer", "fontSize": "18px", "lineHeight": "1"},
                                        ),
                                        dcc.Input(
                                            id="total-target",
                                            type="text",
                                            debounce=True,
                                            value=format_currency(_MEDIA_DEFAULT),
                                            placeholder="e.g., 14,267,487",
                                            style={"width": "160px", "height": "36px", "flexShrink": 0,
                                                   "border": "1px solid #ced4da", "borderRadius": "0",
                                                   "padding": "4px 8px", "textAlign": "right"},
                                        ),
                                        html.Button(
                                            "+", id="budget-plus", n_clicks=0,
                                            style={"width": "34px", "height": "36px", "flexShrink": 0,
                                                   "border": "1px solid #ced4da", "borderLeft": "none",
                                                   "borderRadius": "0 4px 4px 0", "background": "#f8f9fa",
                                                   "cursor": "pointer", "fontSize": "18px", "lineHeight": "1"},
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
                        html.Div(id="scale-warning", style={"marginTop": "8px"}),
                        html.Hr(style={"margin": "14px 0 10px 0", "borderColor": "#e5e7eb"}),
                    ],
                ),

                # ---------- OPTIMIZE ----------
                html.Div(
                    id="bounds-table-optimize",
                    children=[
                        # ── Above-table row: MMM totals (left) + Global bounds (right) ──
                        html.Div(
                            [
                                # LEFT: live MMM totals
                                html.Div(
                                    [
                                        html.Span("MMM Spend: ", style={"fontWeight": 700, "fontSize": "13px", "color": UI_MUTED}),
                                        html.Span(
                                            fmt_money_full(sum(base_investment.values())),
                                            id="mmm-spend-header",
                                            style={"fontWeight": 900, "fontSize": "13px", "color": NAVY_PRIMARY},
                                        ),
                                        html.Span("  |  ", style={"color": UI_MUTED, "margin": "0 8px", "fontWeight": 400}),
                                        html.Span("MMM Revenue: ", style={"fontWeight": 700, "fontSize": "13px", "color": UI_MUTED}),
                                        html.Span(
                                            fmt_money_full(sum(compute_response_for_spend(ch, base_investment[ch]) for ch in channels)),
                                            id="mmm-revenue-header",
                                            style={"fontWeight": 900, "fontSize": "13px", "color": GREEN_POSITIVE},
                                        ),
                                    ],
                                    style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "2px"},
                                ),
                                # RIGHT: global bounds controls
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label("LB %", style={"fontSize": "11px", "fontWeight": 700, "marginBottom": "3px", "display": "block", "color": UI_MUTED, "textAlign": "center"}),
                                                dcc.Input(id="global-lb", type="number", value=-20, style={"width": "68px", "textAlign": "center"}),
                                            ],
                                        ),
                                        html.Div(
                                            [
                                                html.Label("UB %", style={"fontSize": "11px", "fontWeight": 700, "marginBottom": "3px", "display": "block", "color": UI_MUTED, "textAlign": "center"}),
                                                dcc.Input(id="global-ub", type="number", value=20, style={"width": "68px", "textAlign": "center"}),
                                            ],
                                        ),
                                        dbc.Button(
                                            "APPLY BOUNDS",
                                            id="apply-global-bounds",
                                            color="secondary",
                                            className="btn-secondary",
                                            style={"height": "38px", "alignSelf": "flex-end", "whiteSpace": "nowrap", "fontSize": "11px"},
                                        ),
                                    ],
                                    style={"display": "flex", "gap": "8px", "alignItems": "flex-end"},
                                ),
                            ],
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px", "flexWrap": "wrap", "gap": "8px"},
                        ),

                        html.Div(id="global-bounds-msg"),

                        dbc.Table(
                            [
                                html.Thead(
                                    html.Tr(
                                        [
                                            html.Th("Channel", style={"textAlign": "left"}),
                                            html.Th("Include", style={"textAlign": "center"}),
                                            html.Th("Freeze", style={"textAlign": "center"}, title="Freezes this channel's budget during optimization"),
                                            html.Th("Budget ($)", style={"textAlign": "right"}),
                                            html.Th("Expected Revenue ($) ↻", style={"textAlign": "right", "color": GREEN_POSITIVE, "whiteSpace": "nowrap"}),
                                            html.Th("ROAS ↻", style={"textAlign": "right", "color": GREEN_POSITIVE}),
                                            html.Th("Lower Bound (%)", style={"textAlign": "center"}),
                                            html.Th("Upper Bound (%)", style={"textAlign": "center"}),
                                        ]
                                    )
                                ),
                                html.Tbody(channel_rows_optimize()),
                            ],
                            bordered=True,
                            size="sm",
                            className="mt-2",
                        ),

                        # Bottom totals — confirmation row
                        html.Div(
                            [
                                html.Span("Total Spend: ", style={"fontWeight": 700, "fontSize": "12px", "color": UI_MUTED}),
                                html.Span(
                                    fmt_money_full(sum(base_investment.values())),
                                    id="live-total-spend",
                                    style={"fontWeight": 800, "fontSize": "12px", "color": NAVY_PRIMARY, "marginRight": "20px"},
                                ),
                                html.Span("Total Revenue: ", style={"fontWeight": 700, "fontSize": "12px", "color": UI_MUTED}),
                                html.Span(
                                    fmt_money_full(sum(compute_response_for_spend(ch, base_investment[ch]) for ch in channels)),
                                    id="live-total-revenue",
                                    style={"fontWeight": 800, "fontSize": "12px", "color": GREEN_POSITIVE},
                                ),
                            ],
                            style={"textAlign": "right", "marginTop": "6px"},
                        ),

                        html.Div(id="bounds-feasibility", className="mt-2"),
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
    Input("goal", "value"),
)
def update_budget_description(goal):
    if (goal or "").lower() == "backward":
        return (
            "Define the target total revenue or response level. "
            "The model will minimize total investment while meeting this target."
        )
    return (
        "Define the total marketing budget and allocate any scenario / incremental investments. "
        "The remaining budget will be used for MMM optimization."
    )


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
    default_spend    = format_currency(_MEDIA_DEFAULT)
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
        "MMM Budget (USD)",
        "e.g., 14,267,487",
        default_spend,
    )

# =================================================
# Budget % ↔ USD sync (forward mode only)
# =================================================
_BASE_SPEND = _MEDIA_DEFAULT

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
    Output("section-budget", "style"),
    Input("goal", "value"),
)
def toggle_budget_section(goal):
    return {"display": "block"}


_ORIGINAL_MMM_BUDGET = _MEDIA_DEFAULT

@app.callback(
    Output("mmm-budget-display",    "children"),
    Output("mmm-budget-value",      "data"),
    Output("budget-comparison",     "children"),
    Output("budget-comparison",     "style"),
    Output("sticky-bar-budget-info","children"),
    Input("goal",         "value"),
    Input("total-target", "value"),
)
def update_remaining_mmm_budget(goal, total_target):
    hidden = {"display": "none"}

    if (goal or "").lower() == "backward":
        return None, None, None, hidden, None

    mmm_budget = parse_currency(total_target) or 0.0

    orig      = _ORIGINAL_MMM_BUDGET
    delta     = mmm_budget - orig
    delta_pct = (delta / orig * 100) if orig else 0.0
    sign      = "+" if delta >= 0 else ""
    delta_color = "#16a34a" if delta >= 0 else "#dc2626"

    sticky_info = html.Div(
        [
            html.Div(
                [
                    html.Span("MMM Budget  ", className="sticky-bar-label"),
                    html.Span(fmt_money_full(mmm_budget), className="sticky-bar-value"),
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
        None,       # mmm-budget-display hidden (Budget Summary removed)
        mmm_budget, # passed directly to optimizer
        None,
        hidden,
        sticky_info,
    )

# =================================================
# Proportional MMM Channel Scaling
# =================================================
@app.callback(
    Output({"type": "spend", "ch": ALL}, "value", allow_duplicate=True),
    Output("scale-warning", "children"),
    Input("total-target", "value"),
    State({"type": "spend",   "ch": ALL}, "value"),
    State({"type": "lock",    "ch": ALL}, "value"),
    State({"type": "include", "ch": ALL}, "value"),
    prevent_initial_call=True,
)
def scale_mmm_channels(total_target, spend_vals, lock_vals, include_vals):
    new_total = parse_currency(total_target)
    if new_total is None or new_total <= 0:
        raise PreventUpdate

    # Parse current spends (fall back to base_investment if blank/invalid)
    spends = []
    for ch, sv in zip(channels, spend_vals):
        v = parse_currency(sv) if sv else None
        spends.append(v if (v is not None and v > 0) else base_investment[ch])

    # Only consider included channels for budget math
    inc_mask  = [bool(inc) for inc in include_vals]
    lock_mask = [bool(lk) and bool(inc) for lk, inc in zip(lock_vals, include_vals)]

    old_total    = sum(s for s, inc in zip(spends, inc_mask) if inc)
    frozen_total = sum(s for s, fz  in zip(spends, lock_mask) if fz)
    adj_total    = old_total - frozen_total   # budget held by non-frozen included chs

    if old_total <= 0:
        raise PreventUpdate

    remaining = new_total - frozen_total

    # ── Edge cases ──────────────────────────────────────────────────────────
    warn_style = {"fontWeight": 600, "fontSize": "12px",
                  "padding": "5px 10px", "borderRadius": "4px", "marginBottom": "6px"}

    if remaining < 0:
        return no_update, html.Div(
            "⚠ Frozen channels exceed total MMM budget. Unfreeze channels or increase the budget.",
            style={**warn_style, "color": "#dc2626",
                   "background": "#fef2f2", "border": "1px solid #fca5a5"},
        )

    if adj_total <= 0:
        return no_update, html.Div(
            "⚠ All channels are frozen. Cannot adjust to new MMM budget.",
            style={**warn_style, "color": "#92400e",
                   "background": "#fffbeb", "border": "1px solid #fcd34d"},
        )

    # ── Scale non-frozen included channels ──────────────────────────────────
    scale = remaining / adj_total
    adj_indices = [i for i, (inc, fz) in enumerate(zip(inc_mask, lock_mask))
                   if inc and not fz]

    new_spends = list(spend_vals)
    rounded_vals = {}
    for i in adj_indices:
        rounded_vals[i] = round(spends[i] * scale)
        new_spends[i] = format_currency(rounded_vals[i])

    # Rounding correction: absorb any cent-level drift into the last channel
    actual_adj = sum(rounded_vals[i] for i in adj_indices)
    diff = round(new_total) - round(frozen_total) - actual_adj
    if diff != 0 and adj_indices:
        last = adj_indices[-1]
        corrected = rounded_vals[last] + diff
        new_spends[last] = format_currency(max(corrected, 0))

    return new_spends, None


# =================================================
# Hierarchical Budget Validation + Revenue / ROI + Deltas
# =================================================
@app.callback(
    Output("budget-d2c-display",   "children"),
    Output("budget-ipa-display",   "children"),
    Output("budget-media-display", "children"),
    Output("budget-pct-d2c",       "children"),
    Output("budget-pct-ipa",       "children"),
    Output("budget-pct-media",     "children"),
    Output({"type": "budget-pct-inc", "ch": ALL}, "children"),
    Output("budget-d2c-revenue",   "children"),
    Output("budget-d2c-roi",       "children"),
    Output("budget-ipa-revenue",   "children"),
    Output("budget-ipa-roi",       "children"),
    Output({"type": "inc-revenue", "ch": ALL}, "children"),
    Output({"type": "inc-roi",     "ch": ALL}, "children"),
    Output("budget-hierarchy-msg", "children"),
    Output("budget-d2c-delta",     "children"),
    Output("budget-ipa-delta",     "children"),
    Output("budget-media-delta",   "children"),
    Output({"type": "inc-delta",   "ch": ALL}, "children"),
    Output("nonmmm-total-display", "children"),
    Output("total-mktg-display",   "children"),
    Input({"type": "inc-spend",   "ch": ALL}, "value"),
    Input({"type": "inc-include", "ch": ALL}, "value"),
    Input("total-target", "value"),
)
def update_budget_hierarchy(inc_spend_vals, inc_include_vals, total_target):
    # total-target is now the MMM budget directly (no longer derived)
    mmm_budget = parse_currency(total_target) or 0.0
    inc_chs = INC_DOM_ORDER  # DOM-order matches ALL callback arg order

    d2c_spend = d2c_rev_cb = 0.0
    ipa_spend = ipa_rev_cb = 0.0
    pcts, revs, rois, ch_deltas = [], [], [], []

    for ch, spend_str, include in zip(inc_chs, inc_spend_vals, inc_include_vals):
        hist_s = float(INCREMENTAL_CHANNELS[ch]["historical_spend"])
        hist_r = float(INCREMENTAL_CHANNELS[ch]["historical_revenue"])

        spend = parse_currency(spend_str) if spend_str else hist_s
        if spend is None or spend < 0:
            spend = hist_s

        # Linear revenue model: revenue scales proportionally with spend
        rev = hist_r * (spend / hist_s) if hist_s > 0 else 0.0
        roi = rev / spend if spend > 0 else 0.0

        revs.append(fmt_money_short(rev) if rev > 0 else "—")
        rois.append(f"{roi:.2f}" if roi > 0 else "—")

        # delta vs baseline
        delta_pct = (spend - hist_s) / hist_s * 100 if hist_s > 0 else 0.0
        if abs(delta_pct) < 0.05:
            d_color = "#64748B"
            d_str   = "0.0%"
        elif delta_pct > 0:
            d_color = "#16a34a"
            d_str   = f"+{delta_pct:.1f}%"
        else:
            d_color = "#dc2626"
            d_str   = f"{delta_pct:.1f}%"
        ch_deltas.append(html.Span(d_str, style={"color": d_color, "fontWeight": 700}))

        if include:
            if ch in D2C_CHANNELS:
                d2c_spend += spend
                d2c_rev_cb += rev
            elif ch in IPA_CHANNELS:
                ipa_spend += spend
                ipa_rev_cb += rev

    # MMM budget is independent — media row shows it directly
    media = mmm_budget
    nonmmm_total = d2c_spend + ipa_spend
    grand_total  = nonmmm_total + mmm_budget

    # % of Total uses grand_total (Non-MMM + MMM) as denominator
    for ch, spend_str, include in zip(inc_chs, inc_spend_vals, inc_include_vals):
        if include:
            hist_s = float(INCREMENTAL_CHANNELS[ch]["historical_spend"])
            spend  = parse_currency(spend_str) if spend_str else hist_s
            if spend is None or spend < 0:
                spend = hist_s
            pcts.append(f"{spend / grand_total * 100:.1f}%" if grand_total > 0 else "")
        else:
            pcts.append("")

    def pct_str(val, tot):
        return f"{val / tot * 100:.1f}%" if tot > 0 else ""

    def roi_str(rev, spend):
        return f"{rev / spend:.2f}" if spend > 0 else "—"

    def delta_span(new_val, baseline):
        if baseline <= 0:
            return html.Span("0.0%", style={"color": "#64748B", "fontWeight": 700})
        dp = (new_val - baseline) / baseline * 100
        if abs(dp) < 0.05:
            return html.Span("0.0%", style={"color": "#64748B", "fontWeight": 700})
        color = "#16a34a" if dp > 0 else "#dc2626"
        s = f"+{dp:.1f}%" if dp > 0 else f"{dp:.1f}%"
        return html.Span(s, style={"color": color, "fontWeight": 700})

    return (
        fmt_money_full(d2c_spend),
        fmt_money_full(ipa_spend),
        fmt_money_full(media),
        pct_str(d2c_spend, grand_total),
        pct_str(ipa_spend, grand_total),
        pct_str(media, grand_total),
        pcts,
        fmt_money_short(d2c_rev_cb) if d2c_rev_cb > 0 else "—",
        roi_str(d2c_rev_cb, d2c_spend),
        fmt_money_short(ipa_rev_cb) if ipa_rev_cb > 0 else "—",
        roi_str(ipa_rev_cb, ipa_spend),
        revs,
        rois,
        None,   # budget-hierarchy-msg: no mismatch warning (budgets are independent)
        delta_span(d2c_spend, _D2C_DEFAULT),
        delta_span(ipa_spend, _IPA_DEFAULT),
        delta_span(media, _MEDIA_DEFAULT),
        ch_deltas,
        fmt_money_full(nonmmm_total),
        fmt_money_full(grand_total),
    )


# =================================================
# +/- Quick Adjust for incremental channel spends
# =================================================
@app.callback(
    Output({"type": "inc-spend", "ch": ALL}, "value", allow_duplicate=True),
    Input({"type": "inc-plus",  "ch": ALL}, "n_clicks"),
    Input({"type": "inc-minus", "ch": ALL}, "n_clicks"),
    State({"type": "inc-spend", "ch": ALL}, "value"),
    prevent_initial_call=True,
)
def adjust_inc_spend(plus_clicks, minus_clicks, spend_vals):
    if not ctx.triggered_id:
        raise PreventUpdate
    triggered = ctx.triggered_id  # already a dict for pattern-matching IDs
    t_type = triggered.get("type", "")
    t_ch   = triggered.get("ch", "")
    factor = 1.05 if t_type == "inc-plus" else 0.95

    result = list(spend_vals)
    for i, ch in enumerate(INC_DOM_ORDER):
        if ch == t_ch:
            cur = parse_currency(spend_vals[i]) if spend_vals[i] else None
            if cur is None or cur <= 0:
                cur = float(INCREMENTAL_CHANNELS[ch]["historical_spend"])
            result[i] = format_currency(round(cur * factor))
            break
    return result


# =================================================
# Sync LB/UB -> minmax-display + Apply Global Bounds
# =================================================
@app.callback(
    Output({"type": "lb", "ch": ALL}, "value"),
    Output({"type": "ub", "ch": ALL}, "value"),
    Output({"type": "minmax-display", "ch": ALL}, "children"),
    Output("global-bounds-msg", "children"),
    Input({"type": "lb", "ch": ALL}, "value"),
    Input({"type": "ub", "ch": ALL}, "value"),
    Input("apply-global-bounds", "n_clicks"),
    State("global-lb", "value"),
    State("global-ub", "value"),
)
def sync_bounds(lb_vals, ub_vals, n_apply, global_lb, global_ub):
    trigger = ctx.triggered_id or {}

    if trigger == "apply-global-bounds":
        new_lb = [float(global_lb)] * len(channels) if global_lb is not None else [float(v) for v in lb_vals]
        new_ub = [float(global_ub)] * len(channels) if global_ub is not None else [float(v) for v in ub_vals]
        displays = []
        for i, ch in enumerate(channels):
            base = base_investment[ch]
            min_v = base * (1 + new_lb[i] / 100)
            max_v = base * (1 + new_ub[i] / 100)
            displays.append(f"Min: {fmt_money_short(min_v)}  Max: {fmt_money_short(max_v)}")
        msg = html.Div("Applied global bounds to all channels.", className="status-success")
        return new_lb, new_ub, displays, msg

    new_lb, new_ub, displays = [], [], []
    for i, ch in enumerate(channels):
        base = base_investment[ch]
        lb_raw = lb_vals[i]
        ub_raw = ub_vals[i]
        lb = float(lb_raw) if lb_raw not in [None, ""] else 0.0
        ub = float(ub_raw) if ub_raw not in [None, ""] else 0.0
        min_v = base * (1 + lb / 100)
        max_v = base * (1 + ub / 100)
        new_lb.append(round(lb, 2))
        new_ub.append(round(ub, 2))
        displays.append(f"Min: {fmt_money_short(min_v)}  Max: {fmt_money_short(max_v)}")

    return new_lb, new_ub, displays, None


@app.callback(
    Output({"type": "revenue", "ch": ALL}, "children"),
    Output({"type": "roi", "ch": ALL}, "children"),
    Output("live-total-spend", "children"),
    Output("live-total-revenue", "children"),
    Output("mmm-spend-header", "children"),
    Output("mmm-revenue-header", "children"),
    Output("budget-media-revenue-display", "children"),
    Output("budget-media-roi-display", "children"),
    Input({"type": "spend", "ch": ALL}, "value"),
    Input({"type": "include", "ch": ALL}, "value"),
)
def update_live_revenue(spend_vals, include_vals):
    revenues = []
    rois = []
    total_spend = 0.0
    total_rev = 0.0

    for ch, spend_str, include in zip(channels, spend_vals, include_vals):
        spend = parse_currency(spend_str) if spend_str else base_investment[ch]
        if spend is None or spend <= 0:
            spend = base_investment[ch]

        if not include:
            revenues.append("—")
            rois.append("—")
            continue

        rev = compute_response_for_spend(ch, spend)
        roi = (rev / spend) if spend > 0 else 0.0

        revenues.append(fmt_money_short(rev) if rev > 0 else "—")
        rois.append(f"{roi:.2f}" if roi > 0 else "—")
        total_spend += spend
        total_rev += rev

    total_fmt = fmt_money_full(total_spend)
    rev_fmt = fmt_money_full(total_rev)
    media_roi = f"{total_rev / total_spend:.2f}" if total_spend > 0 else "—"
    return revenues, rois, total_fmt, rev_fmt, total_fmt, rev_fmt, fmt_money_short(total_rev) if total_rev > 0 else "—", media_roi

@app.callback(
    Output("bounds-feasibility", "children"),
    Input("goal", "value"),
    Input("mmm-budget-value", "data"),
    Input({"type": "include", "ch": ALL}, "value"),
    Input({"type": "lock", "ch": ALL}, "value"),
    Input({"type": "spend", "ch": ALL}, "value"),
    Input({"type": "lb", "ch": ALL}, "value"),
    Input({"type": "ub", "ch": ALL}, "value"),
)
def show_bounds_feasibility(goal, mmm_budget, include_vals, lock_vals, spend_vals, lb_vals, ub_vals):
    if (goal or "").lower() == "backward":
        return None
    if mmm_budget is None:
        raise PreventUpdate
    mmm_budget = float(mmm_budget)
    min_total = 0.0
    max_total = 0.0
    for ch, include, lock, spend_str, lb, ub in zip(
        channels, include_vals, lock_vals, spend_vals, lb_vals, ub_vals
    ):
        if not include:
            continue
        current_spend = parse_currency(spend_str) if spend_str else base_investment[ch]
        if current_spend is None:
            current_spend = base_investment[ch]
        if lock:
            min_total += current_spend
            max_total += current_spend
            continue
        lb = float(lb) if lb is not None else 0.0
        ub = float(ub) if ub is not None else 0.0
        min_total += current_spend * (1 + lb / 100)
        max_total += current_spend * (1 + ub / 100)
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
    Output("goal",                            "value"),
    Output("total-target",                    "value",    allow_duplicate=True),
    Output("budget-pct",                      "value",    allow_duplicate=True),
    Output("budget-slider",                   "value",    allow_duplicate=True),
    Output("global-lb",                       "value"),
    Output("global-ub",                       "value"),
    Output({"type": "include",    "ch": ALL}, "value",    allow_duplicate=True),
    Output({"type": "lock",       "ch": ALL}, "value",    allow_duplicate=True),
    Output({"type": "spend",      "ch": ALL}, "value",    allow_duplicate=True),
    Output({"type": "lb",         "ch": ALL}, "value",    allow_duplicate=True),
    Output({"type": "ub",         "ch": ALL}, "value",    allow_duplicate=True),
    Output({"type": "minmax-display", "ch": ALL}, "children", allow_duplicate=True),
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
    def_displays = [
        f"Min: {fmt_money_short(base_investment[ch] * (1 + def_lb / 100))}  Max: {fmt_money_short(base_investment[ch] * (1 + def_ub / 100))}"
        for ch in channels
    ]
    def_spends = [format_currency(base_investment[ch]) for ch in channels]
    return (
        "forward",
        format_currency(_MEDIA_DEFAULT),
        0,
        0,
        def_lb,
        def_ub,
        [True]  * n_ch,
        [False] * n_ch,
        def_spends,
        [def_lb] * n_ch,
        [def_ub] * n_ch,
        def_displays,
        [True]  * n_inc,
        [format_currency(float(INCREMENTAL_CHANNELS[ch]["historical_spend"])) for ch in INC_DOM_ORDER],
        None,
        None,
        None,
    )


@app.callback(
    Output("optimizer-results", "data"),
    Output("msg", "children"),
    Output("download-btn", "disabled"),
    Output("run-status", "children"),
    Input("run", "n_clicks"),
    State("goal", "value"),
    State("total-target", "value"),
    State({"type": "include", "ch": ALL}, "value"),
    State({"type": "lock", "ch": ALL}, "value"),
    State({"type": "spend", "ch": ALL}, "value"),
    State({"type": "lb", "ch": ALL}, "value"),
    State({"type": "ub", "ch": ALL}, "value"),
    State({"type": "inc-include", "ch": ALL}, "value"),
    State({"type": "inc-spend", "ch": ALL}, "value"),
    prevent_initial_call=True,
)
def run_optimizer(n_clicks, goal, total_target, include_vals, lock_vals, spend_vals, lb_vals, ub_vals, inc_include_vals, inc_spend_vals):
    # Build editable spends dict
    editable_spends = {}
    for ch, spend_str in zip(channels, spend_vals):
        parsed = parse_currency(spend_str) if spend_str else None
        editable_spends[ch] = parsed if parsed is not None else base_investment[ch]

    total_target_val = parse_currency(total_target)
    total_marketing_budget = total_target_val or 0.0

    incremental_spends = {}
    inc_channels = INC_DOM_ORDER

    if goal != "backward":
        for i, ch in enumerate(inc_channels):
            include = bool(inc_include_vals[i]) if inc_include_vals else False
            spend_val = parse_currency(inc_spend_vals[i]) if inc_spend_vals else None
            if include and spend_val is not None:
                incremental_spends[ch] = float(spend_val)

    # total-target IS the MMM budget directly (D2C/IPA are independent)
    mmm_budget = total_marketing_budget

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

    updated_data = update_data_from_ui(
        data=data,
        optimization_goal=goal,
        total_target=mmm_budget,
        channel_spends=editable_spends,
        bounds_dict=bounds,
    )
    updated_data["incremental_channels"] = INCREMENTAL_CHANNELS
    updated_data["incremental_spends"] = incremental_spends

    out = run_optimizer_for_ui(updated_data)

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
    invest_change_pct = ((k["opt_spend"] / k["actual_spend"]) - 1) * 100 if k["actual_spend"] else 0.0

    if incremental_spends:
        overlay_spend = sum(incremental_spends.values())
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
    Input("goal", "value"),
)
def toggle_mmm_budget_display(goal):
    if (goal or "").lower() == "backward":
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