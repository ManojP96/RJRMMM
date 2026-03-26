"""
build_response_curves.py
========================
Fits Hill (S-curve) response curves at national level for each media channel
using model-decomposed contribution data, then writes a new optimizer_input.json.

Channels: Audio, Display, GSTV, OOH, Search, Video

Input : final_model_contribution_spends_data.csv
Output: optimizer_input.json  (overwrites existing)

Usage:
    python build_response_curves.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# =============================================================================
# Config
# =============================================================================
BASE_DIR = Path(__file__).parent

INPUT_CSV   = BASE_DIR / "final_model_contribution_spends_data.csv"
OUTPUT_JSON = BASE_DIR / "optimizer_input.json"

CHANNELS = ["Audio", "Display", "GSTV", "OOH", "Search", "Video"]

SPEND_COLS = {
    "Audio":   "Audio_spends",
    "Display": "Display_spends",
    "GSTV":    "GSTV_spends",
    "OOH":     "OOH_spends",
    "Search":  "Search_spends",
    "Video":   "Video_spends",
}
REV_COLS = {
    "Audio":   "audio_revenue",
    "Display": "display_revenue",
    "GSTV":    "gstv_revenue",
    "OOH":     "ooh_revenue",
    "Search":  "search_revenue",
    "Video":   "video_revenue",
}

# Incremental (non-MMM) channels kept from existing config
INCREMENTAL_CHANNELS = {
    "D2C Reach":      {"historical_spend": 2657761,  "historical_revenue": 10737664},
    "D2C Rewards":    {"historical_spend": 229446,   "historical_revenue": 5782981},
    "IPA - Strat Cities": {"historical_spend": 1619017, "historical_revenue": 177665},
    "IPA":            {"historical_spend": 4818776,  "historical_revenue": 270754},
    "Sponsorships":   {"historical_spend": 11226052, "historical_revenue": 171116},
    "IPA - TCE":      {"historical_spend": 2692919,  "historical_revenue": 1792753},
}

BOUNDS_DICT = {ch: [-20, 20] for ch in CHANNELS}

# =============================================================================
# Hill curve
# =============================================================================
def hill_curve(x, L, alpha, theta, eps=1e-8):
    """Hill saturation curve: L * x^alpha / (theta^alpha + x^alpha)"""
    x     = np.clip(x, 0, None)
    x_a   = np.power(x, alpha)
    t_a   = np.power(theta, alpha)
    return L * x_a / (x_a + t_a + eps)


def fit_hill(x: np.ndarray, y: np.ndarray, channel: str):
    """
    Fit Hill curve to (x=weekly spend, y=weekly revenue contribution).
    Uses only weeks where spend > 0.
    Returns dict with keys L, alpha, theta.
    """
    mask = x > 0
    xf, yf = x[mask], y[mask]

    n = int(mask.sum())
    print(f"  {channel}: {n} non-zero spend weeks | "
          f"spend [{xf.min():,.0f}, {xf.max():,.0f}] | "
          f"revenue [{yf.min():,.0f}, {yf.max():,.0f}]")

    if n < 3:
        raise ValueError(
            f"Too few data points for {channel} ({n}). "
            "Cannot fit a Hill curve reliably."
        )

    L_init     = max(float(yf.max()) * 2.5, 1.0)
    alpha_init = 1.0
    # theta: start at median spend (half-saturation at median)
    theta_init = float(np.median(xf))

    p0 = [L_init, alpha_init, theta_init]

    # Constrained bounds:
    # - alpha: 0.5-3.0 ensures meaningful diminishing-returns shape
    # - theta: must be >= 10% of max spend to avoid degenerate step-function behavior
    #          must be <= 5× max spend so the curve isn't stuck in near-linear zone
    theta_lb = max(float(xf.max()) * 0.10, float(xf.min()))
    theta_ub = float(xf.max()) * 5.0

    lb = [0.0,  0.5,  theta_lb]
    ub = [float(yf.max()) * 100.0, 3.0, theta_ub]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = curve_fit(
            hill_curve, xf, yf,
            p0=p0, bounds=(lb, ub),
            maxfev=100_000,
            method="trf",
        )

    L, alpha, theta = popt

    # R² on the non-zero weeks
    y_hat = hill_curve(xf, L, alpha, theta)
    ss_res = np.sum((yf - y_hat) ** 2)
    ss_tot = np.sum((yf - yf.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print(f"         -> L={L:,.0f}  alpha={alpha:.3f}  theta={theta:,.0f}  R2={r2:.3f}")

    return {"L": float(L), "alpha": float(alpha), "theta": float(theta)}


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("Building national-level Hill response curves")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load & aggregate to national weekly
    # ------------------------------------------------------------------
    print("\n[1] Loading data …")
    df = pd.read_csv(INPUT_CSV)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])

    agg_cols = (
        list(SPEND_COLS.values())
        + list(REV_COLS.values())
        + ["base_revenue"]
    )
    national = (
        df.groupby("week_start_date")[agg_cols]
        .sum()
        .reset_index()
        .sort_values("week_start_date")
        .reset_index(drop=True)
    )

    print(f"   Weeks: {len(national)}  "
          f"({national['week_start_date'].min().date()} to "
          f"{national['week_start_date'].max().date()})")

    # ------------------------------------------------------------------
    # 2. Fit Hill curves (use ALL available weeks for maximum data)
    # ------------------------------------------------------------------
    print("\n[2] Fitting Hill curves …")
    s_curve_params = {}
    for ch in CHANNELS:
        x = national[SPEND_COLS[ch]].values.astype(float)
        y = national[REV_COLS[ch]].values.astype(float)
        params = fit_hill(x, y, ch)
        s_curve_params[ch] = params

    # ------------------------------------------------------------------
    # 3. Planning period: last 52 weeks for proportions / spends / baseline
    # ------------------------------------------------------------------
    print("\n[3] Computing planning-period inputs (last 52 weeks) …")
    planning = national.tail(52).reset_index(drop=True)
    n_plan   = len(planning)
    print(f"   Period: {planning['week_start_date'].min().date()} to "
          f"{planning['week_start_date'].max().date()}  ({n_plan} weeks)")

    spends     = {}
    proportion = {}
    correction = {}

    for ch in CHANNELS:
        weekly_spends = planning[SPEND_COLS[ch]].values.astype(float)
        total         = float(weekly_spends.sum())

        spends[ch] = total

        if total > 0:
            proportion[ch] = (weekly_spends / total).tolist()
        else:
            # Channel had zero spend in planning period — use uniform fallback
            print(f"   WARNING: {ch} has zero spend in planning period; "
                  "using uniform proportion as fallback.")
            proportion[ch] = [1.0 / n_plan] * n_plan

        correction[ch] = [0.0] * n_plan

    # Weekly baseline (base_revenue) for planning period
    constant = planning["base_revenue"].tolist()

    # Total budget = sum of all media spends in planning period
    total_target = float(sum(spends.values()))

    # Conversion ratio: 1.0 (curves fitted in spend space, not impression space)
    conversion_ratio = {ch: 1.0 for ch in CHANNELS}

    # ------------------------------------------------------------------
    # 4. Calibrate L so predicted response matches actual at current spend
    # ------------------------------------------------------------------
    print("\n[4] Calibrating L to match actual contributions …")
    print(f"   {'Channel':10s} {'Spend':>12s} {'Pred Rev':>12s} {'Actual Rev':>12s} {'Scale':>8s}")

    actual_revenue = {
        ch: float(planning[REV_COLS[ch]].sum())
        for ch in CHANNELS
    }

    for ch in CHANNELS:
        params = s_curve_params[ch]
        spend  = spends[ch]
        prop   = proportion[ch]

        pred = sum(
            hill_curve(spend * p, params["L"], params["alpha"], params["theta"])
            for p in prop
        )

        actual = actual_revenue[ch]

        if pred > 0:
            scale = actual / pred
            params["L"] = params["L"] * scale
        else:
            scale = 1.0

        print(f"   {ch:10s} {spend:>12,.0f} {pred:>12,.0f} {actual:>12,.0f} {scale:>8.3f}")

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print("\n[5] Spend summary (planning period):")
    for ch in CHANNELS:
        print(f"   {ch:10s}: ${spends[ch]:>12,.0f}")
    print(f"   {'TOTAL':10s}: ${total_target:>12,.0f}")
    print(f"   Baseline (sum): ${sum(constant):>12,.0f}")

    # ------------------------------------------------------------------
    # 6. Build and write optimizer_input.json
    # ------------------------------------------------------------------
    optimizer_input = {
        "optimization_goal":  "forward",
        "s_curve_params":     s_curve_params,
        "proportion":         proportion,
        "conversion_ratio":   conversion_ratio,
        "total_target":       total_target,
        "bounds_dict":        BOUNDS_DICT,
        "xtol_tolerance_per": 1,
        "correction":         correction,
        "constant":           constant,
        "spends":             spends,
        "incremental_channels": INCREMENTAL_CHANNELS,
    }

    print(f"\n[6] Writing -> {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(optimizer_input, f, indent=4)

    print("\nDone. Restart the Dash app to pick up the new parameters.")
    print("=" * 60)


if __name__ == "__main__":
    main()
