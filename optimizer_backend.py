import json
import logging
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

# =================================================
# Logging
# =================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# =================================================
# Constants
# =================================================
EPS = 1e-8
DEFAULT_XTOL = 10
MAX_ITER = 100_000

# =================================================
# Data Classes for Type Safety
# =================================================
@dataclass
class OptimizationResult:
    success: bool
    results: Optional[pd.DataFrame]
    totals: Optional[Dict[str, float]]
    warnings: List[str]
    error: Optional[str]
    marginal_rois: Optional[Dict[str, float]] = None
    reallocation_summary: Optional[List[Dict]] = None

# =================================================
# Utility Functions
# =================================================
def convert_lists_to_numpy(obj: Any) -> Any:
    """Recursively convert lists to numpy arrays."""
    if isinstance(obj, list):
        return np.array([convert_lists_to_numpy(x) for x in obj])
    elif isinstance(obj, dict):
        return {k: convert_lists_to_numpy(v) for k, v in obj.items()}
    return obj

def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Safe division that returns NaN for zero denominators."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(denominator != 0, numerator / denominator, np.nan)
    return result

# =================================================
# Hill Curve Functions
# =================================================
def hill_curve(x: np.ndarray, L: float, alpha: float, theta: float, eps: float = 1e-8) -> np.ndarray:
    """
    Hill saturation curve: L * x^α / (x^α + θ^α)
    """
    x = np.clip(x, 0, None)
    x_alpha = np.power(x, alpha)
    theta_alpha = np.power(theta, alpha)
    return L * x_alpha / (x_alpha + theta_alpha + eps)

def hill_curve_derivative(x: np.ndarray, L: float, alpha: float, theta: float) -> np.ndarray:
    """Analytical derivative of Hill curve for gradient-based optimization."""
    x = np.clip(x, EPS, None)
    x_alpha = np.power(x, alpha)
    theta_alpha = np.power(theta, alpha) + EPS

    numerator = L * alpha * theta_alpha * np.power(x, alpha - 1)
    denominator = np.power(x_alpha + theta_alpha, 2)

    return numerator / denominator

# =================================================
# Core Calculation Functions
# =================================================
def get_total_contribution(
    channels: List[str],
    media: np.ndarray,
    proportion: Dict[str, np.ndarray],
    correction: Dict[str, np.ndarray],
    constant: float,
    s_curve_params: Dict[str, Dict],
) -> float:
    """Calculate total response contribution from all channels."""
    total_contribution = 0.0

    for i, ch in enumerate(channels):
        params = s_curve_params[ch]
        spend_prop = media[i] * proportion[ch]

        channel_contribution = (
            hill_curve(spend_prop, params["L"], params["alpha"], params["theta"]).sum()
            + correction[ch].sum()
        )
        total_contribution += channel_contribution

    return total_contribution + constant

def get_total_spends(
    media: np.ndarray,
    conversion_ratio: Dict[str, np.ndarray],
    channels: List[str],
) -> float:
    """Calculate total spend across all channels."""
    total = 0.0
    for i, ch in enumerate(channels):
        cr = np.asarray(conversion_ratio[ch], dtype=float)
        cr_scalar = float(np.mean(cr))
        total += media[i] * cr_scalar
    return total

# =================================================
# Main Optimizer (matches old behavior: no feasibility pre-check)
# =================================================
def optimizer(
    optimization_goal: str,
    media: Dict[str, np.ndarray],
    proportion: Dict[str, np.ndarray],
    correction: Dict[str, np.ndarray],
    constant: List[float],
    s_curve_params: Dict[str, Dict],
    conversion_ratio: Dict[str, np.ndarray],
    bounds_dict: Dict[str, List[float]],
    total_target: float,
    xtol_tolerance_per: float,
) -> Tuple[Dict[str, np.ndarray], bool, str]:
    """
    Main optimization function.

    Args:
        optimization_goal: "forward" (maximize response given budget) or
                           "backward" (minimize spend given response target)
        media: Current media values per channel
        proportion: Weekly proportion of spend per channel
        correction: Correction terms per channel
        constant: Baseline constant terms
        s_curve_params: Hill curve parameters per channel
        conversion_ratio: Spend-to-media conversion per channel
        bounds_dict: [lower%, upper%] bounds per channel
        total_target: Target budget (forward) or response (backward)
        xtol_tolerance_per: Convergence tolerance percentage

    Returns:
        Tuple of (optimized_media_dict, success_flag, message)
    """
    channels = sorted(media.keys())
    actual_media = np.array([media[ch] for ch in channels], dtype=float)
    num_channels = len(actual_media)
    constant_sum = float(sum(constant))

    # -------------------------------------------------
    # Objective (SAFE) – identical logic to old version
    # -------------------------------------------------
    def objective_fun(media_vec: np.ndarray) -> float:
        try:
            if optimization_goal == "forward":
                # Maximize contribution (minimize negative)
                val = -get_total_contribution(
                    channels,
                    media_vec,
                    proportion,
                    correction,
                    constant_sum,
                    s_curve_params,
                )
            else:
                # Minimize spend
                val = get_total_spends(media_vec, conversion_ratio, channels)

            if not np.isfinite(val):
                return 1e20

            return val

        except Exception as e:
            logger.warning(f"Objective function error: {e}")
            return 1e20

    def constraint_fun(media_vec: np.ndarray) -> float:
        if optimization_goal == "forward":
            return get_total_spends(media_vec, conversion_ratio, channels)
        else:
            return get_total_contribution(
                channels,
                media_vec,
                proportion,
                correction,
                constant_sum,
                s_curve_params,
            )

    constraints = {
        "type": "eq",
        "fun": lambda media_vec: constraint_fun(media_vec) - total_target,
    }

    # Bounds (same formula as old code; no clipping to 0 here)
    bounds = [
        (
            actual_media[i] * (1 + bounds_dict[channels[i]][0] / 100),
            actual_media[i] * (1 + bounds_dict[channels[i]][1] / 100),
        )
        for i in range(num_channels)
    ]

    initial_guess = np.array(actual_media, dtype=float)
    positive_media = actual_media[actual_media > 0]
    if positive_media.size > 0:
        xtol = max(DEFAULT_XTOL, (xtol_tolerance_per / 100) * np.min(positive_media))
    else:
        xtol = DEFAULT_XTOL

    logger.info(
        f"Starting optimization: goal={optimization_goal}, target={total_target:,.0f}"
    )

    result: OptimizeResult = minimize(
        objective_fun,
        initial_guess,
        method="trust-constr",
        constraints=constraints,
        bounds=bounds,
        options={
            "disp": False,
            "xtol": xtol,
            "gtol": 1e-4,
            "maxiter": MAX_ITER,
        },
    )

    optimized_media_array = np.clip(result.x, 0.0, None)
    optimized_media = {channels[i]: optimized_media_array[i] for i in range(num_channels)}

    if result.success:
        message = "Optimization successful"
    else:
        message = f"Optimizer did not converge: {result.message}"

    logger.info(message)
    return optimized_media, result.success, message

# =================================================
# Validation (restored old behavior)
# =================================================
def validate_optimization(
    optimized_spends: pd.DataFrame,
    data: Dict,
) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Validate optimization results against constraints.

    This is the same logic as your original working version:
    - tol = 0.005 (0.5%)
    - asymmetric bounds: lower - |lower|*tol, upper + |upper|*tol
    - target within tol * target.
    """
    tol = 0.005
    bounds = data["bounds_dict"]
    goal = data["optimization_goal"].lower()
    target = data["total_target"]
    base = sum(data["constant"])

    # Check per‑channel % change bounds
    for channel, row in optimized_spends.iterrows():
        if channel not in bounds:
            continue
        lower, upper = bounds[channel]
        val = row["Δ Spend (%)"]

        if np.isnan(val):
            continue

        eps = 1e-6  # small slack to avoid float rounding issues

        if not (lower - abs(lower) * tol - eps <= val <= upper + abs(upper) * tol + eps):
            logger.warning(
                f"Channel {channel} outside bounds: {val:.2f}% "
                f"not in [{lower - abs(lower)*tol:.2f}, {upper + abs(upper)*tol:.2f}]"
            )
            return None, False

    # Check target constraint
    if goal == "forward":
        total_spends = optimized_spends["Optimized Spend"].sum()
        if abs(total_spends - target) > tol * target:
            logger.warning(
                f"Total spend {total_spends:,.0f} != target {target:,.0f} "
                f"(tol={tol*100:.1f}%)"
            )
            return None, False
    else:
        total_response = optimized_spends["Optimized Response Metric"].sum() + base
        if abs(total_response - target) > tol * target:
            logger.warning(
                f"Total response {total_response:,.0f} != target {target:,.0f} "
                f"(tol={tol*100:.1f}%)"
            )
            return None, False

    return optimized_spends, True

# =================================================
# Results Builder
# =================================================
def build_optimized_results(
    optimized_media: Dict[str, np.ndarray],
    data: Dict,
) -> Tuple[Optional[pd.DataFrame], bool, List[str]]:
    """Build comprehensive results DataFrame from optimization output."""
    warnings: List[str] = []
    modified_metrics: Dict[str, Dict[str, float]] = {}

    for ch in optimized_media:
        params = data["s_curve_params"][ch]

        weekly_spend = np.asarray(
            data["media"][ch] * data["proportion"][ch],
            dtype=float,
        )
        optimized_weekly = np.asarray(
            optimized_media[ch] * data["proportion"][ch],
            dtype=float,
        )

        weekly_spend = np.clip(weekly_spend, 0.0, None)
        optimized_weekly = np.clip(optimized_weekly, 0.0, None)

        actual_metric = (
            hill_curve(
                weekly_spend,
                params["L"],
                params["alpha"],
                params["theta"],
            ).sum()
            + data["correction"][ch].sum()
        )

        optimized_metric = (
            hill_curve(
                optimized_weekly,
                params["L"],
                params["alpha"],
                params["theta"],
            ).sum()
            + data["correction"][ch].sum()
        )

        modified_metrics[ch] = {
            "actual_metric": float(actual_metric),
            "optimized_metric": float(optimized_metric),
        }

    # Spend series (in money units, not media)
    conversion_rates = pd.Series(
        {
            ch: float(np.mean(np.asarray(data["conversion_ratio"][ch], dtype=float)))
            for ch in data["conversion_ratio"]
        }
    )

    actual_media_series = pd.Series(data["media"], name="Actual/Input Spend")
    optimized_media_series = pd.Series(optimized_media, name="Optimized Spend")

    actual_spend = actual_media_series * conversion_rates
    optimized_spend = optimized_media_series * conversion_rates

    actual_metric_series = pd.Series(
        {ch: modified_metrics[ch]["actual_metric"] for ch in modified_metrics},
        name="Actual Response Metric",
    )
    optimized_metric_series = pd.Series(
        {ch: modified_metrics[ch]["optimized_metric"] for ch in modified_metrics},
        name="Optimized Response Metric",
    )

    # NOTE:
    # These are ROAS-style ratios: Response / Spend.
    # Kept as "ROI" for backward compatibility with existing UI/table IDs.
    actual_roi = (
        actual_metric_series / actual_spend.replace(0, np.nan)
    ).rename("Actual ROI")
    optimized_roi = (
        optimized_metric_series / optimized_spend.replace(0, np.nan)
    ).rename("Optimized ROI")

    optimized_spends = pd.concat(
        [
            actual_spend.rename("Actual/Input Spend"),
            optimized_spend.rename("Optimized Spend"),
            actual_metric_series,
            optimized_metric_series,
            actual_roi,
            optimized_roi,
        ],
        axis=1,
    )

    optimized_spends["Δ Spend (Abs)"] = (
        optimized_spends["Optimized Spend"]
        - optimized_spends["Actual/Input Spend"]
    )
    optimized_spends["Δ Spend (%)"] = (
        optimized_spends["Δ Spend (Abs)"]
        / optimized_spends["Actual/Input Spend"]
    ) * 100

    # Validation (old behavior)
    optimized_spends, success = validate_optimization(optimized_spends, data)
    if not success:
        warnings.append("Validation failed: results may not satisfy all constraints")
        return None, False, warnings
    
    optimized_spends["Channel Type"] = "Optimized"
    
    return optimized_spends, True, warnings

# =================================================
# Marginal ROI Calculator
# =================================================
def calculate_marginal_rois(
    optimized_media: Dict[str, float],
    data: Dict,
) -> Dict[str, float]:
    """Calculate marginal ROI for each channel at optimized spend levels."""
    marginal_rois: Dict[str, float] = {}

    for ch in optimized_media:
        params = data["s_curve_params"][ch]

        # Ensure proportion is a NumPy array, not a Python list
        prop = np.asarray(data["proportion"][ch], dtype=float)

        # optimized_media[ch] is a scalar (float), prop is an array
        current_spend = np.asarray(optimized_media[ch] * prop, dtype=float)

        # Derivative at current point
        marginal_response = hill_curve_derivative(
            current_spend,
            params["L"],
            params["alpha"],
            params["theta"],
        ).mean()

        marginal_rois[ch] = float(marginal_response)

    return marginal_rois

# =================================================
# Update data from UI
# =================================================
def update_data_from_ui(
    data: Dict,
    optimization_goal: str,
    total_target: float,
    channel_spends: Dict[str, float],
    bounds_dict: Dict[str, List[float]],
) -> Dict:
    """
    Take the base `data` dict and update it with values coming from the UI.

    Args:
        data: base model config loaded from optimizer_input.json
        optimization_goal: "forward" (maximize response) or "backward" (minimize spend)
        total_target: budget (forward) or response target (backward)
        channel_spends: dict of {channel: spend} from UI (current/base spends)
        bounds_dict: dict of {channel: [lower_pct, upper_pct]} from UI

    Returns:
        A new data dict ready to be passed into `run_optimizer_for_ui`.
    """
    updated = deepcopy(data)

    updated["optimization_goal"] = optimization_goal.lower()
    updated["total_target"] = float(total_target) if total_target is not None else 0.0
    updated["spends"] = {ch: float(val) for ch, val in channel_spends.items()}
    updated["bounds_dict"] = {
        ch: [float(bounds_dict[ch][0]), float(bounds_dict[ch][1])]
        for ch in bounds_dict
    }

    # Default tolerance if not set
    if "xtol_tolerance_per" not in updated or updated["xtol_tolerance_per"] is None:
        updated["xtol_tolerance_per"] = 1.0

    return updated

# =================================================
# Main Entry Points
# =================================================
def run_optimizer(data: Dict) -> Tuple[Optional[pd.DataFrame], bool, List[str]]:
    """Run the full optimization pipeline (safe wrapper)."""
    data = deepcopy(data)

    # Normalize spends
    for ch in data["spends"]:
        data["spends"][ch] = float(data["spends"][ch])

    # Ensure arrays
    for key in ["conversion_ratio", "proportion", "correction"]:
        for ch in data[key]:
            data[key][ch] = np.asarray(data[key][ch], dtype=float)

    # Convert spends (money) to media units
    data["media"] = {}
    for ch in data["spends"]:
        conv = np.maximum(np.asarray(data["conversion_ratio"][ch], dtype=float), EPS)
        conv_scalar = float(np.mean(conv))
        media_val = data["spends"][ch] / conv_scalar
        data["media"][ch] = np.clip(media_val, 0.0, None)

    optimized_media, opt_success, message = optimizer(
        optimization_goal=data["optimization_goal"],
        media=data["media"],
        proportion=data["proportion"],
        correction=data["correction"],
        constant=data["constant"],
        s_curve_params=data["s_curve_params"],
        bounds_dict=data["bounds_dict"],
        conversion_ratio=data["conversion_ratio"],
        total_target=data["total_target"],
        xtol_tolerance_per=data["xtol_tolerance_per"],
    )

    if not opt_success:
        return None, False, [message]

    optimized_spends, build_success, warnings = build_optimized_results(
        optimized_media, data
    )

    if not build_success:
        return None, False, warnings

    # Attach optimized_media if needed later
    data["optimized_media"] = optimized_media

    return optimized_spends, True, warnings

def simulate_incremental_channels(incremental_channels, user_spends):
    rows = []

    for ch, spend in user_spends.items():  # 🔑 ONLY included channels
        meta = incremental_channels.get(ch, {})
        hist_spend = float(meta.get("historical_spend", 0))
        hist_rev = float(meta.get("historical_revenue", 0))

        efficiency = hist_rev / hist_spend if hist_spend > 0 else 0.0
        new_rev = spend * efficiency

        rows.append({
            "Channel": ch,
            "Channel Type": "Incremental (Linear)",
            "Actual/Input Spend": 0.0,          # 🔑 ZERO when excluded historically
            "Optimized Spend": spend,
            "Δ Spend (Abs)": spend,
            "Δ Spend (%)": np.nan,
            "Actual Response Metric": 0.0,       # 🔑 ZERO unless included
            "Optimized Response Metric": new_rev,
            "Actual ROI": np.nan,
            "Optimized ROI": efficiency,
        })

    return pd.DataFrame(rows)


def run_optimizer_for_ui(data: Dict) -> OptimizationResult:
    """UI-friendly wrapper that returns structured results."""

    # ---- Validation ----
    for ch, cr in data["conversion_ratio"].items():
        if cr is None or np.any(np.asarray(cr) <= 0):
            return OptimizationResult(
                success=False,
                results=None,
                totals=None,
                warnings=[],
                error=f"Invalid conversion_ratio for {ch}",
            )

    # ---- Run MMM optimizer ----
    optimized_spends, success, warnings = run_optimizer(data)
    if not success or optimized_spends is None:
        err_msg = warnings[0] if warnings else "Optimization failed"
        return OptimizationResult(
            success=False,
            results=None,
            totals=None,
            warnings=warnings,
            error=err_msg,
        )

    # ---- Ensure Channel column exists (CRITICAL) ----
    optimized_spends = optimized_spends.reset_index().rename(
        columns={"index": "Channel"}
    )

    # ---- ENSURE Channel Type EXISTS FOR ALL ROWS ----
    if "Channel Type" not in optimized_spends.columns:
        optimized_spends["Channel Type"] = "Optimized"

    # Safety: fill any missing
    optimized_spends["Channel Type"] = optimized_spends["Channel Type"].fillna("Optimized")


    # ---- Incremental channels (scenario-only) ----
    incremental_df = None
    if "incremental_channels" in data and data["incremental_channels"]:
        user_inc_spends = data.get("incremental_spends", {})
        incremental_df = simulate_incremental_channels(
            data["incremental_channels"],
            user_inc_spends,
        )

    # ---- Merge MMM + Incremental ----
    if incremental_df is not None and not incremental_df.empty:
        optimized_spends = pd.concat(
            [optimized_spends, incremental_df],
            axis=0,
            ignore_index=True,
        )

    # ---- Totals (INCLUDE baseline once) ----
    base_constant = sum(data["constant"])

    totals = {
        "actual_spend": float(optimized_spends["Actual/Input Spend"].sum()),
        "optimized_spend": float(optimized_spends["Optimized Spend"].sum()),
        "actual_response": float(
            optimized_spends["Actual Response Metric"].sum() + base_constant
        ),
        "optimized_response": float(
            optimized_spends["Optimized Response Metric"].sum() + base_constant
        ),
    }

    totals["spend_change_pct"] = float(
        safe_divide(
            totals["optimized_spend"] - totals["actual_spend"],
            totals["actual_spend"],
        )
        * 100
    )
    totals["response_change_pct"] = float(
        safe_divide(
            totals["optimized_response"] - totals["actual_response"],
            totals["actual_response"],
        )
        * 100
    )

    goal = data["optimization_goal"].lower()
    totals["mode"] = goal
    totals["target_type"] = "budget" if goal == "forward" else "response"
    totals["target_value"] = float(data["total_target"])

    # ---- Marginal ROAS (MMM ONLY) ----
    optimized_media_for_mroi: Dict[str, float] = {}

    for _, row in optimized_spends.iterrows():
        ch = row["Channel"]

        if ch not in data["conversion_ratio"]:
            continue  # skip incremental

        conv = np.maximum(
            np.asarray(data["conversion_ratio"][ch], dtype=float),
            EPS,
        )
        conv_scalar = float(np.mean(conv))
        optimized_media_for_mroi[ch] = row["Optimized Spend"] / conv_scalar

    marginal_rois = calculate_marginal_rois(
        optimized_media=optimized_media_for_mroi,
        data=data,
    )


    # ---- Reallocation summary (ALL channels) ----
    reallocation_summary: List[Dict[str, Any]] = []
    for _, row in optimized_spends.iterrows():
        reallocation_summary.append(
            {
                "channel": row["Channel"],
                "actual_spend": float(row["Actual/Input Spend"]),
                "optimized_spend": float(row["Optimized Spend"]),
                "delta_spend": float(row["Δ Spend (Abs)"]),
                "delta_spend_pct": float(row["Δ Spend (%)"]),
            }
        )

    return OptimizationResult(
        success=True,
        results=optimized_spends,
        totals=totals,
        warnings=warnings,
        error=None,
        marginal_rois=marginal_rois,
        reallocation_summary=reallocation_summary,
    )

# =================================================
# Optional: load base data (if you still want a module-level `data`)
# =================================================
BASE_DIR = Path(__file__).parent
optimizer_input_path = BASE_DIR / "optimizer_input.json"
if optimizer_input_path.exists():
    with open(optimizer_input_path, "r") as f:
        base_data = json.load(f)
    base_data = convert_lists_to_numpy(base_data)
else:
    base_data = None
