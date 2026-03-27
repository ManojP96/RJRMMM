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

    # Bounds (absolute values)
    bounds = [
        (
            actual_media[i] * (1 + bounds_dict[channels[i]][0] / 100),
            actual_media[i] * (1 + bounds_dict[channels[i]][1] / 100),
        )
        for i in range(num_channels)
    ]

    # ── Separate FIXED channels (lb==ub, i.e. frozen) from FREE channels ──────
    # SLSQP can return the initial guess unchanged when some bounds are
    # degenerate (lb==ub), because the initial guess already satisfies the
    # budget constraint and SLSQP perceives no room to improve.  Fix: remove
    # fixed channels from the variable space entirely, adjust the target, and
    # optimize only over free channels.
    EPS_BOUND = 1.0  # treat as fixed when ub - lb < $1
    fixed_mask  = [abs(bounds[i][1] - bounds[i][0]) < EPS_BOUND for i in range(num_channels)]
    free_indices  = [i for i in range(num_channels) if not fixed_mask[i]]
    fixed_indices = [i for i in range(num_channels) if     fixed_mask[i]]

    free_channels  = [channels[i] for i in free_indices]
    fixed_channels = [channels[i] for i in fixed_indices]
    free_actual    = actual_media[np.array(free_indices, dtype=int)] if free_indices  else np.array([], dtype=float)
    fixed_actual   = actual_media[np.array(fixed_indices, dtype=int)] if fixed_indices else np.array([], dtype=float)
    free_bounds    = [bounds[i] for i in free_indices]

    # Compute fixed channels' contribution to the constraint
    if fixed_indices:
        if optimization_goal == "forward":
            fixed_constraint_val = sum(
                float(fixed_actual[j]) * float(np.mean(np.asarray(conversion_ratio[fixed_channels[j]], dtype=float)))
                for j in range(len(fixed_indices))
            )
        else:
            fixed_constraint_val = get_total_contribution(
                fixed_channels,
                fixed_actual,
                proportion,
                correction,
                0.0,           # constant handled separately
                s_curve_params,
            )
    else:
        fixed_constraint_val = 0.0

    # Remaining target for free channels
    if optimization_goal == "backward":
        # subtract baseline constant from target before splitting
        remaining_target = total_target - constant_sum - fixed_constraint_val
    else:
        remaining_target = total_target - fixed_constraint_val

    logger.info(
        f"Starting optimization: goal={optimization_goal}, target={total_target:,.0f}, "
        f"fixed_channels={len(fixed_indices)}, free_channels={len(free_indices)}, "
        f"fixed_constraint={fixed_constraint_val:,.0f}, remaining_target={remaining_target:,.0f}"
    )

    # ── Early-exit when all channels are frozen ────────────────────────────────
    if not free_indices:
        optimized_media = {channels[i]: float(actual_media[i]) for i in range(num_channels)}
        return optimized_media, True, "All channels frozen; no optimization performed"

    # ── xtol (computed on free channels) ──────────────────────────────────────
    positive_free = free_actual[free_actual > 0]
    if positive_free.size > 0:
        xtol = max(DEFAULT_XTOL, (xtol_tolerance_per / 100) * np.min(positive_free))
    else:
        xtol = DEFAULT_XTOL

    # ── Objective / constraint for free channels only ─────────────────────────
    def free_objective(free_vec: np.ndarray) -> float:
        try:
            if optimization_goal == "forward":
                val = -get_total_contribution(
                    free_channels, free_vec, proportion, correction, 0.0, s_curve_params
                )
            else:
                val = get_total_spends(free_vec, conversion_ratio, free_channels)
            return val if np.isfinite(val) else 1e20
        except Exception:
            return 1e20

    def free_constraint_fun(free_vec: np.ndarray) -> float:
        if optimization_goal == "forward":
            return get_total_spends(free_vec, conversion_ratio, free_channels)
        else:
            return get_total_contribution(
                free_channels, free_vec, proportion, correction, 0.0, s_curve_params
            )

    free_constraints = {
        "type": "eq",
        "fun": lambda v: free_constraint_fun(v) - remaining_target,
    }

    # Scale initial guess to satisfy remaining_target from the start
    free_initial = free_actual.copy()
    current_free = free_constraint_fun(free_initial)
    if current_free > EPS and remaining_target > 0 and optimization_goal == "forward":
        scaled = free_initial * (remaining_target / current_free)
        lo = np.array([b[0] for b in free_bounds])
        hi = np.array([b[1] for b in free_bounds])
        free_initial = np.clip(scaled, lo, hi)

    # ── Primary: SLSQP ────────────────────────────────────────────────────────
    result: OptimizeResult = minimize(
        free_objective,
        free_initial,
        method="SLSQP",
        constraints=free_constraints,
        bounds=free_bounds,
        options={"maxiter": MAX_ITER, "ftol": 1e-9, "disp": False},
    )

    # ── Fallback: trust-constr ────────────────────────────────────────────────
    if not result.success:
        logger.info("SLSQP did not converge, trying trust-constr fallback")
        result_tc: OptimizeResult = minimize(
            free_objective,
            free_initial,
            method="trust-constr",
            constraints=free_constraints,
            bounds=free_bounds,
            options={"disp": False, "xtol": xtol, "gtol": 1e-3, "maxiter": MAX_ITER},
        )
        viol_slsqp = abs(free_constraint_fun(result.x)   - remaining_target)
        viol_tc    = abs(free_constraint_fun(result_tc.x) - remaining_target)
        if viol_tc < viol_slsqp or result_tc.success:
            result = result_tc

    # ── Reassemble full channel array ─────────────────────────────────────────
    optimized_full = actual_media.copy()
    free_result = np.clip(result.x, 0.0, None)
    for j, i in enumerate(free_indices):
        optimized_full[i] = free_result[j]
    # fixed channels remain at actual_media[i]

    optimized_media = {channels[i]: float(optimized_full[i]) for i in range(num_channels)}

    # Accept result if constraint violation is within 0.5% of total_target
    full_constraint = (
        get_total_spends(optimized_full, conversion_ratio, channels)
        if optimization_goal == "forward"
        else get_total_contribution(channels, optimized_full, proportion, correction, constant_sum, s_curve_params)
    )
    constraint_violation = abs(full_constraint - total_target)
    tol_abs = max(0.005 * abs(total_target), 1.0)
    converged = result.success or (constraint_violation <= tol_abs)

    if converged:
        message = "Optimization successful"
    else:
        message = f"Optimizer did not converge: {result.message}"

    logger.info(message)
    return optimized_media, converged, message

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

        delta_abs = spend - hist_spend
        delta_pct = (delta_abs / hist_spend * 100) if hist_spend > 0 else np.nan

        rows.append({
            "Channel": ch,
            "Channel Type": "Incremental (Linear)",
            "Actual/Input Spend": hist_spend,
            "Optimized Spend": spend,
            "Δ Spend (Abs)": delta_abs,
            "Δ Spend (%)": delta_pct,
            "Actual Response Metric": hist_rev,
            "Optimized Response Metric": new_rev,
            "Actual ROI": efficiency,
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
