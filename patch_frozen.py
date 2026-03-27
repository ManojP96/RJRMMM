"""
Patch: fix frozen-channel optimization in optimizer_backend.py

When channels are frozen (lb=ub), SLSQP sees a degenerate problem and returns
the initial guess unchanged. Fix: extract fixed channels, optimize only free
channels against the remaining budget/response target, then reassemble.
"""
import re

with open("optimizer_backend.py", "r", encoding="utf-8") as f:
    src = f.read()

# ── The block we want to REPLACE starts after the bounds list and ends at the
#    return statement (inclusive).  We'll replace lines 203-272.
OLD = '''    # Bounds (same formula as old code; no clipping to 0 here)
    bounds = [
        (
            actual_media[i] * (1 + bounds_dict[channels[i]][0] / 100),
            actual_media[i] * (1 + bounds_dict[channels[i]][1] / 100),
        )
        for i in range(num_channels)
    ]

    # Scale initial guess to satisfy the budget constraint from the start
    initial_guess = np.array(actual_media, dtype=float)
    current_spend = get_total_spends(initial_guess, conversion_ratio, channels)
    if current_spend > EPS and optimization_goal == "forward":
        scaled = initial_guess * (total_target / current_spend)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        initial_guess = np.clip(scaled, lo, hi)

    positive_media = actual_media[actual_media > 0]
    if positive_media.size > 0:
        xtol = max(DEFAULT_XTOL, (xtol_tolerance_per / 100) * np.min(positive_media))
    else:
        xtol = DEFAULT_XTOL

    logger.info(
        f"Starting optimization: goal={optimization_goal}, target={total_target:,.0f}"
    )

    # Primary: SLSQP (handles equality constraints reliably)
    result: OptimizeResult = minimize(
        objective_fun,
        initial_guess,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": MAX_ITER, "ftol": 1e-9, "disp": False},
    )

    # Fallback: trust-constr with relaxed tolerances
    if not result.success:
        logger.info("SLSQP did not converge, trying trust-constr fallback")
        result_tc: OptimizeResult = minimize(
            objective_fun,
            initial_guess,
            method="trust-constr",
            constraints=constraints,
            bounds=bounds,
            options={"disp": False, "xtol": xtol, "gtol": 1e-3, "maxiter": MAX_ITER},
        )
        # Accept whichever result has smaller constraint violation
        viol_slsqp = abs(constraint_fun(result.x) - total_target)
        viol_tc    = abs(constraint_fun(result_tc.x) - total_target)
        if viol_tc < viol_slsqp or result_tc.success:
            result = result_tc

    optimized_media_array = np.clip(result.x, 0.0, None)
    optimized_media = {channels[i]: optimized_media_array[i] for i in range(num_channels)}

    # Accept result if constraint violation is within 0.5% of target
    constraint_violation = abs(constraint_fun(optimized_media_array) - total_target)
    tol_abs = max(0.005 * abs(total_target), 1.0)
    converged = result.success or (constraint_violation <= tol_abs)

    if converged:
        message = "Optimization successful"
    else:
        message = f"Optimizer did not converge: {result.message}"

    logger.info(message)
    return optimized_media, converged, message'''

NEW = '''    # Bounds (absolute values)
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

    # Compute fixed channels\' contribution to the constraint
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
    return optimized_media, converged, message'''

if OLD in src:
    src = src.replace(OLD, NEW, 1)
    with open("optimizer_backend.py", "w", encoding="utf-8") as f:
        f.write(src)
    print("PATCHED OK")
else:
    # Debug: show first 80 chars of each line around line 203
    lines = src.splitlines()
    for i, line in enumerate(lines[198:215], start=199):
        print(f"{i}: {repr(line[:80])}")
    print("NOT FOUND - check diff above")
