"""
Multi-basin OLS fitting for simulate_market parameters.

Four joint objectives (each independently finds local optima):
    mse_d_combined: MSE on a*Δu + b*Δv  (default a=b=0.5)
    mse_bc        : joint Euclidean distance in Beveridge space, mean(du²+dv²)
    mse_flow      : joint Euclidean distance of flow vector, mean(ddu²+ddv²)
    mse_path      : scale-normalised sum of u-level, v-level, Δu, and Δv MSEs;
                    each term divided by the empirical variance of its target so
                    every term contributes ~1 under a constant-mean predictor.

mse_bc and mse_flow cannot be zeroed by cancellation across dimensions.
mse_bc targets static Beveridge-curve geometry; mse_flow targets the direction,
magnitude, and persistence of movement through Beveridge space. mse_path
combines level and flow into one scale-balanced objective.

Pipeline:
    1. latin_hypercube_scan — coarse 3D exploration (parallelised)
    2. find_basins          — greedy max-separation clustering of low-loss region
    3. refine_basin         — Nelder-Mead refinement from each basin seed
    4. fit                  — runs the full pipeline, returns per-objective results

Usage
-----
    from calibration.fit import fit, PARAM_BOUNDS
    results = fit(emp_u, emp_v, time_array, gdp_signal, n_samples=500, n_basins=5)
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

try:
    from .simulate import simulate_market
except ImportError:
    from simulate import simulate_market


# ── Parameter space ───────────────────────────────────────────────────────────

PARAM_NAMES = ["K", "c_exponent", "idio_std"]

PARAM_BOUNDS = {
    "K":          (1.00, 20.00),
    "c_exponent": (0.5, 10.00),
    "idio_std":   (0.001, 0.06),
}

# Fixed parameters (not optimised)
FIXED_PARAMS = {
    "s":                0.01,
    "zero_fraction":    0.50,
    "firing_threshold": 0.1,
    "c_max":            10.0,
}

_BOUNDS_ARRAY = np.array([PARAM_BOUNDS[p] for p in PARAM_NAMES])  # (3, 2)


# ── Core evaluation ───────────────────────────────────────────────────────────

def _params_from_vector(vec):
    return dict(zip(PARAM_NAMES, vec))


def _clip_to_bounds(vec):
    return np.clip(vec, _BOUNDS_ARRAY[:, 0], _BOUNDS_ARRAY[:, 1])


ALL_OBJECTIVES = ("mse_d_combined", "mse_bc", "mse_flow", "mse_path")
_OBJ_IDX = {name: i for i, name in enumerate(ALL_OBJECTIVES)}


def _evaluate(vec, emp_u, emp_v, time_array, gdp_signal,
              a=0.5, b=0.5, sim_kwargs=None, sample_weights=None):
    """
    Run simulation for parameter vector and return four joint loss values:
        (mse_d_combined, mse_bc, mse_flow, mse_path)

    mse_d_combined : MSE on a*Δu + b*Δv (combined first-difference dynamics)
    mse_bc         : joint Euclidean distance in Beveridge space, mean(w*(du²+dv²))
    mse_flow       : joint Euclidean distance of flow vector, mean(w'*(ddu²+ddv²))
    mse_path       : scale-normalised sum of u, v, Δu, Δv MSEs; each term
                     divided by the empirical variance of its target so every
                     term contributes ~1 under a constant-mean predictor.
                     Baseline (no model) ≈ 4.

    Parameters clipped to PARAM_BOUNDS before evaluation.

    sample_weights : array-like of shape (T,), optional
        Per-period weights. COVID episode down-weighting goes here.
        Normalised so mean=1 before use. None means uniform weights.
    """
    vec = _clip_to_bounds(np.asarray(vec, dtype=float))
    p = _params_from_vector(vec)

    kw = dict(
        n_firms=250,
        seed=42,
        burn_in_steps=5,
    )
    if sim_kwargs:
        kw.update(sim_kwargs)

    try:
        u_sim, v_sim = simulate_market(
            time_array=time_array,
            gdp_signal=gdp_signal,
            K=p["K"],
            s=FIXED_PARAMS["s"],
            c_exponent=p["c_exponent"],
            c_max=FIXED_PARAMS["c_max"],
            zero_fraction=FIXED_PARAMS["zero_fraction"],
            firing_threshold=FIXED_PARAMS["firing_threshold"],
            idio_std=p["idio_std"],
            target_u=float(emp_u[0]),
            **kw,
        )
    except Exception:
        return (np.nan,) * len(ALL_OBJECTIVES)

    n = min(len(u_sim), len(emp_u), len(emp_v), len(v_sim))
    u_s, v_s = u_sim[:n], v_sim[:n]
    u_e, v_e = emp_u[:n], emp_v[:n]

    # Build normalised weight vectors
    if sample_weights is not None:
        w = np.asarray(sample_weights, dtype=float)[:n]
        w = w / (w.mean() + 1e-12)          # mean = 1 → weighted avg = unweighted avg scale
        w_diff = (w[:-1] + w[1:]) / 2.0     # weights for difference periods
    else:
        w = np.ones(n)
        w_diff = np.ones(n - 1)

    # Level residuals
    du = u_s - u_e
    dv = v_s - v_e
    ddu = np.diff(u_s) - np.diff(u_e)
    ddv = np.diff(v_s) - np.diff(v_e)

    mse_d_combined = float(np.average((a * ddu + b * ddv) ** 2, weights=w_diff))
    mse_bc         = float(np.average(du ** 2 + dv ** 2,        weights=w))
    mse_flow       = float(np.average(ddu ** 2 + ddv ** 2,      weights=w_diff))

    # mse_path: scale-normalised combined objective. Each of the four sub-terms
    # is divided by the empirical variance of its target, so each one has
    # expected value ~1 under a constant-mean predictor. Total baseline = 4.
    var_u  = float(np.var(u_e)) + 1e-12
    var_v  = float(np.var(v_e)) + 1e-12
    var_du = float(np.var(np.diff(u_e))) + 1e-12
    var_dv = float(np.var(np.diff(v_e))) + 1e-12
    mse_path = (
        float(np.average(du  ** 2, weights=w))      / var_u
      + float(np.average(dv  ** 2, weights=w))      / var_v
      + float(np.average(ddu ** 2, weights=w_diff)) / var_du
      + float(np.average(ddv ** 2, weights=w_diff)) / var_dv
    )

    return mse_d_combined, mse_bc, mse_flow, mse_path


# ── Stage 1: Latin hypercube scan ─────────────────────────────────────────────

def latin_hypercube_scan(
    emp_u, emp_v, time_array, gdp_signal,
    n_samples=500,
    a=0.5, b=0.5,
    n_jobs=8,
    sim_kwargs=None,
    seed=0,
    sample_weights=None,
):
    """
    Sample the 3D parameter space (K, c_exponent, idio_std) with a Latin
    hypercube and evaluate all eight objectives at each point.

    Returns
    -------
    pd.DataFrame
        Columns: K, c_exponent, idio_std,
                 mse_u, mse_v, mse_combined, mse_du, mse_dv, mse_d_combined,
                 mse_bc, mse_flow
    """
    n_params = len(PARAM_NAMES)
    sampler = LatinHypercube(d=n_params, seed=seed)
    unit_samples = sampler.random(n=n_samples)  # (n_samples, n_params) in [0, 1]

    # Scale to parameter bounds
    lo, hi = _BOUNDS_ARRAY[:, 0], _BOUNDS_ARRAY[:, 1]
    samples = lo + unit_samples * (hi - lo)   # (n_samples, 5)

    print(f"[fit] LHS scan: {n_samples} samples across {n_params} parameters ({n_jobs} workers)")
    t0 = time.time()

    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate)(
            samples[i], emp_u, emp_v, time_array, gdp_signal, a, b, sim_kwargs, sample_weights
        )
        for i in range(n_samples)
    )

    print(f"[fit] LHS scan complete in {time.time() - t0:.1f}s")

    df = pd.DataFrame(samples, columns=PARAM_NAMES)
    losses = pd.DataFrame(results, columns=list(ALL_OBJECTIVES))
    return pd.concat([df, losses], axis=1).dropna()


# ── Stage 2: Basin identification ─────────────────────────────────────────────

def _is_interior(norm_vec, margin=0.05):
    """True if all parameters are at least `margin` (normalised) from every boundary."""
    return bool(np.all(norm_vec > margin) and np.all(norm_vec < 1.0 - margin))


def find_basins(scan_df, objective="mse_u", n_basins=5, top_fraction=0.20,
                min_separation=0.15, boundary_margin=0.05):
    """
    Identify n_basins distinct low-loss parameter neighbourhoods from scan results.

    Strategy:
        1. Sort by objective (ascending); take the top `top_fraction` as candidates.
        2. Seed with the globally best candidate.
        3. Greedily add the candidate with the largest minimum normalised distance
           to already-selected seeds, subject to min_separation.
        4. If no selected basin is an interior point (all parameters > boundary_margin
           from their bounds), force-add the best-loss interior candidate so that
           the refinement stage always explores at least one non-boundary region.

    Parameters
    ----------
    scan_df : pd.DataFrame
        Output of latin_hypercube_scan.
    objective : str
        One of 'mse_u', 'mse_v', 'mse_combined'.
    n_basins : int
        Number of distinct basins to return.
    top_fraction : float
        Fraction of scan_df to consider as candidates (lowest-loss).
    min_separation : float
        Minimum normalised distance between basin seeds.
    boundary_margin : float
        Normalised distance from any boundary below which a point is called
        a "boundary point" (default 0.05 = 5% of the parameter range).

    Returns
    -------
    list of dict
        Each entry: {'params': dict, 'loss': float, 'basin_id': int}
    """
    if objective not in scan_df.columns:
        raise ValueError(f"Objective column '{objective}' not found in scan_df")

    # Take the top fraction by loss
    sorted_df = scan_df.sort_values(objective).head(max(1, int(len(scan_df) * top_fraction)))
    param_matrix = sorted_df[PARAM_NAMES].values

    # Normalise parameter dimensions to [0, 1] for distance computation
    lo, hi = _BOUNDS_ARRAY[:, 0], _BOUNDS_ARRAY[:, 1]
    norm_matrix = (param_matrix - lo) / (hi - lo + 1e-12)

    selected_indices = [0]  # best point is always a seed
    for _ in range(n_basins - 1):
        if len(selected_indices) >= len(sorted_df):
            break
        selected_norm = norm_matrix[selected_indices]
        remaining = [i for i in range(len(sorted_df)) if i not in selected_indices]
        if not remaining:
            break
        min_dists = np.array([
            np.min(np.linalg.norm(norm_matrix[i] - selected_norm, axis=1))
            for i in remaining
        ])
        best_candidate = remaining[np.argmax(min_dists)]
        if min_dists[np.argmax(min_dists)] < min_separation:
            break
        selected_indices.append(best_candidate)

    # Ensure at least one interior basin is represented.
    # "Interior" = every parameter > boundary_margin from its bound (normalised).
    has_interior = any(_is_interior(norm_matrix[i], boundary_margin) for i in selected_indices)
    if not has_interior:
        # Search the full sorted candidates (not just the greedy selection) for the
        # best-loss point that is interior, and inject it regardless of separation.
        for i in range(len(sorted_df)):
            if i not in selected_indices and _is_interior(norm_matrix[i], boundary_margin):
                selected_indices.append(i)
                break  # one interior basin is enough

    basins = []
    for basin_id, idx in enumerate(selected_indices):
        row = sorted_df.iloc[idx]
        basins.append({
            "basin_id": basin_id,
            "params": {p: float(row[p]) for p in PARAM_NAMES},
            "loss": float(row[objective]),
            "interior": _is_interior(norm_matrix[idx], boundary_margin),
        })

    return basins


# ── Stage 3: Local refinement ─────────────────────────────────────────────────

def refine_basin(basin, emp_u, emp_v, time_array, gdp_signal,
                 objective="mse_u", a=0.5, b=0.5,
                 sim_kwargs=None, max_iter=500, sample_weights=None):
    """
    Refine a basin seed with Nelder-Mead minimisation.

    Parameters
    ----------
    basin : dict
        As returned by find_basins — must have 'params' and 'basin_id'.
    objective : str
        One of 'mse_u', 'mse_v', 'mse_combined'.
    a, b : float
        Weights for mse_combined.

    Returns
    -------
    dict
        Updated basin dict with refined 'params', 'loss', and 'converged'.
    """
    obj_idx = _OBJ_IDX[objective]

    x0 = np.array([basin["params"][p] for p in PARAM_NAMES])

    def objective_fn(vec):
        losses = _evaluate(vec, emp_u, emp_v, time_array, gdp_signal, a, b, sim_kwargs, sample_weights)
        val = losses[obj_idx]
        return val if np.isfinite(val) else 1e6

    bounds = [PARAM_BOUNDS[p] for p in PARAM_NAMES]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            objective_fn,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": max_iter, "xatol": 1e-5, "fatol": 1e-8},
        )

    refined_vec = _clip_to_bounds(result.x)
    all_losses = _evaluate(refined_vec, emp_u, emp_v, time_array, gdp_signal, a, b, sim_kwargs, sample_weights)

    refined_params = _params_from_vector(refined_vec)
    lo, hi = _BOUNDS_ARRAY[:, 0], _BOUNDS_ARRAY[:, 1]
    norm_refined = (refined_vec - lo) / (hi - lo + 1e-12)
    at_boundary = not _is_interior(norm_refined, margin=0.05)

    loss_dict = {name: float(all_losses[i]) for i, name in enumerate(ALL_OBJECTIVES)}
    return {
        "basin_id":    basin["basin_id"],
        "params":      refined_params,
        "loss":        float(result.fun),
        **loss_dict,
        "converged":   result.success,
        "n_iter":      result.nit,
        "seed_interior": basin.get("interior", None),
        "at_boundary": at_boundary,
    }


# ── Stage 4: Full pipeline ────────────────────────────────────────────────────

def fit(
    emp_u, emp_v, time_array, gdp_signal,
    n_samples=500,
    n_basins=5,
    top_fraction=0.20,
    min_separation=0.15,
    a=0.5, b=0.5,
    n_jobs=8,
    sim_kwargs=None,
    output_dir="output/fit",
    sample_weights=None,
):
    """
    Full multi-basin OLS fitting pipeline.

    Runs the LHS scan once, then independently finds and refines basins for
    each of the three objectives.

    Parameters
    ----------
    emp_u, emp_v : np.ndarray
        Empirical unemployment and vacancy rates.
    time_array : np.ndarray
        Time index aligned to emp_u / emp_v.
    gdp_signal : np.ndarray
        GDP signal aligned to time_array.
    n_samples : int
        Number of LHS samples for Stage 1.
    n_basins : int
        Number of distinct basins to find and refine per objective.
    top_fraction : float
        Fraction of scan used as basin candidates.
    min_separation : float
        Minimum normalised parameter-space distance between basin seeds.
    a, b : float
        Weights for mse_combined objective.
    n_jobs : int
        Parallel workers for the LHS scan.
    sim_kwargs : dict, optional
        Extra kwargs passed to simulate_market (e.g. n_firms, zero_fraction).
    output_dir : str
        Where to save CSV summaries.

    Returns
    -------
    dict with keys:
        'scan'          : pd.DataFrame — full LHS scan results
        'mse_u'         : list of refined basin dicts
        'mse_v'         : list of refined basin dicts
        'mse_combined'  : list of refined basin dicts
        'mse_du'        : list of refined basin dicts
        'mse_dv'        : list of refined basin dicts
        'mse_d_combined': list of refined basin dicts
        'mse_bc'        : list of refined basin dicts
        'mse_flow'      : list of refined basin dicts
    """
    os.makedirs(output_dir, exist_ok=True)

    # Variance baselines for R² computation: a value < 0 means the model fits
    # worse than a constant predictor of the empirical mean.
    du_emp = np.diff(emp_u)
    dv_emp = np.diff(emp_v)
    var_baseline = {
        "mse_d_combined": np.var(a * du_emp + b * dv_emp),
        "mse_bc":         np.var(emp_u) + np.var(emp_v),
        "mse_flow":       np.var(du_emp) + np.var(dv_emp),
        # mse_path sums four normalised terms, each ~1 at the constant-mean
        # baseline, so the no-model loss is ~4.
        "mse_path":       4.0,
    }

    # Stage 1: scan
    scan_df = latin_hypercube_scan(
        emp_u, emp_v, time_array, gdp_signal,
        n_samples=n_samples, a=a, b=b, n_jobs=n_jobs,
        sim_kwargs=sim_kwargs, sample_weights=sample_weights,
    )
    scan_df.to_csv(os.path.join(output_dir, "lhs_scan.csv"), index=False)
    print(f"[fit] Scan saved. Valid samples: {len(scan_df)}/{n_samples}")

    all_results = {"scan": scan_df}

    for objective in ALL_OBJECTIVES:
        print(f"\n[fit] --- Objective: {objective} ---")

        # Stage 2: basins
        basins = find_basins(
            scan_df, objective=objective,
            n_basins=n_basins,
            top_fraction=top_fraction,
            min_separation=min_separation,
        )
        print(f"[fit]   {len(basins)} basins identified")
        for b_ in basins:
            loc = "interior" if b_.get("interior") else "boundary"
            print(f"        basin {b_['basin_id']} [{loc}]: seed loss={b_['loss']:.6f}  params={b_['params']}")

        # Stage 3: refine
        refined = []
        baseline = var_baseline[objective]
        for basin in basins:
            t0 = time.time()
            r = refine_basin(
                basin, emp_u, emp_v, time_array, gdp_signal,
                objective=objective, a=a, b=b,
                sim_kwargs=sim_kwargs, sample_weights=sample_weights,
            )
            loc = "boundary" if r.get("at_boundary") else "interior"
            r2 = 1.0 - r["loss"] / baseline if baseline > 0 else float("nan")
            r["r2"] = r2
            print(f"        basin {r['basin_id']} refined [{loc}]: loss={r['loss']:.6f}  "
                  f"R²={r2:+.3f}  converged={r['converged']}  iters={r['n_iter']}  "
                  f"({time.time()-t0:.1f}s)")
            refined.append(r)

        # Save
        rows = []
        for r in refined:
            row = {"basin_id": r["basin_id"], "objective": objective,
                   "loss": r["loss"], "r2": r["r2"],
                   **{name: r[name] for name in ALL_OBJECTIVES},
                   "converged": r["converged"], "n_iter": r["n_iter"],
                   "seed_interior": r.get("seed_interior"),
                   "at_boundary": r.get("at_boundary")}
            row.update(r["params"])
            rows.append(row)

        results_df = pd.DataFrame(rows).sort_values("loss")
        results_df.to_csv(os.path.join(output_dir, f"basins_{objective}.csv"), index=False)
        print(f"[fit]   Saved to {output_dir}/basins_{objective}.csv")

        all_results[objective] = refined

    return all_results


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from calibration.empirical import load_empirical_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_start", type=str, default="2000-01-01",
                        help="Start of fitting window (ISO date)")
    parser.add_argument("--fit_end",   type=str, default="2025-10-31",
                        help="End of fitting window (ISO date)")
    parser.add_argument("--output_dir", type=str, default="output/fit_weighted")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--n_basins",  type=int, default=5)
    args = parser.parse_args()

    df = load_empirical_data(start_date=args.fit_start, end_date=args.fit_end)
    df = df.dropna(subset=["u_obs", "v_obs", "G"])
    df = df.loc[args.fit_start:args.fit_end]
    print(f"[fit] Fitting window: {df.index[0].date()} – {df.index[-1].date()} "
          f"({len(df)} months)")

    emp_u      = df["u_obs"].values.astype(float)
    emp_v      = df["v_obs"].values.astype(float)
    gdp_signal = df["G"].values.astype(float)
    time_array = np.arange(len(emp_u))

    # Down-weight the acute COVID shock (Apr–Aug 2020) to 0.1 if in window.
    # No-op if the window excludes COVID.
    w = np.ones(len(emp_u))
    covid = (df.index >= "2020-03-01") & (df.index <= "2020-08-01")
    w[covid] = 0.1
    if covid.sum():
        print(f"[fit] COVID down-weight: {covid.sum()} months set to 0.1")

    results = fit(emp_u, emp_v, time_array, gdp_signal,
                  n_samples=args.n_samples, n_basins=args.n_basins,
                  sample_weights=w,
                  output_dir=args.output_dir)
