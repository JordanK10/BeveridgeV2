"""
Slope-matching calibration for simulate_market parameters.

Fits the standardised first differences of u and v:

    loss = MSE( Δu_sim / std(Δu_sim) - Δu_emp / std(Δu_emp) )

This forces the model to match the *direction* and *relative magnitude* of
unemployment and vacancy movements — i.e. the slope of the time series —
without being dominated by level offsets or a single large outlier.

Three objectives (each independently finds local optima):
    slope_u        : standardised-diff MSE on unemployment
    slope_v        : standardised-diff MSE on vacancies
    slope_combined : equal-weight average of the two

Pipeline:
    1. latin_hypercube_scan — coarse 6D exploration (parallelised)
    2. find_basins          — greedy max-separation clustering of low-loss region
    3. refine_basin         — Nelder-Mead refinement from each basin seed

Usage
-----
    python src/calibration/fit_slope.py
    python src/calibration/fit_slope.py output/fit_slope
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from calibration.simulate import simulate_market
    from calibration.empirical import load_empirical_data
    from calibration.fit import (
        PARAM_NAMES, PARAM_BOUNDS, _BOUNDS_ARRAY,
        _clip_to_bounds, _params_from_vector, _is_interior, find_basins,
    )
except ImportError:
    from simulate import simulate_market
    from empirical import load_empirical_data
    from fit import (
        PARAM_NAMES, PARAM_BOUNDS, _BOUNDS_ARRAY,
        _clip_to_bounds, _params_from_vector, _is_interior, find_basins,
    )


ALL_OBJECTIVES = ("slope_u", "slope_v", "slope_combined")
_OBJ_IDX = {name: i for i, name in enumerate(ALL_OBJECTIVES)}


# ── Core evaluation ───────────────────────────────────────────────────────────

def _evaluate(vec, emp_u, emp_v, time_array, gdp_signal,
              sim_kwargs=None, sample_weights=None):
    """
    Run simulation and return slope-matching losses:
        (slope_u, slope_v, slope_combined)

    Each loss is MSE( Δx_sim/std(Δx_sim) - Δx_emp/std(Δx_emp) ).
    Standardising by each series' own std means the loss measures whether the
    simulated series moves in the same direction and with the same relative
    magnitude as empirical — the regression slope — rather than raw scale.

    sample_weights : array-like of shape (T,), optional
        Per-period weights. Applied to the difference periods as the average
        of adjacent period weights, then normalised to mean=1.
    """
    vec = _clip_to_bounds(np.asarray(vec, dtype=float))
    p = _params_from_vector(vec)

    kw = dict(
        n_firms=250,
        seed=42,
        burn_in_steps=1000,
    )
    if sim_kwargs:
        kw.update(sim_kwargs)

    try:
        u_sim, v_sim = simulate_market(
            time_array=time_array,
            gdp_signal=gdp_signal,
            K=p["K"],
            s=p["s"],
            c_exponent=p["c_exponent"],
            c_max=1.0,
            zero_fraction=p["zero_fraction"],
            firing_threshold=p["firing_threshold"],
            idio_std=p["idio_std"],
            **kw,
        )
    except Exception:
        return (np.nan,) * len(ALL_OBJECTIVES)

    n = min(len(u_sim), len(emp_u), len(emp_v), len(v_sim))
    du_sim = np.diff(u_sim[:n])
    dv_sim = np.diff(v_sim[:n])
    du_emp = np.diff(emp_u[:n])
    dv_emp = np.diff(emp_v[:n])

    # Standardise each series by its own std (clamped to avoid div-by-zero)
    std_du_sim = max(float(np.std(du_sim)), 1e-10)
    std_dv_sim = max(float(np.std(dv_sim)), 1e-10)
    std_du_emp = max(float(np.std(du_emp)), 1e-10)
    std_dv_emp = max(float(np.std(dv_emp)), 1e-10)

    du_sim_std = du_sim / std_du_sim
    dv_sim_std = dv_sim / std_dv_sim
    du_emp_std = du_emp / std_du_emp
    dv_emp_std = dv_emp / std_dv_emp

    # Build difference-period weights
    if sample_weights is not None:
        w = np.asarray(sample_weights, dtype=float)[:n]
        w = w / (w.mean() + 1e-12)
        w_diff = (w[:-1] + w[1:]) / 2.0
    else:
        w_diff = np.ones(n - 1)

    res_u = du_sim_std - du_emp_std
    res_v = dv_sim_std - dv_emp_std

    slope_u        = float(np.average(res_u ** 2, weights=w_diff))
    slope_v        = float(np.average(res_v ** 2, weights=w_diff))
    slope_combined = float(np.average((res_u + res_v) ** 2 / 4.0, weights=w_diff))

    return slope_u, slope_v, slope_combined


# ── Stage 1: Latin hypercube scan ─────────────────────────────────────────────

def latin_hypercube_scan(
    emp_u, emp_v, time_array, gdp_signal,
    n_samples=500,
    n_jobs=8,
    sim_kwargs=None,
    seed=0,
    sample_weights=None,
):
    n_params = len(PARAM_NAMES)
    sampler = LatinHypercube(d=n_params, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    lo, hi = _BOUNDS_ARRAY[:, 0], _BOUNDS_ARRAY[:, 1]
    samples = lo + unit_samples * (hi - lo)

    print(f"[fit_slope] LHS scan: {n_samples} samples across {n_params} parameters ({n_jobs} workers)")
    t0 = time.time()

    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate)(
            samples[i], emp_u, emp_v, time_array, gdp_signal, sim_kwargs, sample_weights
        )
        for i in range(n_samples)
    )

    print(f"[fit_slope] LHS scan complete in {time.time() - t0:.1f}s")

    df = pd.DataFrame(samples, columns=PARAM_NAMES)
    losses = pd.DataFrame(results, columns=list(ALL_OBJECTIVES))
    return pd.concat([df, losses], axis=1).dropna()


# ── Stage 3: Local refinement ─────────────────────────────────────────────────

def refine_basin(basin, emp_u, emp_v, time_array, gdp_signal,
                 objective="slope_combined",
                 sim_kwargs=None, max_iter=500, sample_weights=None):
    obj_idx = _OBJ_IDX[objective]
    x0 = np.array([basin["params"][p] for p in PARAM_NAMES])
    bounds = [PARAM_BOUNDS[p] for p in PARAM_NAMES]

    def objective_fn(vec):
        losses = _evaluate(vec, emp_u, emp_v, time_array, gdp_signal, sim_kwargs, sample_weights)
        val = losses[obj_idx]
        return val if np.isfinite(val) else 1e6

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            objective_fn, x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": max_iter, "xatol": 1e-5, "fatol": 1e-8},
        )

    refined_vec = _clip_to_bounds(result.x)
    all_losses = _evaluate(refined_vec, emp_u, emp_v, time_array, gdp_signal, sim_kwargs, sample_weights)

    refined_params = _params_from_vector(refined_vec)
    lo, hi = _BOUNDS_ARRAY[:, 0], _BOUNDS_ARRAY[:, 1]
    norm_refined = (refined_vec - lo) / (hi - lo + 1e-12)
    at_boundary = not _is_interior(norm_refined, margin=0.05)

    loss_dict = {name: float(all_losses[i]) for i, name in enumerate(ALL_OBJECTIVES)}
    return {
        "basin_id":      basin["basin_id"],
        "params":        refined_params,
        "loss":          float(result.fun),
        **loss_dict,
        "converged":     result.success,
        "n_iter":        result.nit,
        "seed_interior": basin.get("interior", None),
        "at_boundary":   at_boundary,
    }


# ── Full pipeline ─────────────────────────────────────────────────────────────

def fit_slope(
    emp_u, emp_v, time_array, gdp_signal,
    n_samples=500,
    n_basins=5,
    top_fraction=0.20,
    min_separation=0.15,
    n_jobs=8,
    sim_kwargs=None,
    output_dir="output/fit_slope",
    sample_weights=None,
):
    os.makedirs(output_dir, exist_ok=True)

    scan_df = latin_hypercube_scan(
        emp_u, emp_v, time_array, gdp_signal,
        n_samples=n_samples, n_jobs=n_jobs,
        sim_kwargs=sim_kwargs, sample_weights=sample_weights,
    )
    scan_df.to_csv(os.path.join(output_dir, "lhs_scan.csv"), index=False)
    print(f"[fit_slope] Scan saved. Valid samples: {len(scan_df)}/{n_samples}")

    all_results = {"scan": scan_df}

    for objective in ALL_OBJECTIVES:
        print(f"\n[fit_slope] --- Objective: {objective} ---")

        basins = find_basins(
            scan_df, objective=objective,
            n_basins=n_basins,
            top_fraction=top_fraction,
            min_separation=min_separation,
        )
        print(f"[fit_slope]   {len(basins)} basins identified")
        for b_ in basins:
            loc = "interior" if b_.get("interior") else "boundary"
            print(f"        basin {b_['basin_id']} [{loc}]: seed loss={b_['loss']:.6f}  params={b_['params']}")

        refined = []
        for basin in basins:
            t0 = time.time()
            r = refine_basin(
                basin, emp_u, emp_v, time_array, gdp_signal,
                objective=objective,
                sim_kwargs=sim_kwargs, sample_weights=sample_weights,
            )
            loc = "boundary" if r.get("at_boundary") else "interior"
            print(f"        basin {r['basin_id']} refined [{loc}]: loss={r['loss']:.6f}  "
                  f"converged={r['converged']}  iters={r['n_iter']}  ({time.time()-t0:.1f}s)")
            refined.append(r)

        rows = []
        for r in refined:
            row = {"basin_id": r["basin_id"], "objective": objective,
                   "loss": r["loss"],
                   **{name: r[name] for name in ALL_OBJECTIVES},
                   "converged": r["converged"], "n_iter": r["n_iter"],
                   "seed_interior": r.get("seed_interior"),
                   "at_boundary": r.get("at_boundary")}
            row.update(r["params"])
            rows.append(row)

        results_df = pd.DataFrame(rows).sort_values("loss")
        results_df.to_csv(os.path.join(output_dir, f"basins_{objective}.csv"), index=False)
        print(f"[fit_slope]   Saved to {output_dir}/basins_{objective}.csv")

        all_results[objective] = refined

    return all_results


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output/fit_slope"

    df = load_empirical_data(end_date="2019-12-31")
    df = df.dropna(subset=["u_obs", "v_obs", "G"])

    emp_u      = df["u_obs"].values.astype(float)
    emp_v      = df["v_obs"].values.astype(float)
    gdp_signal = df["G"].values.astype(float)
    time_array = np.arange(len(emp_u))

    w = np.ones(len(emp_u))
    covid = (df.index >= "2020-03-01") & (df.index <= "2020-08-01")
    w[covid] = 0.1
    print(f"[fit_slope] COVID down-weight: {covid.sum()} months set to 0.1")

    fit_slope(
        emp_u, emp_v, time_array, gdp_signal,
        n_samples=500,
        n_basins=5,
        sample_weights=w,
        output_dir=output_dir,
    )
