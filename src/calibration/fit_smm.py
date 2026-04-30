"""
Simulated Method of Moments (SMM) fitting for simulate_market parameters.

Instead of minimising point-to-point MSE against the empirical time series,
SMM matches a vector of statistics (moments) computed from the simulated output
to the corresponding empirical moments.  Because the loss is based on summary
statistics rather than a single noise realisation, it is:
  - Robust to the stochastic nature of the simulation (moments are averaged
    across n_seeds independent runs to reduce noise).
  - Not dominated by outlier events such as COVID (using percentile-based
    moments rather than the raw maximum).
  - Better identified: each moment constrains a different subset of parameters.

Moment vector targeted:
    mean_u       — steady-state unemployment level (identifies K, s)
    std_u        — cyclical amplitude of unemployment (identifies c_exp, zero_fraction)
    mean_v       — steady-state vacancy rate
    std_v        — cyclical amplitude of vacancies
    corr_uv      — Beveridge correlation (identifies model coherence)
    autocorr_du  — lag-1 autocorrelation of Δu (identifies recovery speed)
    autocorr_dv  — lag-1 autocorrelation of Δv
    skew_du      — skewness of Δu (identifies firing_threshold asymmetry)
    skew_dv      — skewness of Δv
    q90_u        — 90th-percentile unemployment (captures recession severity
                   without COVID domination)
    q10_v        — 10th-percentile vacancy rate (recession-time trough)

Loss function:
    L = Σ_k w_k * ((m_k_sim - m_k_emp) / scale_k)^2

where scale_k = |m_k_emp| for level moments and 1.0 for bounded moments
(correlations, skewness), so all terms are dimensionless.

Usage
-----
    python src/calibration/fit_smm.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats import skew as scipy_skew
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


# ── Moment definitions ────────────────────────────────────────────────────────

MOMENT_NAMES = [
    "mean_u", "std_u", "mean_v", "std_v",
    "corr_uv",
    "autocorr_du", "autocorr_dv",
    "skew_du", "skew_dv",
    "q90_u", "q10_v",
]

# Relative weights: moments in natural units (mean, std, percentile) get
# weight 1; dimensionless moments (correlations, skewness) get weight 0.5
# so they don't overwhelm the level moments.
MOMENT_WEIGHTS = {
    "mean_u":      2.0,   # most important for identification
    "std_u":       2.0,
    "mean_v":      2.0,
    "std_v":       2.0,
    "corr_uv":     1.0,
    "autocorr_du": 1.0,
    "autocorr_dv": 1.0,
    "skew_du":     0.5,
    "skew_dv":     0.5,
    "q90_u":       1.0,
    "q10_v":       1.0,
}


def compute_moments(u, v):
    """Compute the SMM moment vector from unemployment and vacancy arrays."""
    du = np.diff(u)
    dv = np.diff(v)

    if len(du) > 2:
        ac_du = float(np.corrcoef(du[:-1], du[1:])[0, 1])
        ac_dv = float(np.corrcoef(dv[:-1], dv[1:])[0, 1])
    else:
        ac_du = ac_dv = 0.0

    return {
        "mean_u":      float(np.mean(u)),
        "std_u":       float(np.std(u)),
        "mean_v":      float(np.mean(v)),
        "std_v":       float(np.std(v)),
        "corr_uv":     float(np.corrcoef(u, v)[0, 1]),
        "autocorr_du": ac_du,
        "autocorr_dv": ac_dv,
        "skew_du":     float(scipy_skew(du)) if len(du) > 2 else 0.0,
        "skew_dv":     float(scipy_skew(dv)) if len(dv) > 2 else 0.0,
        "q90_u":       float(np.percentile(u, 90)),
        "q10_v":       float(np.percentile(v, 10)),
    }


def _moment_scales(emp_moments):
    """
    Normalising denominators for the loss: |empirical value| for level moments,
    1.0 for dimensionless ones.  Clamped to avoid division near zero.
    """
    dimensionless = {"corr_uv", "autocorr_du", "autocorr_dv", "skew_du", "skew_dv"}
    return {
        k: 1.0 if k in dimensionless else max(abs(emp_moments[k]), 1e-6)
        for k in MOMENT_NAMES
    }


# ── Core SMM evaluation ───────────────────────────────────────────────────────

def _evaluate_smm(vec, emp_moments, scales, time_array, gdp_signal,
                  n_seeds=3, sim_kwargs=None):
    """
    Run n_seeds simulations, average their moments, return the weighted
    squared distance to the empirical moments.
    """
    vec = _clip_to_bounds(np.asarray(vec, dtype=float))
    p = _params_from_vector(vec)

    kw = dict(
        n_firms=250,
        burn_in_steps=500,
    )
    if sim_kwargs:
        kw.update(sim_kwargs)

    sim_moment_lists = {k: [] for k in MOMENT_NAMES}

    for seed in range(42, 42 + n_seeds):
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
                seed=seed,
                **kw,
            )
        except Exception:
            return np.nan

        n = min(len(u_sim), len(gdp_signal))
        moms = compute_moments(u_sim[:n], v_sim[:n])
        for k in MOMENT_NAMES:
            sim_moment_lists[k].append(moms[k])

    if not sim_moment_lists["mean_u"]:
        return np.nan

    # Average moments across seeds
    sim_moments = {k: float(np.mean(vals)) for k, vals in sim_moment_lists.items()}

    # Weighted sum of squared relative deviations
    loss = sum(
        MOMENT_WEIGHTS[k] * ((sim_moments[k] - emp_moments[k]) / scales[k]) ** 2
        for k in MOMENT_NAMES
    )
    return float(loss)


# ── Stage 1: LHS scan ─────────────────────────────────────────────────────────

def smm_lhs_scan(emp_moments, scales, time_array, gdp_signal,
                 n_samples=500, n_seeds=3, n_jobs=8, sim_kwargs=None, seed=0):
    """
    Latin-hypercube scan of the parameter space, evaluating the SMM loss.

    Returns pd.DataFrame with PARAM_NAMES + 'smm_loss'.
    """
    n_params = len(PARAM_NAMES)
    sampler = LatinHypercube(d=n_params, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    lo, hi = _BOUNDS_ARRAY[:, 0], _BOUNDS_ARRAY[:, 1]
    samples = lo + unit_samples * (hi - lo)

    print(f"[smm] LHS scan: {n_samples} samples × {n_params} params × {n_seeds} seeds ({n_jobs} workers)")
    t0 = time.time()

    losses = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_smm)(
            samples[i], emp_moments, scales, time_array, gdp_signal, n_seeds, sim_kwargs
        )
        for i in range(n_samples)
    )

    print(f"[smm] LHS scan complete in {time.time() - t0:.1f}s")

    df = pd.DataFrame(samples, columns=PARAM_NAMES)
    df["smm_loss"] = losses
    return df.dropna()


# ── Stage 3: Local refinement ─────────────────────────────────────────────────

def smm_refine_basin(basin, emp_moments, scales, time_array, gdp_signal,
                     n_seeds=3, sim_kwargs=None, max_iter=500):
    """Nelder-Mead refinement of a single basin for the SMM objective."""
    x0 = np.array([basin["params"][p] for p in PARAM_NAMES])
    bounds = [PARAM_BOUNDS[p] for p in PARAM_NAMES]

    def obj_fn(vec):
        val = _evaluate_smm(vec, emp_moments, scales, time_array, gdp_signal, n_seeds, sim_kwargs)
        return val if (val is not None and np.isfinite(val)) else 1e6

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            obj_fn, x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": max_iter, "xatol": 1e-5, "fatol": 1e-8},
        )

    refined_vec = _clip_to_bounds(result.x)
    lo, hi = _BOUNDS_ARRAY[:, 0], _BOUNDS_ARRAY[:, 1]
    norm_refined = (refined_vec - lo) / (hi - lo + 1e-12)
    at_boundary = not _is_interior(norm_refined, margin=0.05)

    # Final evaluation: more seeds for a cleaner estimate
    final_loss = _evaluate_smm(refined_vec, emp_moments, scales,
                                time_array, gdp_signal,
                                n_seeds=max(n_seeds, 5), sim_kwargs=sim_kwargs)

    return {
        "basin_id":      basin["basin_id"],
        "params":        _params_from_vector(refined_vec),
        "smm_loss":      float(result.fun),
        "smm_loss_eval": float(final_loss) if np.isfinite(final_loss) else np.nan,
        "converged":     result.success,
        "n_iter":        result.nit,
        "seed_interior": basin.get("interior", None),
        "at_boundary":   at_boundary,
    }


# ── Full pipeline ─────────────────────────────────────────────────────────────

def fit_smm(emp_u, emp_v, time_array, gdp_signal,
            n_samples=500, n_basins=5, n_seeds=3,
            top_fraction=0.20, min_separation=0.15,
            n_jobs=8, sim_kwargs=None,
            output_dir="output/fit_smm"):
    """
    Full SMM fitting pipeline.

    Parameters
    ----------
    emp_u, emp_v : np.ndarray
    time_array   : np.ndarray
    gdp_signal   : np.ndarray
    n_samples    : int   — LHS scan points
    n_basins     : int   — basins to refine
    n_seeds      : int   — simulation seeds averaged per evaluation
    output_dir   : str

    Returns
    -------
    dict:
        'scan'    : pd.DataFrame — LHS results
        'basins'  : list of refined basin dicts
        'emp_moments': dict
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute empirical moments
    emp_moments = compute_moments(emp_u, emp_v)
    scales = _moment_scales(emp_moments)

    print("[smm] Empirical moments:")
    for k in MOMENT_NAMES:
        print(f"      {k:15s} = {emp_moments[k]:+.5f}  (scale={scales[k]:.5f})")

    # Stage 1: scan
    scan_df = smm_lhs_scan(emp_moments, scales, time_array, gdp_signal,
                            n_samples=n_samples, n_seeds=n_seeds,
                            n_jobs=n_jobs, sim_kwargs=sim_kwargs)
    scan_df.to_csv(os.path.join(output_dir, "smm_lhs_scan.csv"), index=False)
    print(f"[smm] Scan saved. Valid: {len(scan_df)}/{n_samples}  "
          f"best loss={scan_df['smm_loss'].min():.4f}")

    # Stage 2: basin identification
    basins = find_basins(scan_df, objective="smm_loss",
                         n_basins=n_basins,
                         top_fraction=top_fraction,
                         min_separation=min_separation)
    print(f"[smm] {len(basins)} basins identified")
    for b in basins:
        loc = "interior" if b.get("interior") else "boundary"
        print(f"      basin {b['basin_id']} [{loc}]: seed loss={b['loss']:.4f}  params={b['params']}")

    # Stage 3: refinement
    refined = []
    for basin in basins:
        t0 = time.time()
        r = smm_refine_basin(basin, emp_moments, scales, time_array, gdp_signal,
                             n_seeds=n_seeds, sim_kwargs=sim_kwargs)
        loc = "boundary" if r["at_boundary"] else "interior"
        print(f"      basin {r['basin_id']} refined [{loc}]: "
              f"loss={r['smm_loss']:.4f}  converged={r['converged']}  "
              f"iters={r['n_iter']}  ({time.time()-t0:.1f}s)")
        refined.append(r)

    # Save
    rows = []
    for r in refined:
        row = {
            "basin_id": r["basin_id"],
            "smm_loss": r["smm_loss"],
            "smm_loss_eval": r["smm_loss_eval"],
            "converged": r["converged"],
            "n_iter": r["n_iter"],
            "seed_interior": r["seed_interior"],
            "at_boundary": r["at_boundary"],
        }
        row.update(r["params"])
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("smm_loss")
    results_df.to_csv(os.path.join(output_dir, "smm_basins.csv"), index=False)

    moments_df = pd.DataFrame([emp_moments])
    moments_df.to_csv(os.path.join(output_dir, "smm_emp_moments.csv"), index=False)

    print(f"[smm] Saved to {output_dir}/")
    return {"scan": scan_df, "basins": refined, "emp_moments": emp_moments}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_empirical_data(end_date="2019-12-31")
    df = df.dropna(subset=["u_obs", "v_obs", "G"])

    emp_u      = df["u_obs"].values.astype(float)
    emp_v      = df["v_obs"].values.astype(float)
    gdp_signal = df["G"].values.astype(float)
    time_array = np.arange(len(emp_u))

    results = fit_smm(
        emp_u, emp_v, time_array, gdp_signal,
        n_samples=500,
        n_basins=5,
        n_seeds=3,
        output_dir="output/fit_smm",
    )
