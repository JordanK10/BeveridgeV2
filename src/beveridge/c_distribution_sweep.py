"""Sweep power-law C parameters and summarize post–burn-in aggregate U/V observables."""

import copy
import os

import numpy as np
from scipy import stats

from plotting.c_sweep_figs import plot_c_distributions_across_sweep, plot_c_sweep_grid
from plotting.uv_crosscorr import _rates_from_levels, crosscorr_pearson

from . import config
from .experiments import MAX_LAG_STEPS, _simulate_multi_firm

_POWER_LAW_PARAM_KEYS = frozenset({"exponent", "c_max", "zero_fraction"})


def format_c_sweep_subtitle(
    num_firms,
    param_name,
    c_params_base,
    c_distribution_method,
    signal_clause,
):
    """Shared subtitle for sweep grid and C-distribution cross-section figures."""
    held = [f"{num_firms} firms", signal_clause, str(c_distribution_method)]
    if param_name != "c_max":
        held.append(f"c_max={c_params_base.get('c_max')}")
    if param_name != "exponent":
        held.append(f"exponent={c_params_base.get('exponent')}")
    if param_name != "zero_fraction":
        held.append(f"zero_fraction={c_params_base.get('zero_fraction')}")
    held.append(f"seed={c_params_base.get('seed')}")
    if c_params_base.get("power_law_flip"):
        held.append("C=c_max(1-U^(1/α))")
    else:
        held.append("C=c_max·U^(1/α)")
    if c_distribution_method == "power_law":
        mz = int(
            c_params_base.get("power_law_max_zero_firms", config.POWER_LAW_MAX_ZERO_FIRMS)
        )
        held.append(f"C=0 cap={mz} firms")
    return " · ".join(held)


def _ar1_coef(x):
    x = np.asarray(x, dtype=float).ravel()
    if len(x) < 3:
        return float("nan")
    a, b = x[:-1], x[1:]
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    c = np.corrcoef(a, b)
    return float(c[0, 1])


def compute_post_burn_in_observables(
    unemployment,
    aggregate_vacancies,
    population,
    burn_in,
    max_lag_steps,
):
    """
    Post–burn-in statistics on aggregate rates u=U/L, v=V/(V+E), plus UV joint metrics.

    Uses the same rate construction as ``plot_fluctuation_analysis`` / UV CCF helpers.
    """
    ur, vr = _rates_from_levels(unemployment, aggregate_vacancies, population, burn_in)

    skew_u = float(stats.skew(ur, bias=False))
    skew_v = float(stats.skew(vr, bias=False))
    std_u = float(np.std(ur, ddof=1)) if len(ur) > 1 else float("nan")
    std_v = float(np.std(vr, ddof=1)) if len(vr) > 1 else float("nan")
    mean_u = float(np.mean(ur))
    mean_v = float(np.mean(vr))

    ar1_u = _ar1_coef(ur)
    ar1_v = _ar1_coef(vr)

    if np.std(ur) < 1e-15 or np.std(vr) < 1e-15:
        corr_uv = float("nan")
    else:
        corr_uv = float(np.corrcoef(ur, vr)[0, 1])

    u_dm = ur - np.mean(ur)
    v_dm = vr - np.mean(vr)
    min_seg = 10
    L = min(int(max_lag_steps), max(0, len(u_dm) - min_seg))
    _, rho_uv, _ = crosscorr_pearson(u_dm, v_dm, L)
    finite_r = np.isfinite(rho_uv)
    max_abs_rho_demeaned = float(np.max(np.abs(rho_uv[finite_r]))) if np.any(finite_r) else float("nan")

    U = np.asarray(unemployment, dtype=float)[burn_in:]
    V = np.asarray(aggregate_vacancies, dtype=float)[burn_in:]
    dU = np.diff(U)
    dV = np.diff(V)
    Ld = min(int(max_lag_steps), max(0, len(dU) - min_seg))
    _, rho_d, _ = crosscorr_pearson(dU, dV, Ld)
    finite_d = np.isfinite(rho_d)
    max_abs_rho_delta = float(np.max(np.abs(rho_d[finite_d]))) if np.any(finite_d) else float("nan")

    return {
        "mean_u": mean_u,
        "mean_v": mean_v,
        "std_u": std_u,
        "std_v": std_v,
        "skew_u": skew_u,
        "skew_v": skew_v,
        "ar1_u": ar1_u,
        "ar1_v": ar1_v,
        "corr_uv": corr_uv,
        "max_abs_rho_demeaned": max_abs_rho_demeaned,
        "max_abs_rho_delta": max_abs_rho_delta,
    }


def run_c_distribution_sweep(
    param_name,
    param_values,
    num_firms=None,
    c_params_base=None,
    signal_name=None,
    output_dir=None,
    c_distribution_method="power_law",
    write_sweep_grid_pdf=True,
):
    """
    For each value in ``param_values``, set ``c_params_base[param_name]`` and run
    multi-firm simulation; collect observables and write a grid scatter PDF.

    ``param_name`` must be one of ``exponent``, ``c_max``, ``zero_fraction`` when
    using ``power_law`` (the usual case).

    Returns:
        list of dicts, each with key ``param`` plus all keys from
        :func:`compute_post_burn_in_observables`.

    If ``num_firms`` is None, uses :data:`config.MULTI_FIRM_COUNT` (labor force
    scales with firm count via :func:`initialize_economy`).

    Set ``write_sweep_grid_pdf=False`` when batching multiple signals into one
    combined figure (see :func:`plotting.c_sweep_figs.plot_c_sweep_grid_multi`).
    """
    if num_firms is None:
        num_firms = config.MULTI_FIRM_COUNT

    if c_distribution_method == "power_law" and param_name not in _POWER_LAW_PARAM_KEYS:
        raise ValueError(
            f"param_name must be one of {_POWER_LAW_PARAM_KEYS} for power_law; got {param_name!r}"
        )

    if c_params_base is None:
        c_params_base = {
            "c_max": 0.08,
            "exponent": 2.0,
            "zero_fraction": 0.25,
            "seed": 42,
            "power_law_flip": True,
        }
    else:
        c_params_base = copy.deepcopy(c_params_base)

    if signal_name is None:
        signal_name = config.GDP_SIGNAL_NAME

    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, "c_distribution_sweep")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for val in param_values:
        c_params = copy.deepcopy(c_params_base)
        if param_name == "zero_fraction":
            c_params["zero_fraction"] = float(val)
        elif param_name == "exponent":
            c_params["exponent"] = float(val)
        elif param_name == "c_max":
            c_params["c_max"] = float(val)
        else:
            c_params[param_name] = val

        sim = _simulate_multi_firm(
            num_firms,
            c_distribution_method,
            c_params,
            signal_name=signal_name,
        )
        obs = compute_post_burn_in_observables(
            sim["unemployment"],
            sim["aggregate_vacancies"],
            sim["population"],
            config.BURN_IN,
            MAX_LAG_STEPS,
        )
        rows.append({"param": float(val), **obs})

    title_extra = format_c_sweep_subtitle(
        num_firms,
        param_name,
        c_params_base,
        c_distribution_method,
        f"signal={signal_name}",
    )

    # Default experiment uses flipped quantile; legacy mapping gets a distinct filename.
    legacy_suffix = "" if c_params_base.get("power_law_flip") else "_legacy_cdf_u"
    out_name = f"sweep_{param_name}_{signal_name}{legacy_suffix}.pdf"
    out_path = os.path.join(output_dir, out_name)
    if write_sweep_grid_pdf:
        plot_c_sweep_grid(rows, param_name, title_extra, out_path)
        print(f"Saved C-distribution sweep figure to {out_path}")

    if c_distribution_method == "power_law":
        dist_path = os.path.join(
            output_dir, f"c_distributions_{param_name}_{signal_name}{legacy_suffix}.pdf"
        )
        plot_c_distributions_across_sweep(
            param_name,
            param_values,
            num_firms,
            c_params_base,
            dist_path,
            title_extra=title_extra,
            c_distribution_method=c_distribution_method,
        )
        print(f"Saved C cross-section distributions figure to {dist_path}")

    return rows
