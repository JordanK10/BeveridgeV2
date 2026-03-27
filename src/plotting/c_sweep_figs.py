"""Grid scatter figures for C-distribution parameter sweeps."""

import copy
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
import numpy as np

from beveridge import config


def _c_hist_bin_edges(cmax, n_bins=None):
    """``n_bins`` bars on ``[0, cmax * 1.02]`` (16 edges for 15 bins)."""
    if n_bins is None:
        n_bins = int(config.C_DISTRIBUTION_HIST_BINS)
    n_bins = max(1, int(n_bins))
    return np.linspace(0.0, float(cmax) * 1.02, n_bins + 1)


def plot_firm_sensitivity_distribution(c_values, output_path, suptitle=""):
    """
    Histogram + empirical CDF of firm-level sensitivity ``C`` (single run).

    Saves a PDF to ``output_path``.
    """
    c = np.asarray(c_values, dtype=float).ravel()
    if c.size == 0:
        raise ValueError("c_values is empty")

    cmax = float(np.max(c))
    if cmax <= 0:
        cmax = 0.01
    bins = _c_hist_bin_edges(cmax)

    fig, (ax_h, ax_e) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)

    ax_h.hist(c, bins=bins, density=True, color="C0", alpha=0.75, edgecolor="C0", linewidth=0.6)
    ax_h.set_xlabel(r"Firm sensitivity $C$")
    ax_h.set_ylabel("Density")
    ax_h.set_title("Histogram (normalized)")
    ax_h.grid(True, alpha=0.3)

    x_s = np.sort(c)
    y_s = np.arange(1, len(x_s) + 1, dtype=float) / len(x_s)
    ax_e.step(x_s, y_s, where="post", color="C0", linewidth=1.4)
    ax_e.set_xlabel(r"Firm sensitivity $C$")
    ax_e.set_ylabel(r"Empirical cdf $F(C)$")
    ax_e.set_title("CDF")
    ax_e.grid(True, alpha=0.3)
    ax_e.set_ylim(0.0, 1.01)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_c_distributions_across_sweep(
    param_name,
    param_values,
    num_firms,
    c_params_base,
    output_path,
    title_extra="",
    c_distribution_method="power_law",
):
    """
    Visual aid: for each swept parameter value, plot the implied cross-sectional
    distribution of firm-level ``C`` on the horizontal axis.

    Left: normalized histograms (overlaid, transparent). Right: empirical CDFs.
    Uses the same ``resolve_sensitivity_coefficients`` construction as simulation
    (fixed RNG seed in ``c_params_base`` implies the same uniforms when varying
    ``exponent``, etc.).
    """
    from beveridge.experiments import resolve_sensitivity_coefficients

    base = copy.deepcopy(c_params_base)
    vals = np.asarray(param_values, dtype=float)
    order = np.argsort(vals)
    vals_s = vals[order]

    series = []
    c_global_max = 0.0
    for val in vals_s:
        c_params = copy.deepcopy(base)
        if param_name == "zero_fraction":
            c_params["zero_fraction"] = float(val)
        elif param_name == "exponent":
            c_params["exponent"] = float(val)
        elif param_name == "c_max":
            c_params["c_max"] = float(val)
        else:
            c_params[param_name] = val
        c = resolve_sensitivity_coefficients(num_firms, c_distribution_method, c_params)
        series.append((float(val), c))
        c_global_max = max(c_global_max, float(np.max(c)))

    if c_global_max <= 0:
        c_global_max = float(base.get("c_max", 0.2))
    bins = _c_hist_bin_edges(c_global_max)

    vmin, vmax = float(np.min(vals_s)), float(np.max(vals_s))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        norm = mcolors.Normalize(vmin=vmin - 1.0, vmax=vmax + 1.0)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps["viridis"]

    fig, (ax_h, ax_e) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    for val, c in series:
        color = cmap(norm(val))
        ax_h.hist(
            c,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.28,
            color=color,
            edgecolor=color,
            linewidth=0.6,
        )
        x_s = np.sort(c)
        y_s = np.arange(1, len(x_s) + 1, dtype=float) / len(x_s)
        ax_e.step(x_s, y_s, where="post", color=color, linewidth=1.4, alpha=0.9)

    ax_h.set_xlabel(r"Firm sensitivity $C$")
    ax_h.set_ylabel("Density")
    ax_h.set_title("Histogram (normalized)")
    ax_h.grid(True, alpha=0.3)

    ax_e.set_xlabel(r"Firm sensitivity $C$")
    ax_e.set_ylabel(r"Empirical cdf $F(C)$")
    ax_e.set_title("CDF")
    ax_e.grid(True, alpha=0.3)
    ax_e.set_ylim(0.0, 1.01)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_h, ax_e], shrink=0.85, pad=0.02)
    cbar.set_label(param_name)

    ttl = f"Cross-sectional $C$ for each {param_name}"
    if title_extra:
        ttl += f"\n{title_extra}"
    fig.suptitle(ttl, fontsize=11)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _sorted_param_columns(rows, order):
    """Return xs and a getter for sorted columns by observable key."""

    def col(key):
        return np.array([r[key] for r in rows], dtype=float)[order]

    return col


def plot_c_sweep_grid_multi(series_specs, param_name, title_extra, output_path):
    """
    Same layout as :func:`plot_c_sweep_grid`, but overlays multiple sweeps on each axis.

    Args:
        series_specs: list of dicts, each with keys ``rows`` (list of result dicts),
            ``linestyle`` (e.g. ``"-"``, ``"--"``), and optional ``label`` (str).
            If ``label`` is omitted or falsy, legends use the short single-series
            names (``U rate`` / ``V rate``).
        param_name, title_extra, output_path: same as :func:`plot_c_sweep_grid`.
    """
    if not series_specs:
        raise ValueError("series_specs must be non-empty")
    for spec in series_specs:
        if not spec.get("rows"):
            raise ValueError("each series must have non-empty rows")

    marker_pairs = [("o", "o"), ("s", "s"), ("^", "^"), ("D", "D")]
    joint_colors = ["C2", "C3", "C4", "C5"]

    fig, axs = plt.subplots(3, 3, figsize=(11.5, 9.0), constrained_layout=True)
    ax_flat = axs.ravel()

    def plot_dual(ax, key_u, key_v, ylabel):
        for i, spec in enumerate(series_specs):
            rows = spec["rows"]
            ls = spec.get("linestyle", "-")
            tag = spec.get("label") or None
            x = np.array([r["param"] for r in rows], dtype=float)
            order = np.argsort(x)
            xs = x[order]
            col = _sorted_param_columns(rows, order)
            mu, mv = col(key_u), col(key_v)
            mku, mkv = marker_pairs[i % len(marker_pairs)]
            u_leg = f"U · {tag}" if tag else "U rate"
            v_leg = f"V · {tag}" if tag else "V rate"
            ax.scatter(xs, mu, color="C0", s=28, alpha=0.85, marker=mku, label=u_leg)
            ax.scatter(xs, mv, color="C1", s=28, alpha=0.85, marker=mkv, label=v_leg)
            ax.plot(xs, mu, color="C0", linewidth=0.9, alpha=0.5, linestyle=ls)
            ax.plot(xs, mv, color="C1", linewidth=0.9, alpha=0.5, linestyle=ls)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(param_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=6)

    def plot_joint(ax, key, ylabel, base_label):
        for i, spec in enumerate(series_specs):
            rows = spec["rows"]
            ls = spec.get("linestyle", "-")
            tag = spec.get("label") or None
            x = np.array([r["param"] for r in rows], dtype=float)
            order = np.argsort(x)
            xs = x[order]
            col = _sorted_param_columns(rows, order)
            y = col(key)
            color = joint_colors[i % len(joint_colors)]
            leg = base_label if not tag else f"{base_label} ({tag})"
            ax.scatter(xs, y, color=color, s=30, alpha=0.85, label=leg)
            ax.plot(xs, y, color=color, linewidth=0.9, alpha=0.5, linestyle=ls)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(param_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=6)

    plot_dual(ax_flat[0], "mean_u", "mean_v", "Mean rate")
    plot_dual(ax_flat[1], "std_u", "std_v", "Std dev")
    plot_dual(ax_flat[2], "skew_u", "skew_v", "Skewness")
    plot_dual(ax_flat[3], "ar1_u", "ar1_v", "AR(1)")

    plot_joint(ax_flat[4], "corr_uv", r"$\rho$", r"corr$(u_t, v_t)$")
    plot_joint(
        ax_flat[5],
        "max_abs_rho_demeaned",
        r"max $|\rho|$",
        r"demeaned $u,v$ CCF",
    )
    plot_joint(
        ax_flat[6],
        "max_abs_rho_delta",
        r"max $|\rho|$",
        r"$\Delta U,\Delta V$ CCF",
    )

    for j in range(7, 9):
        ax_flat[j].set_visible(False)

    fig.suptitle(f"C distribution sweep vs {param_name}\n{title_extra}", fontsize=11)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_c_sweep_grid(rows, param_name, title_extra, output_path):
    """
    Multi-panel scatter: x = swept parameter, y = observable.

    Each dual-metric panel plots U (C0) and V (C1); joint panels plot one series.

    Args:
        rows: list of dicts with keys ``param``, ``mean_u``, ``mean_v``, …
        param_name: x-axis label key (e.g. ``exponent``).
        title_extra: subtitle text (fixed hyperparameters).
        output_path: ``.pdf`` path; parent dirs created if needed.
    """
    if not rows:
        raise ValueError("rows must be non-empty")
    plot_c_sweep_grid_multi(
        [{"rows": rows, "linestyle": "-", "label": None}],
        param_name,
        title_extra,
        output_path,
    )
