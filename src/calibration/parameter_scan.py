"""
2-D parameter scans over moment residuals.

Scan 1: c_max × c_exponent  (fixed K, firing_threshold)
Scan 2: idio_std × firing_threshold (fixed c_max, c_exponent)

For each grid point, runs the market simulation and computes normalised
residuals vs empirical moments:

    r = (moment_data - moment_sim) / moment_data

Moments tracked:
  - skew(u)
  - skew(v)
  - skew(Δu)
  - corr(u, v)

Caches scan results as NPZ files in cache_dir, then regenerates plots from cache.

Usage:
    python -m src.calibration.parameter_scan [--output-dir OUTPUT_DIR] [--cache-dir CACHE_DIR]
    python -m src.calibration.parameter_scan --regen-plots [--output-dir OUTPUT_DIR] [--cache-dir CACHE_DIR]
"""

import os
import sys
import argparse
import warnings
import time
import signal
import traceback

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import skew as scipy_skew
from joblib import Parallel, delayed
from tqdm import tqdm

# Suppress openpyxl warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Timeout handler
def timeout_handler(signum, frame):
    print("\n" + "="*70)
    print("TIMEOUT: Operation took longer than 10 seconds")
    print("="*70)
    traceback.print_stack(frame)
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from calibration.empirical import load_empirical_data
from calibration.simulate import simulate_market


# ---------------------------------------------------------------------------
# Scan 1: c_max × c_exponent
# ---------------------------------------------------------------------------
C_MAX_VALUES      = np.linspace(0.05, 2.00, 16)
C_EXPONENT_VALUES = np.linspace(0.25, 5.00, 16)

ZERO_FRACTION_VALUES = [0.0625, 0.125, 0.25]  # baseline, 2x, 4x

SCAN1_DEFAULTS = dict(
    n_firms=250,
    K=2.31,
    s=0.0043,
    target_u=0.05,
    seed=42,
    firing_threshold=0.10,
)

# Reference values to mark on scan 1 axes
SCAN1_REF_C_MAX      = 0.40   # current diagnostic setting
SCAN1_REF_C_EXPONENT = 1.20   # current diagnostic setting

# ---------------------------------------------------------------------------
# Scan 2: idiosyncratic noise std × firing_threshold
# ---------------------------------------------------------------------------
IDIO_STD_VALUES  = np.linspace(0.0001, 0.25 , 16)   # 0.01% to 2.5%
FT_VALUES        = np.linspace(0.00, 0.50, 16)      # 0% – 20%

SCAN2_DEFAULTS = dict(
    n_firms=250,
    c_max=0.40,
    c_exponent=1.20,
    K=2.31,
    s=0.0043,
    target_u=0.05,
    seed=42,
)

# Reference values to mark on scan 2 axes
SCAN2_REF_IDIO_STD = 0.0089   # current default
SCAN2_REF_FT       = 0.10     # current default

# ---------------------------------------------------------------------------
# Scan 3: K × s (separation rate)
# ---------------------------------------------------------------------------
K_VALUES = np.linspace(2.31/2, 2.31*2, 16)  # 1.155 to 4.62 (avg matching time)
S_VALUES = np.linspace(0.005, 0.2, 16)  # 0.00215 to 0.0086 (separation rate)

SCAN3_DEFAULTS = dict(
    n_firms=250,
    c_max=0.40,
    c_exponent=1.20,
    firing_threshold=0.10,
    target_u=0.05,
    seed=42,
)

# Reference values to mark on scan 3 axes
SCAN3_REF_K = 2.31    # current default (avg matching time)
SCAN3_REF_S = 0.0043  # current default (separation rate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _moments(u, v, u_emp=None, v_emp=None):
    """
    Return (skew_du, skew_dv, mse).

    - skew_du: skewness of Δu (change in unemployment)
    - skew_dv: skewness of Δv (change in vacancy)
    - mse: mean squared error vs empirical data (if provided)
    """
    du = np.diff(u)
    dv = np.diff(v)
    skew_du = float(scipy_skew(du))
    skew_dv = float(scipy_skew(dv))

    # Calculate MSE if empirical data provided
    if u_emp is not None and v_emp is not None:
        # Match lengths and compute MSE
        min_len = min(len(u), len(u_emp), len(v), len(v_emp))
        u_cmp = u[:min_len]
        v_cmp = v[:min_len]
        u_emp_cmp = u_emp[:min_len]
        v_emp_cmp = v_emp[:min_len]
        mse = float(np.mean((u_cmp - u_emp_cmp)**2 + (v_cmp - v_emp_cmp)**2))
    else:
        mse = np.nan

    return (skew_du, skew_dv, mse)


def _run_single_point(i, j, kw1, v1, kw2, v2, time_array, gdp_signal, fixed_kwargs, emp_moments, emp_u, emp_v):
    """Evaluate a single grid point."""
    try:
        sim_u, sim_v = simulate_market(
            time_array, gdp_signal,
            **{kw1: v1, kw2: v2},
            **fixed_kwargs,
        )
        moments = _moments(sim_u, sim_v, emp_u, emp_v)
        return (i, j, moments)
    except Exception:
        return (i, j, [np.nan] * 4)


def _run_grid(param_pairs, fixed_kwargs, gdp_signal, time_array, emp_u, emp_v, label, n_jobs=8):
    """
    Run simulate_market over a list of (p1, p2, kw1, kw2) tuples in parallel.

    param_pairs : list of (i, j, kwarg_name_1, val_1, kwarg_name_2, val_2)
    Returns three (nr × nc) grids: [mean_du, mean_dv, mse]
    """
    nr, nc = fixed_kwargs.pop("_shape")
    grids = [np.full((nr, nc), np.nan) for _ in range(3)]
    total = len(param_pairs)

    print(f"  {label} running {total} points with {n_jobs} workers...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_point)(i, j, kw1, v1, kw2, v2, time_array, gdp_signal, fixed_kwargs, None, emp_u, emp_v)
        for i, j, kw1, v1, kw2, v2 in tqdm(param_pairs, desc=label, total=total, ncols=80)
    )

    for i, j, moments in results:
        for g, m in zip(grids, moments):
            g[i, j] = m

    print(f"  {label} scan complete.")
    return grids


def _plot_scan(grids, x_values, y_values, xlabel, ylabel, suptitle,
               ref_x, ref_y, output_dir, prefix):
    """Save individual panels + combined 2×2 for one scan using contour plots."""
    panel_meta = [
        (grids[0], r"skew($u$)",        "skew_u"),
        (grids[1], r"skew($v$)",        "skew_v"),
        (grids[2], r"skew($\Delta u$)", "skew_du"),
        (grids[3], r"corr($u$, $v$)",   "corr_uv"),
    ]

    cmap = plt.get_cmap("RdBu_r")
    extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]
    nx, ny = len(x_values), len(y_values)
    xs = np.linspace(x_values[0], x_values[-1], nx)
    ys = np.linspace(y_values[0], y_values[-1], ny)

    # Create individual subfolder
    individual_dir = os.path.join(output_dir, prefix)
    os.makedirs(individual_dir, exist_ok=True)

    def _decorate(ax, grid, title, xlabel_ax=None, ylabel_ax=None):
        if xlabel_ax:
            ax.set_xlabel(xlabel_ax, fontsize=10)
        if ylabel_ax:
            ax.set_ylabel(ylabel_ax, fontsize=10)
        ax.set_title(title, fontsize=10)
        # reference lines
        if ref_x is not None and x_values[0] <= ref_x <= x_values[-1]:
            ax.axvline(ref_x, color="lime", linewidth=1.4, linestyle=":")
        elif ref_x is not None:
            edge = x_values[0] if ref_x < x_values[0] else x_values[-1]
            ax.axvline(edge, color="lime", linewidth=1.0, linestyle=":",
                       label=f"{xlabel.strip('$')}={ref_x:.2f} (outside range)")
        if ref_y is not None and y_values[0] <= ref_y <= y_values[-1]:
            ax.axhline(ref_y, color="lime", linewidth=1.4, linestyle=":")
        elif ref_y is not None:
            edge = y_values[0] if ref_y < y_values[0] else y_values[-1]
            ax.axhline(edge, color="lime", linewidth=1.0, linestyle=":",
                       label=f"{ylabel.strip('$')}={ref_y:.2f} (outside range)")

    def _get_norm_and_levels(grid):
        """Create norm and levels for contour plot, handling edge cases."""
        finite = grid[np.isfinite(grid)]
        if finite.size > 0:
            vmin, vmax = float(finite.min()), float(finite.max())
        else:
            vmin, vmax = -1.0, 1.0

        # Ensure vmin < vcenter < vmax for TwoSlopeNorm
        vcenter = 0.0
        if vmin >= vcenter:
            vmin = vcenter - 1.0
        if vmax <= vcenter:
            vmax = vcenter + 1.0
        if vmin == vmax:
            vmin, vmax = vcenter - 1.0, vcenter + 1.0

        norm = mcolors.TwoSlopeNorm(vcenter=vcenter)
        levels = np.linspace(vmin, vmax, 25)
        return norm, levels, vmin, vmax

    # individual panels as contour plots
    for grid, title, fname in panel_meta:
        fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
        norm, levels, vmin, vmax = _get_norm_and_levels(grid)

        # Contour plot with 15 levels
        cs = ax.contourf(xs, ys, grid, levels=levels, cmap=cmap, norm=norm)
        ax.contour(xs, ys, grid, levels=levels, colors="k", linewidths=0.3, alpha=0.3)

        cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"$(m_{\rm data}-m_{\rm sim})/m_{\rm data}$", fontsize=9)
        _decorate(ax, grid, r"Normalised residual: " + title, xlabel_ax=xlabel, ylabel_ax=ylabel)

        path = os.path.join(individual_dir, f"{fname}.pdf")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")

    # combined 2×2
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(suptitle, fontsize=12)
    for ax, (grid, title, _) in zip(axs.ravel(), panel_meta):
        norm, levels, vmin, vmax = _get_norm_and_levels(grid)
        cs = ax.contourf(xs, ys, grid, levels=levels, cmap=cmap, norm=norm)
        ax.contour(xs, ys, grid, levels=levels, colors="k", linewidths=0.3, alpha=0.3)

        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        _decorate(ax, grid, r"Normalised residual: " + title)

    path = os.path.join(output_dir, f"{prefix}_combined.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _save_grids_cache(grids, cache_path):
    """Save grids to NPZ cache file."""
    np.savez(cache_path,
             skew_du=grids[0],
             skew_dv=grids[1],
             mse=grids[2])
    print(f"  Cached grids to {cache_path}")


def _load_grids_cache(cache_path):
    """Load grids from NPZ cache file."""
    data = np.load(cache_path)
    return [data['skew_du'], data['skew_dv'], data['mse']]


def _plot_scan_comparison(grids_list, zero_fractions, x_values, y_values, xlabel, ylabel, suptitle,
                          ref_x, ref_y, output_dir, prefix):
    """Plot 3 side-by-side heatmaps comparing different zero_fraction values."""
    panel_meta = [
        (0, r"skew($\Delta u$)",   "skew_du"),
        (1, r"skew($\Delta v$)",   "skew_dv"),
        (2, r"MSE vs empirical",  "mse"),
    ]

    cmap = plt.get_cmap("RdBu_r")
    extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]
    nx, ny = len(x_values), len(y_values)
    xs = np.linspace(x_values[0], x_values[-1], nx)
    ys = np.linspace(y_values[0], y_values[-1], ny)

    # Create comparison subfolder
    comparison_dir = os.path.join(output_dir, f"{prefix}_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    def _decorate(ax, grid, title, xlabel_ax=None, ylabel_ax=None, show_x=True, show_y=True):
        if show_x and xlabel_ax:
            ax.set_xlabel(xlabel_ax, fontsize=12)
        if show_y and ylabel_ax:
            ax.set_ylabel(ylabel_ax, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=9)
        # reference lines
        if ref_x is not None and x_values[0] <= ref_x <= x_values[-1]:
            ax.axvline(ref_x, color="lime", linewidth=1.2, linestyle=":")
        if ref_y is not None and y_values[0] <= ref_y <= y_values[-1]:
            ax.axhline(ref_y, color="lime", linewidth=1.2, linestyle=":")

    def _get_norm_and_levels(grids):
        """Get shared norm across all grids with center at 0."""
        all_finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids])
        if all_finite.size > 0:
            vmin, vmax = float(all_finite.min()), float(all_finite.max())
        else:
            vmin, vmax = -1.0, 1.0

        vcenter = 0.0

        # Ensure vmin < vcenter < vmax
        if vmin >= vcenter:
            vmin = vcenter - (abs(vmax) + 1e-6 if vmax > vcenter else 1.0)
        if vmax <= vcenter:
            vmax = vcenter + (abs(vmin) + 1e-6 if vmin < vcenter else 1.0)
        if vmin >= vmax:
            vmin, vmax = -1.0, 1.0

        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        levels = np.linspace(vmin, vmax, 25)
        return norm, levels

    # For each panel metric, create a 3-panel side-by-side figure
    for grid_idx, title, fname in panel_meta:
        t0 = time.time()

        fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

        # Extract the grid for this metric from all zero_fraction variants
        grids = [grids_list[i][grid_idx] for i in range(3)]
        norm, levels = _get_norm_and_levels(grids)

        for ax_idx, (ax, grid, zf) in enumerate(zip(axs, grids, zero_fractions)):
            cs = ax.contourf(xs, ys, grid, levels=levels, cmap=cmap, norm=norm)
            # White contour at zero
            ax.contour(xs, ys, grid, levels=[0], colors="white", linewidths=2.0)
            # Subtle grid contours
            ax.contour(xs, ys, grid, levels=levels, colors="k", linewidths=0.2, alpha=0.2)

            # Bold black outline around contour closest to zero
            closest_level = levels[np.argmin(np.abs(levels))]
            ax.contour(xs, ys, grid, levels=[closest_level], colors="black", linewidths=3.5)

            subtitle = f"fraction of C=0: {zf:.4f}"
            _decorate(ax, grid, subtitle, xlabel_ax=xlabel if ax_idx == 1 else None,
                     ylabel_ax=ylabel if ax_idx == 0 else None)

        # Add shared colorbar
        cbar = fig.colorbar(cs, ax=axs, fraction=0.046, pad=0.04,
                           label=r"$(m_{\rm data}-m_{\rm sim})/m_{\rm data}$")
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(r"$(m_{\rm data}-m_{\rm sim})/m_{\rm data}$", fontsize=12)
        fig.suptitle(f"{title} — {suptitle}", fontsize=12)

        path = os.path.join(comparison_dir, f"{fname}_comparison.pdf")
        t1 = time.time()
        fig.savefig(path, dpi=150)
        save_time = time.time() - t1
        plt.close(fig)

        total_time = time.time() - t0
        if total_time > 10:
            print(f"  {fname}: {total_time:.2f}s total ({save_time:.2f}s save)")
        else:
            print(f"  Saved {path} ({total_time:.2f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scan(output_dir="output/residual_plots", cache_dir="output/scan_cache", skip_plotting=False):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Check which cache files exist and load them
    print("Checking cache status...")
    signal.signal(signal.SIGALRM, timeout_handler)

    signal.alarm(10)
    t0 = time.time()
    grids1_list = _load_grids_from_cache(cache_dir, "scan1_cmax_cexp", ZERO_FRACTION_VALUES)
    t1 = time.time()
    signal.alarm(0)
    if grids1_list is not None:
        print(f"✓ Scan 1 loaded from cache ({t1 - t0:.4f}s)")
    else:
        print("✗ Scan 1 cache missing — will regenerate")

    signal.alarm(10)
    t0 = time.time()
    grids2_list = _load_grids_from_cache(cache_dir, "scan2_idio_ft", ZERO_FRACTION_VALUES)
    t1 = time.time()
    signal.alarm(0)
    if grids2_list is not None:
        print(f"✓ Scan 2 loaded from cache ({t1 - t0:.4f}s)")
    else:
        print("✗ Scan 2 cache missing — will regenerate")

    signal.alarm(10)
    t0 = time.time()
    grids3_list = _load_grids_from_cache(cache_dir, "scan3_k_s", ZERO_FRACTION_VALUES)
    t1 = time.time()
    signal.alarm(0)
    if grids3_list is not None:
        print(f"✓ Scan 3 loaded from cache ({t1 - t0:.4f}s)")
    else:
        print("✗ Scan 3 cache missing — will regenerate")

    # If all cache exists, skip scans and go directly to plotting
    if grids1_list is not None and grids2_list is not None and grids3_list is not None:
        print("\nAll cache complete. Skipping scans...")
        if skip_plotting:
            print("Data cached. Skipping plot generation.")
            return
        print("\nGenerating plots from cached data...")
        signal.alarm(10)
        t0 = time.time()
        _generate_all_plots(grids1_list, grids2_list, grids3_list, output_dir)
        t1 = time.time()
        signal.alarm(0)
        print(f"Plot generation time: {t1 - t0:.4f}s")
        return

    # Otherwise, proceed to run missing scans only
    print("\nLoading empirical data...")
    emp = load_empirical_data()
    emp = emp.dropna(subset=["u_obs", "v_obs", "G"])

    emp_u      = emp["u_obs"].values.astype(float)
    emp_v      = emp["v_obs"].values.astype(float)
    gdp_signal = emp["G"].values.astype(float)
    time_array = np.arange(len(emp_u))

    skew_du_data, skew_dv_data, _ = _moments(emp_u, emp_v)

    print(f"Empirical moments:")
    print(f"  skew(Δu) = {skew_du_data:.6f}")
    print(f"  skew(Δv) = {skew_dv_data:.6f}\n")

    # ------------------------------------------------------------------
    # Scan 1: c_max × c_exponent (3 variants with different zero_fraction)
    # ------------------------------------------------------------------
    if grids1_list is None:
        print("Running scan 1: c_max × c_exponent ...")
        nr1, nc1 = len(C_EXPONENT_VALUES), len(C_MAX_VALUES)
        grids1_list = []
        for zf in ZERO_FRACTION_VALUES:
            print(f"  Scan 1 variant: zero_fraction={zf:.4f}")
            pairs1 = [
                (i, j, "c_max", c_max, "c_exponent", c_exp)
                for i, c_exp in enumerate(C_EXPONENT_VALUES)
                for j, c_max in enumerate(C_MAX_VALUES)
            ]
            fixed1 = dict(SCAN1_DEFAULTS, zero_fraction=zf, _shape=(nr1, nc1))
            grids1 = _run_grid(pairs1, fixed1, gdp_signal, time_array, emp_u, emp_v, f"scan1_zf{zf:.4f}")
            grids1_list.append(grids1)

            # Cache the grids
            cache_path = os.path.join(cache_dir, f"scan1_cmax_cexp_zf{zf:.4f}.npz")
            _save_grids_cache(grids1, cache_path)

    # ------------------------------------------------------------------
    # Scan 2: idio_std × firing_threshold (3 variants with different zero_fraction)
    # ------------------------------------------------------------------
    if grids2_list is None:
        print("Running scan 2: idio_std × firing_threshold ...")
        nr2, nc2 = len(FT_VALUES), len(IDIO_STD_VALUES)
        grids2_list = []
        for zf in ZERO_FRACTION_VALUES:
            print(f"  Scan 2 variant: zero_fraction={zf:.4f}")
            pairs2 = [
                (i, j, "idio_std", idio_std, "firing_threshold", ft)
                for i, ft in enumerate(FT_VALUES)
                for j, idio_std in enumerate(IDIO_STD_VALUES)
            ]
            fixed2 = dict(SCAN2_DEFAULTS, zero_fraction=zf, _shape=(nr2, nc2))
            grids2 = _run_grid(pairs2, fixed2, gdp_signal, time_array, emp_u, emp_v, f"scan2_zf{zf:.4f}")
            grids2_list.append(grids2)

            # Cache the grids
            cache_path = os.path.join(cache_dir, f"scan2_idio_ft_zf{zf:.4f}.npz")
            _save_grids_cache(grids2, cache_path)

    # ------------------------------------------------------------------
    # Scan 3: K × s (separation rate) (3 variants with different zero_fraction)
    # ------------------------------------------------------------------
    if grids3_list is None:
        print("Running scan 3: K × s ...")
        nr3, nc3 = len(S_VALUES), len(K_VALUES)
        grids3_list = []
        for zf in ZERO_FRACTION_VALUES:
            print(f"  Scan 3 variant: zero_fraction={zf:.4f}")
            pairs3 = [
                (i, j, "K", k, "s", s)
                for i, s in enumerate(S_VALUES)
                for j, k in enumerate(K_VALUES)
            ]
            fixed3 = dict(SCAN3_DEFAULTS, zero_fraction=zf, _shape=(nr3, nc3))
            grids3 = _run_grid(pairs3, fixed3, gdp_signal, time_array, emp_u, emp_v, f"scan3_zf{zf:.4f}")
            grids3_list.append(grids3)

            # Cache the grids
            cache_path = os.path.join(cache_dir, f"scan3_k_s_zf{zf:.4f}.npz")
            _save_grids_cache(grids3, cache_path)

    # Skip plotting if requested (e.g., for data-only runs)
    if skip_plotting:
        print("\nData caching complete. Skipping plot generation.")
        return

    # ------------------------------------------------------------------
    # Generate plots from cached data
    # ------------------------------------------------------------------
    print("\nGenerating plots from cached data...")
    _generate_all_plots(grids1_list, grids2_list, grids3_list, output_dir)


def _load_grids_from_cache(cache_dir, scan_name, zero_fractions):
    """Load cached grids for a given scan across all zero_fraction variants."""
    grids_list = []
    for zf in zero_fractions:
        cache_path = os.path.join(cache_dir, f"{scan_name}_zf{zf:.4f}.npz")
        if not os.path.exists(cache_path):
            return None  # Cache incomplete

        t0 = time.time()
        data = np.load(cache_path, mmap_mode='r')
        load_time = time.time() - t0

        t1 = time.time()
        grids = [np.array(data['skew_du']), np.array(data['skew_dv']),
                 np.array(data['mse'])]
        access_time = time.time() - t1

        grids_list.append(grids)

        if load_time > 0.1 or access_time > 0.1:
            print(f"  [{scan_name}_zf{zf:.4f}] load: {load_time:.4f}s, access: {access_time:.4f}s")

    return grids_list


def _cache_complete(cache_dir):
    """Check if all required cache files exist."""
    required_files = [
        f"scan1_cmax_cexp_zf{zf:.4f}.npz"
        for zf in ZERO_FRACTION_VALUES
    ] + [
        f"scan2_idio_ft_zf{zf:.4f}.npz"
        for zf in ZERO_FRACTION_VALUES
    ] + [
        f"scan3_k_s_zf{zf:.4f}.npz"
        for zf in ZERO_FRACTION_VALUES
    ]

    return all(os.path.exists(os.path.join(cache_dir, f)) for f in required_files)


def _generate_all_plots(grids1_list, grids2_list, grids3_list, output_dir):
    """Generate comparison plots only (no individual scan folders)."""

    # Plot Scan 1 comparison (3 side-by-side for each metric)
    print("\n  Generating Scan 1 comparison plots...")
    _plot_scan_comparison(
        grids1_list,
        zero_fractions=ZERO_FRACTION_VALUES,
        x_values=C_MAX_VALUES,
        y_values=C_EXPONENT_VALUES,
        xlabel=r"$C_{\max}$",
        ylabel=r"Power-law exponent $\alpha$",
        suptitle=(
            r"Scan 1 — Normalised residuals $(m_{\rm data}-m_{\rm sim})/m_{\rm data}$"
            r" ($C_{\max}$ × power-law exponent)"
        ),
        ref_x=SCAN1_REF_C_MAX,
        ref_y=SCAN1_REF_C_EXPONENT,
        output_dir=output_dir,
        prefix="scan1_cmax_cexp",
    )

    # Plot Scan 2 comparison (3 side-by-side for each metric)
    print("\n  Generating Scan 2 comparison plots...")
    _plot_scan_comparison(
        grids2_list,
        zero_fractions=ZERO_FRACTION_VALUES,
        x_values=IDIO_STD_VALUES,
        y_values=FT_VALUES,
        xlabel=r"Idiosyncratic noise std $\sigma_\eta$",
        ylabel=r"Firing threshold $\theta$",
        suptitle=(
            r"Scan 2 — Normalised residuals $(m_{\rm data}-m_{\rm sim})/m_{\rm data}$"
            r" ($\sigma_\eta$ × firing threshold)"
        ),
        ref_x=SCAN2_REF_IDIO_STD,
        ref_y=SCAN2_REF_FT,
        output_dir=output_dir,
        prefix="scan2_idio_ft",
    )

    # Plot Scan 3 comparison (3 side-by-side for each metric)
    print("\n  Generating Scan 3 comparison plots...")
    _plot_scan_comparison(
        grids3_list,
        zero_fractions=ZERO_FRACTION_VALUES,
        x_values=K_VALUES,
        y_values=S_VALUES,
        xlabel=r"$K$ (matching constant)",
        ylabel=r"Separation rate $s$",
        suptitle=(
            r"Scan 3 — Normalised residuals $(m_{\rm data}-m_{\rm sim})/m_{\rm data}$"
            r" ($K$ × separation rate)"
        ),
        ref_x=SCAN3_REF_K,
        ref_y=SCAN3_REF_S,
        output_dir=output_dir,
        prefix="scan3_k_s",
    )

    print(f"\nPlot generation complete. Output saved to {output_dir}")


def regenerate_plots_from_cache(output_dir="output/residual_plots", cache_dir="output/scan_cache"):
    """Regenerate all comparison plots from cached NPZ data without running scans."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading cached scan data...")

    # Load Scan 1
    print("\nScan 1 (c_max × c_exponent):")
    grids1_list = _load_grids_from_cache(cache_dir, "scan1_cmax_cexp", ZERO_FRACTION_VALUES)
    if grids1_list is None:
        print("ERROR: Scan 1 cache not found. Run full scan first with: python -m src.calibration.parameter_scan")
        return

    # Load Scan 2
    print("Scan 2 (idio_std × firing_threshold):")
    grids2_list = _load_grids_from_cache(cache_dir, "scan2_idio_ft", ZERO_FRACTION_VALUES)
    if grids2_list is None:
        print("ERROR: Scan 2 cache not found. Run full scan first with: python -m src.calibration.parameter_scan")
        return

    # Load Scan 3
    print("Scan 3 (K × s):")
    grids3_list = _load_grids_from_cache(cache_dir, "scan3_k_s", ZERO_FRACTION_VALUES)
    if grids3_list is None:
        print("ERROR: Scan 3 cache not found. Run full scan first with: python -m src.calibration.parameter_scan")
        return

    # Generate plots
    print("\nGenerating plots from cached data...")
    _generate_all_plots(grids1_list, grids2_list, grids3_list, output_dir)


def check_cache_status(cache_dir="output/scan_cache"):
    """Check and report which cache files exist."""
    print(f"Checking cache status in {cache_dir}...\n")

    scans = [
        ("Scan 1 (c_max × c_exponent)", [f"scan1_cmax_cexp_zf{zf:.4f}.npz" for zf in ZERO_FRACTION_VALUES]),
        ("Scan 2 (idio_std × firing_threshold)", [f"scan2_idio_ft_zf{zf:.4f}.npz" for zf in ZERO_FRACTION_VALUES]),
        ("Scan 3 (K × s)", [f"scan3_k_s_zf{zf:.4f}.npz" for zf in ZERO_FRACTION_VALUES]),
    ]

    all_complete = True
    for scan_name, files in scans:
        missing = [f for f in files if not os.path.exists(os.path.join(cache_dir, f))]
        if not missing:
            print(f"✓ {scan_name}: COMPLETE ({len(files)} files)")
        else:
            print(f"✗ {scan_name}: INCOMPLETE ({len(files) - len(missing)}/{len(files)} files)")
            all_complete = False

    print()
    if all_complete:
        print("All cache files exist. Use --regen-plots to regenerate plots without re-running scans.")
    else:
        print("Some cache files are missing. Run full scan to regenerate all cache.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run parameter scans or regenerate plots from cache"
    )
    parser.add_argument(
        "--output-dir",
        default="output/residual_plots",
        help="Output directory for plots (default: output/residual_plots)"
    )
    parser.add_argument(
        "--cache-dir",
        default="output/scan_cache",
        help="Cache directory containing NPZ files (default: output/scan_cache)"
    )
    parser.add_argument(
        "--regen-plots",
        action="store_true",
        help="Regenerate plots from existing cache without running scans"
    )
    parser.add_argument(
        "--skip-plotting",
        action="store_true",
        help="Run scans but skip plot generation"
    )
    parser.add_argument(
        "--check-cache",
        action="store_true",
        help="Check cache status and report which files exist"
    )

    args = parser.parse_args()

    if args.check_cache:
        check_cache_status(cache_dir=args.cache_dir)
    elif args.regen_plots:
        regenerate_plots_from_cache(output_dir=args.output_dir, cache_dir=args.cache_dir)
    else:
        run_scan(output_dir=args.output_dir, cache_dir=args.cache_dir, skip_plotting=args.skip_plotting)
