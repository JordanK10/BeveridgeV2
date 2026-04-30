"""
Plot simulated vs empirical trajectories for all refined MSE-fit basin parameter sets.

Layout per basin (one PDF per objective):
  Left column  (3 rows, shared x-axis):
    [0] u(t) empirical vs simulated
    [1] v(t) empirical vs simulated
    [2] GDP signal G(t)
  Right column (3 rows):
    [0] Δu distribution — empirical vs simulated (histogram + KDE)
    [1] Δv distribution — empirical vs simulated
    [2] top panel : employment time series for 10 most GDP-responsive firms
                    + 2 unresponsive firms
        bottom panel: corr(u, G) and corr(v, G) at lags −12 … +12 months

Usage
-----
    python src/calibration/fit_plots.py                     # default: output/fit_weighted
    python src/calibration/fit_plots.py output/fit
    python src/calibration/fit_plots.py output/fit_weighted output/plots_fit_weighted
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from calibration.simulate import simulate_market_with_firms
    from calibration.empirical import load_empirical_data
    from calibration.fit import FIXED_PARAMS, PARAM_NAMES
except ImportError:
    from simulate import simulate_market_with_firms
    from empirical import load_empirical_data
    from fit import FIXED_PARAMS, PARAM_NAMES

OBJECTIVES = ["mse_d_combined", "mse_bc", "mse_flow", "mse_path"]

N_RESPONSIVE  = 10   # most GDP-sensitive firms to plot
N_INSENSITIVE = 2    # least GDP-sensitive firms to plot
MAX_LAG       = 12   # months for cross-correlation panel
FIT_END       = pd.Timestamp("2020-01-01")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_empirical():
    df = load_empirical_data().dropna(subset=["u_obs", "v_obs", "G"])
    log_gdp = np.log(df["gdp_real_level"].values.astype(float))
    return (
        df["u_obs"].values.astype(float),
        df["v_obs"].values.astype(float),
        df["G"].values.astype(float),
        np.arange(len(df)),
        df.index,
        log_gdp,
    )


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_simulation(params, time_array, gdp_signal, emp_u):
    """Run simulation and return (u, v, firms). Fixed params merged from FIXED_PARAMS.
    target_u is set to the empirical mean of u so the labor force matches the data."""
    return simulate_market_with_firms(
        time_array=time_array,
        gdp_signal=gdp_signal,
        n_firms=250,
        c_max=FIXED_PARAMS["c_max"],
        burn_in_steps=5,
        seed=42,
        K=params["K"],
        s=FIXED_PARAMS["s"],
        c_exponent=params["c_exponent"],
        zero_fraction=FIXED_PARAMS["zero_fraction"],
        firing_threshold=FIXED_PARAMS["firing_threshold"],
        idio_std=params["idio_std"],
        target_u=float(emp_u[0]),
    )


# ── Panel helpers ──────────────────────────────────────────────────────────────

def _plot_timeseries(ax, dates, emp, sim, ylabel, color_sim, n):
    ax.plot(dates[:n], emp[:n] * 100, color="black", lw=1.4, label="Empirical")
    ax.plot(dates[:n], sim[:n] * 100, color=color_sim, lw=1.1, ls="--", label="Simulated")
    ax.axvline(FIT_END, color="gray", lw=0.8, ls=":")
    ax.set_ylabel(ylabel, fontsize=16)
    ax.legend(fontsize=14, loc="upper left")
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.25)


def _plot_gdp(ax, dates, gdp, n, log_gdp=None):
    colors = ["tab:red" if g < 0 else "tab:green" for g in gdp[:n]]
    ax.bar(dates[:n], gdp[:n], color=colors, alpha=0.7, width=25, label="G(t)")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(FIT_END, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("G(t)", fontsize=16)
    ax.set_xlabel("Date", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.25)

    if log_gdp is not None:
        ax2 = ax.twinx()
        ax2.plot(dates[:n], log_gdp[:n], color="tab:blue", lw=1.2, label="log GDP")
        ax2.set_ylabel("log GDP", fontsize=16, color="tab:blue")
        ax2.tick_params(axis="y", labelsize=14, labelcolor="tab:blue")


def _plot_diff_dist_pair(ax, emp_vals, sim_vals, label, color_sim, n_bins=120):
    """Histogram + KDE overlay for a single first-difference series.

    x-window: [min(emp,sim mean − 2σ) − 1, max(emp,sim mean + 2σ) + 1] in pp.
    """
    emp_pp = emp_vals * 100
    sim_pp = sim_vals * 100

    lows  = [emp_pp.mean() - 2 * emp_pp.std(), sim_pp.mean() - 2 * sim_pp.std()]
    highs = [emp_pp.mean() + 2 * emp_pp.std(), sim_pp.mean() + 2 * sim_pp.std()]
    left  = min(lows)  - 1.0
    right = max(highs) + 1.0
    bins  = np.linspace(left, right, n_bins + 1)

    for vals, color, lbl in [(emp_pp, "black", "Empirical"),
                             (sim_pp, color_sim, "Simulated")]:
        ax.hist(vals, bins=bins, density=True, alpha=0.35, color=color, label=lbl)
        if len(vals) > 3 and np.std(vals) > 1e-8:
            kde = gaussian_kde(vals)
            xs = np.linspace(left, right, 400)
            ax.plot(xs, kde(xs), color=color, lw=1.1)

    ax.axvline(0, color="gray", lw=0.7, ls=":")
    ax.set_xlim(left, right)
    ax.set_xlabel(label, fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    emp_sk = float(pd.Series(emp_vals).skew())
    sim_sk = float(pd.Series(sim_vals).skew())
    ax.set_title(f"skew: emp={emp_sk:.2f}  sim={sim_sk:.2f}", fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.25)


def _plot_firm_employment(ax, firms, gdp_signal, dates, n,
                          n_responsive=N_RESPONSIVE, n_insensitive=N_INSENSITIVE):
    """Plot employment time series for the most/least GDP-responsive firms."""
    c_vals = np.array([f.c for f in firms])
    sorted_idx = np.argsort(c_vals)

    insensitive_idx = sorted_idx[:n_insensitive]
    responsive_idx  = sorted_idx[-n_responsive:]

    cmap_r = plt.get_cmap("Reds")
    cmap_i = plt.get_cmap("Blues")

    for k, idx in enumerate(responsive_idx):
        f = firms[idx]
        emp = f.employment[:n] / f.sigma   # normalise by baseline size
        alpha = 0.4 + 0.5 * k / max(n_responsive - 1, 1)
        ax.plot(dates[:n], emp, color=cmap_r(0.4 + 0.5 * k / max(n_responsive - 1, 1)),
                lw=0.8, alpha=alpha,
                label=f"Resp c={f.c:.2f}" if k == n_responsive - 1 else None)

    for k, idx in enumerate(insensitive_idx):
        f = firms[idx]
        emp = f.employment[:n] / f.sigma
        ax.plot(dates[:n], emp, color=cmap_i(0.5 + 0.4 * k / max(n_insensitive - 1, 1)),
                lw=1.0, ls="--", alpha=0.8,
                label=f"Insens c={f.c:.2f}" if k == 0 else None)

    ax.axvline(FIT_END, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("e / σ", fontsize=16)
    ax.set_title(
        f"Firm employment: {n_responsive} most responsive (red), "
        f"{n_insensitive} unresponsive (blue)",
        fontsize=14,
    )
    ax.legend(fontsize=12, ncol=2, loc="upper left")
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.25)


def _cross_corr(x, y, max_lag):
    """Cross-correlation of x with y at lags -max_lag … +max_lag.
    Positive lag k means x leads y by k periods.
    """
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag >= 0:
            corrs.append(float(np.mean(x[lag:] * y[:len(x) - lag])) if len(x) > lag else np.nan)
        else:
            corrs.append(float(np.mean(x[:len(x) + lag] * y[-lag:])) if len(x) > -lag else np.nan)
    return list(lags), corrs


def _plot_beveridge(ax, emp_u, emp_v, sim_u, sim_v, n):
    """Beveridge curve: scatter empirical and simulated (u, v) trajectories."""
    eu = emp_u[:n] * 100
    ev = emp_v[:n] * 100
    su = sim_u[:n] * 100
    sv = sim_v[:n] * 100

    # Empirical trajectory: connected line, color shows time progression
    times = np.arange(n)
    ax.scatter(eu, ev, c=times, cmap="Greys", s=10, alpha=0.7,
               edgecolors="black", linewidths=0.3, label="Empirical")
    ax.plot(eu, ev, color="black", lw=0.4, alpha=0.3)

    # Simulated trajectory: same color scheme but in red
    ax.scatter(su, sv, c=times, cmap="Reds", s=10, alpha=0.7,
               edgecolors="darkred", linewidths=0.3, label="Simulated")
    ax.plot(su, sv, color="tab:red", lw=0.4, alpha=0.3)

    # Mark start and end of each trajectory
    ax.scatter(eu[0], ev[0], color="green", s=60, marker="o",
               edgecolors="black", zorder=5, label="start")
    ax.scatter(eu[-1], ev[-1], color="purple", s=60, marker="s",
               edgecolors="black", zorder=5, label="end")

    ax.set_xlabel("u (%)", fontsize=16)
    ax.set_ylabel("v (%)", fontsize=16)
    ax.set_title("Beveridge curve (empirical = grey, simulated = red)", fontsize=16)
    ax.legend(fontsize=14, loc="best")
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.25)


def _plot_gdp_correlations(ax, u, v, gdp, n, max_lag=MAX_LAG):
    """Cross-correlations of u and v with the GDP signal at various lags."""
    u_s, v_s, g_s = u[:n], v[:n], gdp[:n]
    lags, corr_u = _cross_corr(u_s, g_s, max_lag)
    _,    corr_v = _cross_corr(v_s, g_s, max_lag)

    ax.bar([l - 0.2 for l in lags], corr_u, width=0.38,
           color="tab:blue", alpha=0.7, label="corr(u, G)")
    ax.bar([l + 0.2 for l in lags], corr_v, width=0.38,
           color="tab:orange", alpha=0.7, label="corr(v, G)")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="gray", lw=0.7, ls=":")
    ax.set_xlabel("Lag (months, + = u/v leads G)", fontsize=16)
    ax.set_ylabel("Correlation", fontsize=16)
    ax.set_title("Cross-correlation with GDP", fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.25)


# ── Per-basin figure ───────────────────────────────────────────────────────────

def plot_basin(params, basin_id, loss, at_boundary,
               emp_u, emp_v, gdp_signal, time_array, dates, log_gdp=None):
    """
    Build and return a figure for one basin parameter set.

    Layout (2 columns):
      Left  (3 rows, shared x): u(t) | v(t) | G(t)
      Right (3 rows):
        [0] Beveridge curve (u vs v, empirical vs simulated)
        [1] Δu and Δv distribution panels side-by-side (each ±2σ ± 1pp window)
        [2] firm employment (top) + GDP cross-correlations (bottom)
    """
    u_sim, v_sim, firms = run_simulation(params, time_array, gdp_signal, emp_u)
    n = min(len(u_sim), len(emp_u), len(dates))
    t = dates[:n]

    du_emp = np.diff(emp_u[:n])
    dv_emp = np.diff(emp_v[:n])
    du_sim = np.diff(u_sim[:n])
    dv_sim = np.diff(v_sim[:n])

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    loc = "boundary" if at_boundary else "interior"
    param_str = "  ".join(f"{k}={params[k]:.3g}" for k in PARAM_NAMES)
    fig.suptitle(
        f"Basin {basin_id} [{loc}]  loss={loss:.5f}\n{param_str}",
        fontsize=18, fontweight="bold",
    )

    # ── left column: 3 time-series panels with shared x-axis ──────────────────
    gs_left  = gridspec.GridSpec(3, 1, figure=fig,
                                  left=0.06, right=0.50,
                                  top=0.90, bottom=0.06,
                                  hspace=0.08)
    ax_u   = fig.add_subplot(gs_left[0])
    ax_v   = fig.add_subplot(gs_left[1], sharex=ax_u)
    ax_gdp = fig.add_subplot(gs_left[2], sharex=ax_u)
    plt.setp(ax_u.get_xticklabels(),   visible=False)
    plt.setp(ax_v.get_xticklabels(),   visible=False)

    _plot_timeseries(ax_u,   t, emp_u, u_sim, "u (%)",   "tab:blue",   n)
    _plot_timeseries(ax_v,   t, emp_v, v_sim, "v (%)",   "tab:orange", n)
    _plot_gdp(ax_gdp, t, gdp_signal, n, log_gdp=log_gdp)

    # ── right column: Beveridge + combined Δ dist + firm/corr ─────────────────
    gs_right = gridspec.GridSpec(3, 1, figure=fig,
                                  left=0.55, right=0.98,
                                  top=0.90, bottom=0.06,
                                  hspace=0.45)
    ax_bc   = fig.add_subplot(gs_right[0])

    # Δu and Δv side-by-side, each with own x-axis
    gs_diff = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_right[1], wspace=0.30
    )
    ax_du = fig.add_subplot(gs_diff[0])
    ax_dv = fig.add_subplot(gs_diff[1])

    _plot_beveridge(ax_bc, emp_u, emp_v, u_sim, v_sim, n)
    _plot_diff_dist_pair(ax_du, du_emp, du_sim, "Δu (pp)", "tab:blue")
    _plot_diff_dist_pair(ax_dv, dv_emp, dv_sim, "Δv (pp)", "tab:orange")

    # Bottom-right: firm employment (top half) + GDP correlations (bottom half)
    gs_firm_corr = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_right[2], hspace=0.55
    )
    ax_firm = fig.add_subplot(gs_firm_corr[0])
    ax_corr = fig.add_subplot(gs_firm_corr[1])

    _plot_firm_employment(ax_firm, firms, gdp_signal, t, n)
    _plot_gdp_correlations(ax_corr, u_sim, v_sim, gdp_signal, n)

    return fig


# ── Per-objective driver ───────────────────────────────────────────────────────

def plot_objective(objective, basins_df, emp_u, emp_v, gdp_signal,
                   time_array, dates, out_dir, log_gdp=None):
    for _, row in basins_df.iterrows():
        params = {p: row[p] for p in PARAM_NAMES}
        basin_id = int(row["basin_id"])
        loss = float(row["loss"])
        at_boundary = bool(row.get("at_boundary", True))

        fig = plot_basin(params, basin_id, loss, at_boundary,
                         emp_u, emp_v, gdp_signal, time_array, dates,
                         log_gdp=log_gdp)

        path = os.path.join(out_dir, f"fit_{objective}_basin{basin_id}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fit_plots] {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main(fit_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    emp_u, emp_v, gdp_signal, time_array, dates, log_gdp = load_empirical()
    print(f"[fit_plots] {len(emp_u)} steps  {dates[0].date()} – {dates[-1].date()}")
    print(f"[fit_plots] Reading from {fit_dir}")

    for objective in OBJECTIVES:
        csv_path = os.path.join(fit_dir, f"basins_{objective}.csv")
        if not os.path.exists(csv_path):
            print(f"[fit_plots] skipping {objective} (no CSV)")
            continue
        basins_df = pd.read_csv(csv_path)
        print(f"[fit_plots] {objective}: {len(basins_df)} basins")
        plot_objective(objective, basins_df, emp_u, emp_v,
                       gdp_signal, time_array, dates, out_dir,
                       log_gdp=log_gdp)

    print(f"[fit_plots] Done → {out_dir}")


if __name__ == "__main__":
    fit_dir = sys.argv[1] if len(sys.argv) > 1 else "output/fit_weighted"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else fit_dir.replace("output/", "output/plots_")
    main(fit_dir, out_dir)
