"""
Plot SMM fit results: time series comparison and moment diagnostic.

For each refined basin:
  - Left figure  : u and v time series (empirical vs simulated, averaged over seeds)
  - Right figure : bar chart comparing all 11 empirical vs simulated moments

Usage
-----
    python src/calibration/fit_smm_plots.py
    python src/calibration/fit_smm_plots.py output/fit_smm
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from calibration.simulate import simulate_market
    from calibration.empirical import load_empirical_data
    from calibration.fit_smm import compute_moments, MOMENT_NAMES, MOMENT_WEIGHTS
except ImportError:
    from simulate import simulate_market
    from empirical import load_empirical_data
    from fit_smm import compute_moments, MOMENT_NAMES, MOMENT_WEIGHTS

PARAM_NAMES = ["K", "s", "c_exponent", "zero_fraction", "firing_threshold", "idio_std"]
N_SEEDS = 5
FIT_END = pd.Timestamp("2020-01-01")


def load_empirical():
    # Full series through Sep 2025; fit was done on pre-2020 data
    df = load_empirical_data().dropna(subset=["u_obs", "v_obs", "G"])
    return (
        df["u_obs"].values.astype(float),
        df["v_obs"].values.astype(float),
        df["G"].values.astype(float),
        np.arange(len(df)),
        df.index,
    )


def run_averaged(params, time_array, gdp_signal, n_seeds=N_SEEDS):
    """Run n_seeds simulations and return the mean u and v trajectories."""
    us, vs = [], []
    for seed in range(42, 42 + n_seeds):
        u, v = simulate_market(
            time_array=time_array,
            gdp_signal=gdp_signal,
            n_firms=250,
            c_max=1.0,
            burn_in_steps=500,
            seed=seed,
            K=params["K"],
            s=params["s"],
            c_exponent=params["c_exponent"],
            zero_fraction=params["zero_fraction"],
            firing_threshold=params["firing_threshold"],
            idio_std=params["idio_std"],
        )
        us.append(u)
        vs.append(v)
    return np.mean(us, axis=0), np.mean(vs, axis=0)


def plot_basin_timeseries(basin_rows, emp_u, emp_v, gdp_signal, time_array, dates, out_dir):
    """One figure with all basins: 3 columns (u, v, Beveridge) × n_basins rows."""
    n = len(basin_rows)
    fig, axes = plt.subplots(n, 3, figsize=(18, 3.5 * n), squeeze=False)
    fig.suptitle("SMM Fit — Time Series & Beveridge Curve", fontsize=13, fontweight="bold")

    for row_idx, row in enumerate(basin_rows):
        params = {p: row[p] for p in PARAM_NAMES}
        basin_id = int(row["basin_id"])
        loss = row["smm_loss"]
        loc = "boundary" if row.get("at_boundary", True) else "interior"

        u_sim, v_sim = run_averaged(params, time_array, gdp_signal)
        nt = min(len(u_sim), len(emp_u), len(dates))
        t = dates[:nt]

        param_str = "  ".join(f"{k}={v:.3g}" for k, v in params.items())
        title = f"Basin {basin_id} [{loc}]  SMM loss={loss:.4f}\n{param_str}"

        in_fit = t < FIT_END

        # u time series
        ax = axes[row_idx, 0]
        ax.plot(t, emp_u[:nt] * 100, color="black", lw=1.5, label="Empirical")
        ax.plot(t, u_sim[:nt] * 100, color="tab:blue", lw=1.2, ls="--",
                label=f"Sim (mean {N_SEEDS} seeds)")
        ax.axvline(FIT_END, color="gray", lw=1.0, ls=":", label="Fit end")
        ax.set_ylabel("u (%)")
        ax.set_title(title, fontsize=7)
        ax.legend(fontsize=7)

        # v time series
        ax = axes[row_idx, 1]
        ax.plot(t, emp_v[:nt] * 100, color="black", lw=1.5, label="Empirical")
        ax.plot(t, v_sim[:nt] * 100, color="tab:orange", lw=1.2, ls="--",
                label=f"Sim (mean {N_SEEDS} seeds)")
        ax.axvline(FIT_END, color="gray", lw=1.0, ls=":", label="Fit end")
        ax.set_ylabel("v (%)")
        ax.set_title(title, fontsize=7)
        ax.legend(fontsize=7)

        # Beveridge curve — split fit vs out-of-sample
        ax = axes[row_idx, 2]
        ax.scatter(emp_u[:nt][in_fit[:nt]]  * 100, emp_v[:nt][in_fit[:nt]]  * 100,
                   s=8, alpha=0.6, color="black",   label="Emp (fit)",  zorder=3)
        ax.scatter(emp_u[:nt][~in_fit[:nt]] * 100, emp_v[:nt][~in_fit[:nt]] * 100,
                   s=8, alpha=0.4, color="dimgray", label="Emp (OOS)",  zorder=3)
        ax.scatter(u_sim[:nt] * 100, v_sim[:nt] * 100, s=8, alpha=0.4,
                   color="tab:purple", label="Simulated")
        emp_r = float(np.corrcoef(emp_u[:nt], emp_v[:nt])[0, 1])
        sim_r = float(np.corrcoef(u_sim[:nt], v_sim[:nt])[0, 1])
        ax.set_xlabel("u (%)")
        ax.set_ylabel("v (%)")
        ax.set_title(f"Beveridge  emp r={emp_r:.2f}  sim r={sim_r:.2f}", fontsize=8)
        ax.legend(fontsize=7)

    for col in range(3):
        axes[-1, col].set_xlabel("Date" if col < 2 else "u (%)")

    fig.tight_layout()
    path = os.path.join(out_dir, "smm_timeseries.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [smm_plots] {path}")


def plot_moment_comparison(basin_rows, emp_moments, gdp_signal, time_array, out_dir):
    """
    One figure with n_basins columns.  Each column is a grouped bar chart
    comparing empirical vs simulated moments for that basin.
    """
    n = len(basin_rows)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 7), squeeze=False)
    fig.suptitle("SMM Moment Comparison: Empirical vs Simulated", fontsize=12, fontweight="bold")

    emp_vals = np.array([emp_moments[k] for k in MOMENT_NAMES])

    for col_idx, row in enumerate(basin_rows):
        params = {p: row[p] for p in PARAM_NAMES}
        basin_id = int(row["basin_id"])

        # Average moments across seeds
        all_moms = []
        for seed in range(42, 42 + N_SEEDS):
            u, v = simulate_market(
                time_array=time_array,
                gdp_signal=gdp_signal,
                n_firms=250,
                c_max=1.0,
                burn_in_steps=500,
                seed=seed,
                **{k: v for k, v in params.items() if k != "c_max"},
            )
            all_moms.append([compute_moments(u, v)[k] for k in MOMENT_NAMES])
        sim_vals = np.mean(all_moms, axis=0)

        ax = axes[0, col_idx]
        x = np.arange(len(MOMENT_NAMES))
        w = 0.35

        # Normalise both to empirical scale for visual comparability
        scales = np.abs(emp_vals)
        scales[scales < 1e-6] = 1.0
        emp_norm = emp_vals / scales
        sim_norm = sim_vals / scales

        bars_e = ax.bar(x - w / 2, emp_norm, w, label="Empirical", color="steelblue", alpha=0.8)
        bars_s = ax.bar(x + w / 2, sim_norm, w, label="Simulated", color="tomato", alpha=0.8)

        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(MOMENT_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Value / |empirical|")
        ax.set_title(f"Basin {basin_id}\nloss={row['smm_loss']:.4f}", fontsize=8)
        ax.legend(fontsize=7)

        # Annotate relative error
        for xi, (ev, sv) in enumerate(zip(emp_vals, sim_vals)):
            rel_err = (sv - ev) / max(abs(ev), 1e-6)
            color = "darkgreen" if abs(rel_err) < 0.1 else ("darkorange" if abs(rel_err) < 0.3 else "red")
            ax.text(xi, max(emp_norm[xi], sim_norm[xi]) + 0.05,
                    f"{rel_err:+.1%}", ha="center", fontsize=5.5, color=color)

    fig.tight_layout()
    path = os.path.join(out_dir, "smm_moments.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [smm_plots] {path}")


def main(smm_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    basins_csv = os.path.join(smm_dir, "smm_basins.csv")
    moments_csv = os.path.join(smm_dir, "smm_emp_moments.csv")
    if not os.path.exists(basins_csv):
        print(f"[smm_plots] ERROR: {basins_csv} not found. Run fit_smm.py first.")
        return

    basins_df = pd.read_csv(basins_csv)
    basin_rows = [row for _, row in basins_df.iterrows()]
    print(f"[smm_plots] {len(basin_rows)} basins from {basins_csv}")

    emp_u, emp_v, gdp_signal, time_array, dates = load_empirical()
    print(f"[smm_plots] {len(emp_u)} steps  {dates[0].date()} – {dates[-1].date()}")

    if os.path.exists(moments_csv):
        emp_moments = pd.read_csv(moments_csv).iloc[0].to_dict()
    else:
        emp_moments = compute_moments(emp_u, emp_v)

    print("[smm_plots] Generating time series plot...")
    plot_basin_timeseries(basin_rows, emp_u, emp_v, gdp_signal, time_array, dates, out_dir)

    print("[smm_plots] Generating moment comparison plot...")
    plot_moment_comparison(basin_rows, emp_moments, gdp_signal, time_array, out_dir)

    print(f"[smm_plots] Done → {out_dir}")


if __name__ == "__main__":
    smm_dir = sys.argv[1] if len(sys.argv) > 1 else "output/fit_smm"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/plots_fit_smm"
    main(smm_dir, out_dir)
