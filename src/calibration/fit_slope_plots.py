"""
Plot slope-fit results: time series comparison for all refined basins.

For each objective (slope_u, slope_v, slope_combined):
  - One figure with n_basins rows × 3 columns:
      col 0: unemployment time series (empirical vs simulated)
      col 1: vacancy time series
      col 2: Beveridge curve (fit period vs out-of-sample)

Usage
-----
    python src/calibration/fit_slope_plots.py
    python src/calibration/fit_slope_plots.py output/fit_slope
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
except ImportError:
    from simulate import simulate_market
    from empirical import load_empirical_data

OBJECTIVES = ["slope_u", "slope_v", "slope_combined"]
PARAM_NAMES = ["K", "s", "c_exponent", "zero_fraction", "firing_threshold", "idio_std"]

FIT_END = pd.Timestamp("2020-01-01")


def load_empirical():
    df = load_empirical_data().dropna(subset=["u_obs", "v_obs", "G"])
    return (
        df["u_obs"].values.astype(float),
        df["v_obs"].values.astype(float),
        df["G"].values.astype(float),
        np.arange(len(df)),
        df.index,
    )


def run_simulation(params, time_array, gdp_signal):
    return simulate_market(
        time_array=time_array,
        gdp_signal=gdp_signal,
        n_firms=250,
        c_max=1.0,
        burn_in_steps=1000,
        seed=42,
        K=params["K"],
        s=params["s"],
        c_exponent=params["c_exponent"],
        zero_fraction=params["zero_fraction"],
        firing_threshold=params["firing_threshold"],
        idio_std=params["idio_std"],
    )


def plot_objective(objective, basins_df, emp_u, emp_v, gdp_signal,
                   time_array, dates, out_dir):
    n_basins = len(basins_df)
    fig, axes = plt.subplots(n_basins, 3, figsize=(18, 3.5 * n_basins), squeeze=False)
    fig.suptitle(f"Slope fit — Objective: {objective}", fontsize=13, fontweight="bold")

    for row_idx, (_, row) in enumerate(basins_df.iterrows()):
        params = {p: row[p] for p in PARAM_NAMES}
        basin_id = int(row["basin_id"])
        loss = row["loss"]
        loc = "boundary" if row.get("at_boundary", True) else "interior"

        u_sim, v_sim = run_simulation(params, time_array, gdp_signal)
        n = min(len(u_sim), len(emp_u), len(dates))
        t = dates[:n]

        param_str = "  ".join(f"{k}={v:.3g}" for k, v in params.items())
        title = f"Basin {basin_id} [{loc}]  loss={loss:.5f}\n{param_str}"

        in_fit = t < FIT_END

        # u time series
        ax = axes[row_idx, 0]
        ax.plot(t, emp_u[:n] * 100, color="black", lw=1.5, label="Empirical")
        ax.plot(t, u_sim[:n] * 100, color="tab:blue", lw=1.2, ls="--", label="Simulated")
        ax.axvline(FIT_END, color="gray", lw=1.0, ls=":", label="Fit end")
        ax.set_ylabel("u (%)")
        ax.set_title(title, fontsize=7)
        ax.legend(fontsize=7)

        # v time series
        ax = axes[row_idx, 1]
        ax.plot(t, emp_v[:n] * 100, color="black", lw=1.5, label="Empirical")
        ax.plot(t, v_sim[:n] * 100, color="tab:orange", lw=1.2, ls="--", label="Simulated")
        ax.axvline(FIT_END, color="gray", lw=1.0, ls=":", label="Fit end")
        ax.set_ylabel("v (%)")
        ax.set_title(title, fontsize=7)
        ax.legend(fontsize=7)

        # Beveridge curve — split fit vs out-of-sample
        ax = axes[row_idx, 2]
        ax.scatter(emp_u[:n][in_fit[:n]]  * 100, emp_v[:n][in_fit[:n]]  * 100,
                   s=8, alpha=0.6, color="black",   label="Emp (fit)",  zorder=3)
        ax.scatter(emp_u[:n][~in_fit[:n]] * 100, emp_v[:n][~in_fit[:n]] * 100,
                   s=8, alpha=0.4, color="dimgray", label="Emp (OOS)",  zorder=3)
        ax.scatter(u_sim[:n] * 100, v_sim[:n] * 100, s=8, alpha=0.4,
                   color="tab:purple", label="Simulated")
        emp_r = float(np.corrcoef(emp_u[:n], emp_v[:n])[0, 1])
        sim_r = float(np.corrcoef(u_sim[:n], v_sim[:n])[0, 1])
        ax.set_xlabel("u (%)")
        ax.set_ylabel("v (%)")
        ax.set_title(f"Beveridge  emp r={emp_r:.2f}  sim r={sim_r:.2f}", fontsize=8)
        ax.legend(fontsize=7)

    for col in range(3):
        axes[-1, col].set_xlabel("Date" if col < 2 else "u (%)")

    fig.tight_layout()
    path = os.path.join(out_dir, f"fit_{objective}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fit_slope_plots] {path}")


def main(fit_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    emp_u, emp_v, gdp_signal, time_array, dates = load_empirical()
    print(f"[fit_slope_plots] {len(emp_u)} steps  {dates[0].date()} – {dates[-1].date()}")
    print(f"[fit_slope_plots] Reading from {fit_dir}")

    for objective in OBJECTIVES:
        csv_path = os.path.join(fit_dir, f"basins_{objective}.csv")
        if not os.path.exists(csv_path):
            print(f"[fit_slope_plots] skipping {objective} (no CSV)")
            continue
        basins_df = pd.read_csv(csv_path)
        print(f"[fit_slope_plots] {objective}: {len(basins_df)} basins")
        plot_objective(objective, basins_df, emp_u, emp_v,
                       gdp_signal, time_array, dates, out_dir)

    print(f"[fit_slope_plots] Done → {out_dir}")


if __name__ == "__main__":
    fit_dir = sys.argv[1] if len(sys.argv) > 1 else "output/fit_slope"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else fit_dir.replace("output/", "output/plots_")
    main(fit_dir, out_dir)
