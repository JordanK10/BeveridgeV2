"""
PHASE 6: Main calibration entry point.

Orchestrates the full pipeline: empirical data → simulation → optimization → plots.
"""

import os
import sys

import numpy as np

from .empirical import load_empirical_data
from .optimize import calibrate
from .plotting import plot_fitted_trajectories


def main(
    output_dir="output/calibration",
    K=5.4,
    n_firms=250,
    method="L-BFGS-B",
    verbose=True,
):
    """
    Full calibration pipeline.

    Parameters
    ----------
    output_dir : str
        Where to save results
    K : float
        Fixed matching constant
    n_firms : int
        Number of firms in market
    method : str
        Optimization method: "L-BFGS-B" or "differential_evolution"
    verbose : bool
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("CALIBRATION PIPELINE")
    print("=" * 70)

    # Phase 1: Load empirical data
    print("\n[1/5] Loading empirical data...")
    empirical_path = os.path.join(output_dir, "empirical_data.csv")
    empirical_data = load_empirical_data(output_path=empirical_path)
    empirical_data = empirical_data.dropna(subset=["u_obs", "v_obs", "G"])

    time_array = np.arange(len(empirical_data))
    gdp_signal = empirical_data["G"].values.astype(float)

    # Phase 2–4: Optimize
    print("\n[2/5] Initializing simulator...")
    print(f"  Time series length: {len(time_array)}")
    print(f"  GDP signal range: [{gdp_signal.min():.4f}, {gdp_signal.max():.4f}]")
    print(f"  Empirical u range: [{empirical_data['u_obs'].min():.4f}, {empirical_data['u_obs'].max():.4f}]")
    print(f"  Empirical v range: [{empirical_data['v_obs'].min():.4f}, {empirical_data['v_obs'].max():.4f}]")

    print(f"\n[3/5] Running calibration ({method})...")
    result = calibrate(
        empirical_data,
        time_array,
        gdp_signal,
        K=K,
        n_firms=n_firms,
        method=method,
        verbose=verbose,
    )

    params_opt = result["params_optimal"]
    loss_opt = result["loss_optimal"]
    loss_comp = result["loss_components"]

    print(f"\n  Optimal parameters:")
    print(f"    s = {params_opt['s']:.6f}")
    print(f"    c_max = {params_opt['c_max']:.6f}")
    print(f"    c_exponent = {params_opt['c_exponent']:.6f}")
    print(f"    zero_fraction = {params_opt['zero_fraction']:.6f}")
    print(f"    firing_threshold = {params_opt.get('firing_threshold', 0.0):.6f}")
    print(f"    hiring_efficiency = {params_opt.get('hiring_efficiency', 1.0):.6f}")
    print(f"\n  Loss components:")
    print(f"    MSE_u = {loss_comp['mse_u']:.8f}")
    print(f"    MSE_v = {loss_comp['mse_v']:.8f}")
    print(f"    Total loss = {loss_opt:.8f}")
    print(f"\n  Correlations:")
    print(f"    Simulated u-v: {loss_comp['corr_sim']:.4f}")
    print(f"    Empirical u-v: {loss_comp['corr_emp']:.4f}")

    # Phase 5: Plot
    print(f"\n[4/5] Generating plots...")
    plot_path = plot_fitted_trajectories(
        empirical_data,
        result,
        gdp_signal,
        output_dir=output_dir,
    )

    # Save parameters
    print(f"\n[5/5] Saving results...")
    params_path = os.path.join(output_dir, "calibration_params.txt")
    with open(params_path, "w") as f:
        f.write("CALIBRATED PARAMETERS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"K (matching constant, fixed) = {K}\n")
        f.write(f"n_firms = {n_firms}\n")
        f.write(f"s (separation rate) = {params_opt['s']:.8f}\n")
        f.write(f"c_max = {params_opt['c_max']:.8f}\n")
        f.write(f"c_exponent = {params_opt['c_exponent']:.8f}\n")
        f.write(f"zero_fraction = {params_opt['zero_fraction']:.8f}\n")
        f.write(f"firing_threshold = {params_opt.get('firing_threshold', 0.0):.8f}\n")
        f.write(f"hiring_efficiency = {params_opt.get('hiring_efficiency', 1.0):.8f}\n")

    print(f"\nCalibration complete!")
    print(f"Results saved to {output_dir}/")
    print(f"  - empirical_data.csv")
    print(f"  - fitted_trajectories.pdf")
    print(f"  - calibration_metrics.txt")
    print(f"  - calibration_params.txt")

    return result


if __name__ == "__main__":
    main()
