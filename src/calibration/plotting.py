"""
PHASE 5: Plotting fitted trajectories.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_fitted_trajectories(
    empirical_data,
    calib_result,
    gdp_signal,
    output_dir="output/calibration",
):
    """
    Plot empirical vs simulated u, v, Beveridge curve.

    Parameters
    ----------
    empirical_data : pd.DataFrame
        With columns: u_obs, v_obs, G
    calib_result : dict
        From calibrate(): sim_u, sim_v, params_optimal, loss_components
    gdp_signal : np.ndarray
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)

    emp_u = empirical_data["u_obs"].values
    emp_v = empirical_data["v_obs"].values
    sim_u = calib_result["sim_u"]
    sim_v = calib_result["sim_v"]
    params = calib_result["params_optimal"]
    loss_comp = calib_result["loss_components"]

    # Align lengths
    min_len = min(len(emp_u), len(sim_u))
    emp_u_trim = emp_u[:min_len]
    emp_v_trim = emp_v[:min_len]
    sim_u_trim = sim_u[:min_len]
    sim_v_trim = sim_v[:min_len]
    gdp_trim = gdp_signal[:min_len]

    t = np.arange(min_len)

    fig, axs = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)

    # Top left: unemployment rate
    axs[0, 0].plot(t, emp_u_trim, "o-", label="Empirical", alpha=0.7, markersize=3)
    axs[0, 0].plot(t, sim_u_trim, "s-", label="Simulated", alpha=0.7, markersize=3)
    axs[0, 0].set_title("Unemployment Rate u(t)")
    axs[0, 0].set_ylabel("u")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Top middle: vacancy rate
    axs[0, 1].plot(t, emp_v_trim, "o-", label="Empirical", alpha=0.7, markersize=3)
    axs[0, 1].plot(t, sim_v_trim, "s-", label="Simulated", alpha=0.7, markersize=3)
    axs[0, 1].set_title("Vacancy Rate v/(v+e)")
    axs[0, 1].set_ylabel("v")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Top right: GDP signal
    axs[0, 2].bar(t, gdp_trim, alpha=0.6, color="green")
    axs[0, 2].set_title("GDP Signal G(t)")
    axs[0, 2].set_ylabel("G")
    axs[0, 2].axhline(0, color="k", linestyle="-", linewidth=0.5)
    axs[0, 2].grid(True, alpha=0.3)

    # Middle left: Beveridge curve (empirical)
    axs[1, 0].scatter(emp_u_trim, emp_v_trim, alpha=0.5, s=10, label="Empirical")
    axs[1, 0].set_title("Beveridge Curve (Empirical)")
    axs[1, 0].set_xlabel("u")
    axs[1, 0].set_ylabel("v")
    axs[1, 0].grid(True, alpha=0.3)

    # Middle middle: Beveridge curve (simulated)
    axs[1, 1].scatter(sim_u_trim, sim_v_trim, alpha=0.5, s=10, color="orange", label="Simulated")
    axs[1, 1].set_title("Beveridge Curve (Simulated)")
    axs[1, 1].set_xlabel("u")
    axs[1, 1].set_ylabel("v")
    axs[1, 1].grid(True, alpha=0.3)

    # Middle right: both Beveridge curves
    axs[1, 2].scatter(emp_u_trim, emp_v_trim, alpha=0.4, s=10, label="Empirical")
    axs[1, 2].scatter(sim_u_trim, sim_v_trim, alpha=0.4, s=10, color="orange", label="Simulated")
    axs[1, 2].set_title("Beveridge Curve (Overlay)")
    axs[1, 2].set_xlabel("u")
    axs[1, 2].set_ylabel("v")
    axs[1, 2].legend()
    axs[1, 2].grid(True, alpha=0.3)

    # Bottom left: residuals u
    axs[2, 0].plot(t, emp_u_trim - sim_u_trim, "o-", markersize=3, alpha=0.7)
    axs[2, 0].set_title("Residuals: u_emp - u_sim")
    axs[2, 0].set_xlabel("Time")
    axs[2, 0].set_ylabel("Residual")
    axs[2, 0].axhline(0, color="k", linestyle="--", linewidth=0.5)
    axs[2, 0].grid(True, alpha=0.3)

    # Bottom middle: residuals v
    axs[2, 1].plot(t, emp_v_trim - sim_v_trim, "o-", markersize=3, alpha=0.7)
    axs[2, 1].set_title("Residuals: v_emp - v_sim")
    axs[2, 1].set_xlabel("Time")
    axs[2, 1].set_ylabel("Residual")
    axs[2, 1].axhline(0, color="k", linestyle="--", linewidth=0.5)
    axs[2, 1].grid(True, alpha=0.3)

    # Bottom right: metrics table (text)
    axs[2, 2].axis("off")
    metrics_text = f"""
Calibration Results

Parameters:
  s = {params["s"]:.4f}
  c_max = {params["c_max"]:.4f}
  c_exponent = {params["c_exponent"]:.4f}
  zero_fraction = {params["zero_fraction"]:.4f}

Loss Components:
  MSE_u = {loss_comp["mse_u"]:.6f}
  MSE_v = {loss_comp["mse_v"]:.6f}
  Loss_level = {loss_comp["loss_level"]:.6f}
  Loss_corr = {loss_comp["loss_corr"]:.6f}
  Loss_bev = {loss_comp["loss_bev"]:.6f}

Correlations:
  Sim u-v: {loss_comp["corr_sim"]:.4f}
  Emp u-v: {loss_comp["corr_emp"]:.4f}

Valid points: {loss_comp["n_valid"]}
    """
    axs[2, 2].text(0.1, 0.5, metrics_text, fontsize=9, family="monospace")

    fig.suptitle("Fitted Trajectories: Empirical vs Simulated", fontsize=14, y=0.995)

    out_path = os.path.join(output_dir, "fitted_trajectories.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved fitted trajectories to {out_path}")

    # Save summary metrics
    metrics_path = os.path.join(output_dir, "calibration_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("CALIBRATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("OPTIMAL PARAMETERS\n")
        f.write(f"  s (separation rate):     {params['s']:.6f}\n")
        f.write(f"  c_max:                   {params['c_max']:.6f}\n")
        f.write(f"  c_exponent:              {params['c_exponent']:.6f}\n")
        f.write(f"  zero_fraction:           {params['zero_fraction']:.6f}\n\n")
        f.write("LOSS COMPONENTS\n")
        f.write(f"  MSE (u):                 {loss_comp['mse_u']:.8f}\n")
        f.write(f"  MSE (v):                 {loss_comp['mse_v']:.8f}\n")
        f.write(f"  Loss (level):            {loss_comp['loss_level']:.8f}\n")
        f.write(f"  Loss (correlation):      {loss_comp['loss_corr']:.8f}\n")
        f.write(f"  Loss (Beveridge):        {loss_comp['loss_bev']:.8f}\n")
        f.write(f"  Total Loss:              {calib_result['loss_optimal']:.8f}\n\n")
        f.write("GOODNESS OF FIT\n")
        f.write(f"  Simulated u-v correlation:  {loss_comp['corr_sim']:.4f}\n")
        f.write(f"  Empirical u-v correlation:  {loss_comp['corr_emp']:.4f}\n")
        f.write(f"  Valid points:               {loss_comp['n_valid']}\n")

    print(f"Saved metrics to {metrics_path}")

    return out_path
