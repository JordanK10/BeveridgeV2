"""
Diagnostic plots: empirical data + baseline simulation comparison.

Shows what we have before running the full calibration.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

try:
    from .simulate import simulate_market
except ImportError:
    from simulate import simulate_market


def plot_diagnostics(output_dir="output/calibration_diagnostics"):
    """
    Generate diagnostic plots: empirical data + baseline simulation.

    Parameters
    ----------
    output_dir : str
        Where to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading empirical data (JOLTS + BLS + FRED)...")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from calibration.empirical import load_empirical_data

    empirical_data = load_empirical_data()
    empirical_data = empirical_data.dropna(subset=["u_obs", "v_obs", "G"])

    emp_u = empirical_data["u_obs"].values.astype(float)
    emp_v = empirical_data["v_obs"].values.astype(float)
    gdp_signal = empirical_data["G"].values.astype(float)
    dates = empirical_data.index  # DatetimeIndex for x-axis
    time_array = np.arange(len(emp_u))

    from scipy.stats import skew as scipy_skew
    print(f"  Observations: {len(emp_u)}")
    print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"  u range: [{emp_u.min():.4f}, {emp_u.max():.4f}]")
    print(f"  v (JOLTS JOR) range: [{emp_v.min():.4f}, {emp_v.max():.4f}]")
    print(f"  G range: [{gdp_signal.min():.4f}, {gdp_signal.max():.4f}]")
    print(f"  Correlation (u, v): {np.corrcoef(emp_u, emp_v)[0, 1]:.4f}")
    print(f"  skew(u): {scipy_skew(emp_u):.4f},  skew(v): {scipy_skew(emp_v):.4f}")
    print(f"  skew(Δu): {scipy_skew(np.diff(emp_u)):.4f},  skew(Δv): {scipy_skew(np.diff(emp_v)):.4f}")

    # Run baseline simulation with default parameters
    print("\nRunning baseline simulation...")
    sim_u, sim_v = simulate_market(
        time_array,
        gdp_signal,
        n_firms=250,
        c_exponent=1.2,
        c_max=0.40,
        zero_fraction=0.0625,
        K=2.31,
        s=0.0043,
        target_u=0.05,
        seed=42,
        firing_threshold=0.10,
    )

    from scipy.stats import skew as scipy_skew
    print(f"Simulated data:")
    print(f"  u range: [{sim_u.min():.4f}, {sim_u.max():.4f}]")
    print(f"  v range: [{sim_v.min():.4f}, {sim_v.max():.4f}]")
    print(f"  Correlation (u, v): {np.corrcoef(sim_u, sim_v)[0, 1]:.4f}")
    print(f"  skew(u): {scipy_skew(sim_u):.4f},  skew(v): {scipy_skew(sim_v):.4f}")
    print(f"  skew(Δu): {scipy_skew(np.diff(sim_u)):.4f},  skew(Δv): {scipy_skew(np.diff(sim_v)):.4f}")

    # ── Helper: format x-axis as calendar years ────────────────────────────
    def fmt_years(ax):
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.tick_params(axis="x", which="major", labelsize=8)

    # ── 6-panel diagnostic figure ──────────────────────────────────────────
    fig, axs = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

    # Row 0: Empirical series
    axs[0, 0].plot(dates, emp_u, linewidth=1, markersize=2, alpha=0.8, color="C0")
    axs[0, 0].axhline(np.mean(emp_u), color="C0", linestyle="--", alpha=0.5,
                      label=f"Mean: {np.mean(emp_u):.3f}")
    axs[0, 0].set_title("Empirical: Unemployment Rate u(t)", fontweight="bold")
    axs[0, 0].set_ylabel("u")
    axs[0, 0].legend(fontsize=9)
    axs[0, 0].grid(True, alpha=0.3)
    fmt_years(axs[0, 0])

    axs[0, 1].plot(dates, emp_v, linewidth=1, markersize=2, alpha=0.8, color="C1")
    axs[0, 1].axhline(np.mean(emp_v), color="C1", linestyle="--", alpha=0.5,
                      label=f"Mean: {np.mean(emp_v):.3f}")
    axs[0, 1].set_title("Empirical: Vacancy Rate v(t)", fontweight="bold")
    axs[0, 1].set_ylabel("v")
    axs[0, 1].legend(fontsize=9)
    axs[0, 1].grid(True, alpha=0.3)
    fmt_years(axs[0, 1])

    axs[0, 2].bar(dates, gdp_signal, width=25, alpha=0.7, color=["C3" if g < 0 else "C2" for g in gdp_signal])
    axs[0, 2].axhline(0, color="k", linewidth=0.8)
    axs[0, 2].set_title("GDP Signal G(t) — quarterly log-growth", fontweight="bold")
    axs[0, 2].set_ylabel("G")
    axs[0, 2].grid(True, alpha=0.3)
    fmt_years(axs[0, 2])

    # Row 1: Simulated series
    axs[1, 0].plot(dates, sim_u, linewidth=1, markersize=2, alpha=0.8, color="C3")
    axs[1, 0].axhline(np.mean(sim_u), color="C3", linestyle="--", alpha=0.5,
                      label=f"Mean: {np.mean(sim_u):.3f}")
    axs[1, 0].set_title("Simulated (Baseline): Unemployment Rate u(t)", fontweight="bold")
    axs[1, 0].set_ylabel("u")
    axs[1, 0].set_xlabel("Year")
    axs[1, 0].legend(fontsize=9)
    axs[1, 0].grid(True, alpha=0.3)
    fmt_years(axs[1, 0])

    axs[1, 1].plot(dates, sim_v, linewidth=1, markersize=2, alpha=0.8, color="C4")
    axs[1, 1].axhline(np.mean(sim_v), color="C4", linestyle="--", alpha=0.5,
                      label=f"Mean: {np.mean(sim_v):.3f}")
    axs[1, 1].set_title("Simulated (Baseline): Vacancy Rate v(t)", fontweight="bold")
    axs[1, 1].set_ylabel("v")
    axs[1, 1].set_xlabel("Year")
    axs[1, 1].legend(fontsize=9)
    axs[1, 1].grid(True, alpha=0.3)
    fmt_years(axs[1, 1])

    # Beveridge curves overlaid
    axs[1, 2].scatter(emp_u, emp_v, alpha=0.5, s=15, color="C2", label="Empirical")
    axs[1, 2].scatter(sim_u, sim_v, alpha=0.5, s=15, color="C5", label="Simulated")
    corr_emp = np.corrcoef(emp_u, emp_v)[0, 1]
    corr_sim = np.corrcoef(sim_u, sim_v)[0, 1]
    axs[1, 2].set_title("Beveridge Curve (Overlay)", fontweight="bold")
    axs[1, 2].set_xlabel("u")
    axs[1, 2].set_ylabel("v")
    axs[1, 2].legend(fontsize=9)
    axs[1, 2].grid(True, alpha=0.3)
    axs[1, 2].text(0.05, 0.95,
                   f"Emp corr: {corr_emp:.3f}\nSim corr: {corr_sim:.3f}",
                   transform=axs[1, 2].transAxes, fontsize=9,
                   verticalalignment="top",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Model Diagnostics: Empirical Data vs. Baseline Simulation",
                 fontsize=14, fontweight="bold")
    fig_path = os.path.join(output_dir, "diagnostic_comparison.pdf")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved diagnostic plot to {fig_path}")

    # ── Overlay figure: u, v, G stacked with shared x-axis ────────────────
    fig, axs2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)

    axs2[0].plot(dates, emp_u, linewidth=1.5, alpha=0.8, label="Empirical", color="C0")
    axs2[0].plot(dates, sim_u, linewidth=1.5, alpha=0.8, linestyle="--",
                 label="Simulated (baseline)", color="C3")
    axs2[0].set_ylabel("u")
    axs2[0].set_title("Unemployment Rate u(t)", fontweight="bold")
    axs2[0].legend(fontsize=10)
    axs2[0].grid(True, alpha=0.3)

    axs2[1].plot(dates, emp_v, linewidth=1.5, alpha=0.8, label="Empirical", color="C1")
    axs2[1].plot(dates, sim_v, linewidth=1.5, alpha=0.8, linestyle="--",
                 label="Simulated (baseline)", color="C4")
    axs2[1].set_ylabel("v")
    axs2[1].set_title("Vacancy Rate v(t)", fontweight="bold")
    axs2[1].legend(fontsize=10)
    axs2[1].grid(True, alpha=0.3)

    axs2[2].bar(dates, gdp_signal, width=25, alpha=0.7,
                color=["C3" if g < 0 else "C2" for g in gdp_signal])
    axs2[2].axhline(0, color="k", linewidth=0.8)
    axs2[2].set_ylabel("G")
    axs2[2].set_title("GDP Signal G(t) — quarterly log-growth, normalized", fontweight="bold")
    axs2[2].grid(True, alpha=0.3)
    fmt_years(axs2[2])
    axs2[2].set_xlabel("Year")

    fig.suptitle("Empirical vs. Simulated: u(t), v(t), and GDP Signal",
                 fontsize=13, fontweight="bold")
    fig_path2 = os.path.join(output_dir, "overlay_comparison.pdf")
    fig.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay plot to {fig_path2}")

    # ── Summary statistics ─────────────────────────────────────────────────
    summary_path = os.path.join(output_dir, "diagnostic_summary.txt")
    with open(summary_path, "w") as f:
        f.write("DIAGNOSTIC SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write("EMPIRICAL DATA (JOLTS JOR + BLS CPS + FRED GDP)\n")
        f.write(f"  Time span: {dates[0].date()} to {dates[-1].date()} ({len(empirical_data)} months)\n")
        f.write(f"  u: mean={np.mean(emp_u):.4f}, std={np.std(emp_u):.4f}, range=[{np.min(emp_u):.4f}, {np.max(emp_u):.4f}]\n")
        f.write(f"  v: mean={np.mean(emp_v):.4f}, std={np.std(emp_v):.4f}, range=[{np.min(emp_v):.4f}, {np.max(emp_v):.4f}]\n")
        f.write(f"  u-v correlation: {np.corrcoef(emp_u, emp_v)[0, 1]:.4f}\n")
        f.write(f"  skew(u): {scipy_skew(emp_u):.4f},  skew(v): {scipy_skew(emp_v):.4f}\n")
        f.write(f"  skew(Δu): {scipy_skew(np.diff(emp_u)):.4f},  skew(Δv): {scipy_skew(np.diff(emp_v)):.4f}\n\n")

        f.write("SIMULATED (BASELINE PARAMETERS)\n")
        f.write(f"  Parameters: K=2.31 (t_half=6mo), s=0.0043, c_max=0.15, c_exponent=1.2, zero_fraction=0.0625, firing_threshold=0.10, n_firms=250\n")
        f.write(f"  u: mean={np.mean(sim_u):.4f}, std={np.std(sim_u):.4f}, range=[{np.min(sim_u):.4f}, {np.max(sim_u):.4f}]\n")
        f.write(f"  v: mean={np.mean(sim_v):.4f}, std={np.std(sim_v):.4f}, range=[{np.min(sim_v):.4f}, {np.max(sim_v):.4f}]\n")
        f.write(f"  u-v correlation: {np.corrcoef(sim_u, sim_v)[0, 1]:.4f}\n")
        f.write(f"  skew(u): {scipy_skew(sim_u):.4f},  skew(v): {scipy_skew(sim_v):.4f}\n")
        f.write(f"  skew(Δu): {scipy_skew(np.diff(sim_u)):.4f},  skew(Δv): {scipy_skew(np.diff(sim_v)):.4f}\n\n")

        f.write("GAPS (EMPIRICAL - SIMULATED)\n")
        f.write(f"  Δ(mean u): {np.mean(emp_u) - np.mean(sim_u):+.4f}\n")
        f.write(f"  Δ(mean v): {np.mean(emp_v) - np.mean(sim_v):+.4f}\n")
        f.write(f"  Δ(std u):  {np.std(emp_u) - np.std(sim_u):+.4f}\n")
        f.write(f"  Δ(std v):  {np.std(emp_v) - np.std(sim_v):+.4f}\n")
        f.write(f"  Δ(corr):   {np.corrcoef(emp_u, emp_v)[0, 1] - np.corrcoef(sim_u, sim_v)[0, 1]:+.4f}\n")

    print(f"Saved summary to {summary_path}")

    return {
        "empirical_data": empirical_data,
        "sim_u": sim_u,
        "sim_v": sim_v,
        "diagnostic_plot": fig_path,
        "overlay_plot": fig_path2,
        "summary": summary_path,
    }


if __name__ == "__main__":
    plot_diagnostics()
