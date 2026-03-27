"""Figures for GDP step-shock relaxation (U, V) and power-law diagnostics."""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def fit_power_law_decay_window(
    series,
    tau0=1.0,
    tau_min=1,
    tau_max_exclusive=None,
):
    """
    Fit log|y(τ) - y_end| ≈ a + β log(τ + τ0) over interior τ.

    ``series`` is y(τ) for τ = 0, …, T with y_end = y(T). The last point has
    zero residual and is excluded; τ=0 is excluded when using log(τ+τ0) with
    emphasis on decay shape (plan: τ = 1 … T-1).

    Returns (beta, r_squared) or (np.nan, np.nan) if insufficient valid points.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 3:
        return np.nan, np.nan

    y_end = series[-1]
    y = np.abs(series - y_end)
    tau = np.arange(n, dtype=float)

    if tau_max_exclusive is None:
        tau_max_exclusive = n - 1

    mask = (tau >= tau_min) & (tau < tau_max_exclusive)
    y_fit = y[mask]
    tau_fit = tau[mask]

    positive = y_fit > 0
    if np.sum(positive) < 3:
        return np.nan, np.nan

    y_fit = y_fit[positive]
    tau_fit = tau_fit[positive]

    log_x = np.log(tau_fit + tau0)
    log_y = np.log(y_fit)

    slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)
    return slope, r_value**2


def half_life_time_in_window(series, dt, atol=1e-6):
    """
    Half-life as time to go halfway from the first sample to the last (window endpoint).

    Let ``m = (x(0) + x(T)) / 2``. If ``x(T) > x(0)``, return the smallest ``tau >= 1``
    such that ``x(tau) >= m``; if ``x(T) < x(0)``, return the smallest ``tau`` with
    ``x(tau) <= m``. Time returned is ``tau * dt`` (same units as the simulation clock).

    If ``|x(T)-x(0)| < atol`` or no crossing occurs within the window, returns ``nan``.
    """
    x = np.asarray(series, dtype=float)
    if len(x) < 2:
        return np.nan
    x0, x_end = x[0], x[-1]
    if abs(x_end - x0) < atol:
        return np.nan
    midpoint = 0.5 * (x0 + x_end)
    upward = x_end > x0
    for tau in range(1, len(x)):
        if upward:
            if x[tau] >= midpoint:
                return float(tau * dt)
        else:
            if x[tau] <= midpoint:
                return float(tau * dt)
    return np.nan


def plot_gdp_shock_response_figure(
    g_values,
    tau_axis,
    U_by_g,
    V_by_g,
    beta_U,
    beta_V,
    half_life_U,
    half_life_V,
    r2_U=None,
    r2_V=None,
    output_path=None,
    title="GDP shock response: U, V and power-law exponents",
):
    """
    2×3 layout: time series | β vs g | half-life vs g for U (row 0) and V (row 1).

    ``tau_axis`` is time since shock (length of each series), e.g. τ * DT.
    Half-lives are in the same time units as ``tau_axis``.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    cmap = plt.get_cmap("coolwarm")
    g_arr = np.asarray(g_values, dtype=float)
    gmin, gmax = float(np.min(g_arr)), float(np.max(g_arr))
    span = gmax - gmin if gmax > gmin else 1.0
    norm = plt.Normalize(vmin=gmin - 0.05 * span, vmax=gmax + 0.05 * span)

    for j, g in enumerate(g_values):
        color = cmap(norm(g))
        axs[0, 0].plot((tau_axis), np.log(U_by_g[j]), color=color, label=f"g={g:.2f}", linewidth=1.2, alpha=0.9)
        axs[1, 0].plot((tau_axis), np.log(V_by_g[j]), color=color, label=f"g={g:.2f}", linewidth=1.2, alpha=0.9)

    axs[0, 0].set_title(r"$U(t)$ after shock (aggregate)")
    axs[0, 0].set_xlabel(r"Time since shock ($\tau$)")
    axs[0, 0].set_ylabel("$ln(U)$")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=7, ncol=2, loc="best")

    axs[1, 0].set_title(r"$V(t)$ after shock (aggregate)")
    axs[1, 0].set_xlabel(r"Time since shock ($\tau$)")
    axs[1, 0].set_ylabel("$ln(V)$")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=7, ncol=2, loc="best")

    valid_U = np.isfinite(beta_U)
    valid_V = np.isfinite(beta_V)
    hl_U_arr = np.asarray(half_life_U, dtype=float)
    hl_V_arr = np.asarray(half_life_V, dtype=float)
    valid_hl_U = np.isfinite(hl_U_arr)
    valid_hl_V = np.isfinite(hl_V_arr)



    axs[0, 1].scatter(g_arr[valid_hl_U], hl_U_arr[valid_hl_U], c="C0", s=36, zorder=3, marker="s", alpha=0.85)
    axs[0, 1].set_title(r"Half-life $U$: time to midpoint of $U(0)\to U(T)$")
    axs[0, 1].set_xlabel(r"Post-shock signal $g$")
    axs[0, 1].set_ylabel("Half-life (time units)")
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 1].scatter(g_arr[valid_hl_V], hl_V_arr[valid_hl_V], c="C1", s=36, zorder=3, marker="s", alpha=0.85)
    axs[1, 1].set_title(r"Half-life $V$: time to midpoint of $V(0)\to V(T)$")
    axs[1, 1].set_xlabel(r"Post-shock signal $g$")
    axs[1, 1].set_ylabel("Half-life (time units)")
    axs[1, 1].grid(True, alpha=0.3)

    if r2_U is not None and np.any(np.isfinite(r2_U)):
        axs[0, 1].text(
            0.02,
            0.98,
            "Note: true relaxation near a fixed point is often exponential;\n"
            r"log–log $R^2$ can be low. See docstring in experiments.",
            transform=axs[0, 1].transAxes,
            fontsize=7,
            verticalalignment="top",
        )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return fig


def default_shock_g_grid():
    """Post-shock constant G levels: 0.4 … 0.1 and -0.1 … -1.0."""
    pos = [0.4, 0.3, 0.2, 0.1, .6, .8, 1, 1.3, 1.6,2]
    neg = [-.1,-.3,-.5]
    return np.sort(pos + neg)