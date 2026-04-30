import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats

from plotting.paths import OUTPUT_DIR
from plotting.rates import aggregate_vacancy_rate
from time_grid import mid_run_plot_slice


def plot_beveridge_trajectory(economy_name, unemployment_rate, vacancy_rate, output_dir, burn_in=0):
    """Color-coded Beveridge trajectory (unemployment vs vacancy rate)."""
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(8, 6))

    ur = np.array(unemployment_rate[burn_in:])
    vr = np.array(vacancy_rate[burn_in:])

    points = np.array([ur, vr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    n_segments = len(segments)
    colors = np.linspace(0, 1, n_segments)

    lc = LineCollection(segments, cmap='viridis_r', linewidth=2)
    lc.set_array(colors)

    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax, label='Time (early → late)')

    ax.set_xlim(ur.min() * 0.95, ur.max() * 1.05)
    ax.set_ylim(vr.min() * 0.95, vr.max() * 1.05)

    ax.set_xlabel("Unemployment Rate")
    ax.set_ylabel("Vacancy Rate")
    ax.set_title(f"Beveridge Curve ({economy_name})")
    ax.grid(True, alpha=0.3)

    filename = "beveridge" + economy_name + ".pdf"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath)
    plt.close()


def plot_beveridge_comparison(curves_data, filename=None):
    if filename is None:
        filename = os.path.join(OUTPUT_DIR, "beveridge_comparison.pdf")
    """
    Plots multiple Beveridge curves on the same axes for comparison.

    Args:
        curves_data (dict): A dictionary where keys are labels for the curves
                            and values are tuples of (vacancy_rate, unemployment_rate).
        filename (str, optional): The name of the file to save the plot to. 
                                  Defaults to "beveridge_comparison.pdf".
    """
    plt.figure(figsize=(8, 6))
    
    for label, (vr, ur) in curves_data.items():
        plt.plot(ur, vr, label=label)

    plt.xlabel("Unemployment Rate")
    plt.ylabel("Vacancy Rate")
    plt.title("Beveridge Curve Comparison")
    # plt.axvline(x=SURPLUS_XI, color='red', linestyle='--', label=f'Min Unemployment Rate ($\\xi={SURPLUS_XI}$)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved Beveridge curve comparison plot to {filename}")

def plot_response_to_demand(demand_signal, aggregate_unemployment, aggregate_vacancies, economy_name, population, output_dir=None):
    """
    Plots unemployment rate ``U/L`` and vacancy rate ``v/(v+e)`` (aggregate vacancies over
    ``v+e`` with ``e = L - U``) as functions of the demand signal.

    Args:
        demand_signal (pd.Series): The time series of the economic signal (G(t)).
        aggregate_unemployment (list): The aggregate unemployment levels over time.
        aggregate_vacancies (list): The aggregate vacancy levels over time.
        economy_name (str): A name for the economy (e.g., "single_firm_ar2") for plot titles and filenames.
        population (float): Labor force ``L`` for unemployment rate and vacancy rate.
        output_dir (str, optional): Directory to save the plot. Defaults to OUTPUT_DIR.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    u_arr = np.asarray(aggregate_unemployment, dtype=float)
    unemployment_rate = u_arr / float(population)
    vacancy_rate = aggregate_vacancy_rate(u_arr, aggregate_vacancies, population)

    plt.figure(figsize=(12, 10), constrained_layout=True)

    # Plot Unemployment Rate vs. Demand
    plt.subplot(2, 1, 1) # 2 rows, 1 column, first plot
    plt.plot(demand_signal, unemployment_rate, label='Unemployment Rate', color='blue')
    plt.xlabel("Demand Signal (G(t))")
    plt.ylabel(r"Unemployment rate $u=U/L$")
    plt.title(f"Unemployment Rate vs. Demand Signal ({economy_name})")
    plt.grid(True)
    plt.legend()

    # Plot Vacancy Rate vs. Demand
    plt.subplot(2, 1, 2) # 2 rows, 1 column, second plot
    plt.plot(demand_signal, vacancy_rate, label='Vacancy Rate', color='orange')
    plt.xlabel("Demand Signal (G(t))")
    plt.ylabel(r"Vacancy rate $v/(v+e)$")
    plt.title(f"Vacancy Rate vs. Demand Signal ({economy_name})")
    plt.grid(True)
    plt.legend()

    filename = f"response_to_demand_{economy_name}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved response to demand plot to {filepath}")
def plot_multi_employment(firms, unemployment, time, population, economy_name, output_dir=None):
    """
    Plots the employment of multiple individual firms and the aggregate unemployment.
    """
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    llim, ulim = mid_run_plot_slice()
    
    # Extract C values for colormapping
    c_values = [firm.sensitivity_coefficient for firm in firms]
    min_c = min(c_values)
    max_c = max(c_values)
    
    # Use a continuous colormap
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=min_c, vmax=max_c)
    
    # Plot employment for each firm
    for i, firm in enumerate(firms):
        c_val = firm.sensitivity_coefficient
        ax.plot(time[llim:ulim], firm.employment[llim:ulim], color=cmap(norm(c_val)), linestyle='-', linewidth=1.5, alpha=0.8)
        ax.plot(time[llim:ulim], firm.vacancies[llim:ulim], color=cmap(norm(c_val)), linestyle='--', linewidth=1.5, alpha=0.8)

    # Plot aggregate unemployment
    ax.plot(time[llim:ulim], unemployment[llim:ulim], label="Aggregate Unemployment", color='black', linestyle=':', linewidth=2)

    # --- Population Conservation Check ---
    # Plot theoretical constant population
    # ax.axhline(y=population, color='gray', linestyle='--', label='Total Population (Theoretical)')

    # Calculate and plot the actual total people in the system
    # total_employment = np.sum([np.array(f.employment) for f in firms], axis=0)
    # total_people_actual = total_employment + np.array(unemployment)
    # ax.plot(time[llim:ulim], total_people_actual[llim:ulim], color='blue', linestyle='-.', label='Total Population (Actual)')

    ax.set_xlabel("Time")
    ax.set_ylabel("Number of People")
    ax.set_title(f"Employment Dynamics ({economy_name})")
    
    # Legend only for the aggregate/system lines, avoiding clutter from individual firms
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Sensitivity Coefficient (C)')
    
    filename = f"employment_{economy_name}.pdf"
    if output_dir:
        filename = os.path.join(output_dir, filename)
    else:
        filename = os.path.join(OUTPUT_DIR, filename)
        
    plt.savefig(filename)
    plt.close()
    print(f"Saved multi-firm employment plot to {filename}")

def plot_employment_growth_rate(firms, time, economy_name, output_dir=None):
    """
    Plots the employment growth rate (de/dt)/e vs time for each firm.
    This shows the percentage change in employment per unit time.
    """
    plt.figure(figsize=(10, 6))
    
    colors = cm.get_cmap('tab10', len(firms))
    for i, firm in enumerate(firms):
        employment = np.array(firm.employment)
        
        # Calculate growth rate: (e[t+1] - e[t]) / e[t]
        # Handle division by zero by using np.where
        growth_rate = np.zeros_like(employment, dtype=float)
        for t in range(len(employment) - 1):
            if employment[t] > 0:
                growth_rate[t] = (employment[t+1] - employment[t]) / employment[t]
            else:
                growth_rate[t] = 0.0
        
        # Convert to percentage
        growth_rate_pct = growth_rate * 100
        
        plt.plot(time[:-1], growth_rate_pct[:-1], 
                label=f"Firm {i+1} Growth Rate (C={firm.sensitivity_coefficient})", 
                color=colors(i), linestyle='-', linewidth=2)
    
    # Add a horizontal line at 0 for reference
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add a vertical line at the shock time
    shock_time_index = 40
    if shock_time_index < len(time):
        plt.axvline(x=time[shock_time_index], color='red', linestyle=':', linewidth=2, 
                   label='Recession Start', alpha=0.7)

    plt.xlabel("Time")
    plt.ylabel("Employment Growth Rate (% per time unit)")
    plt.title(f"Employment Growth Rate Over Time ({economy_name})")
    plt.legend()
    plt.grid(True)
    
    filename = f"employment_growth_rate_{economy_name}.pdf"
    if output_dir:
        filename = os.path.join(output_dir, filename)
    else:
        filename = os.path.join(OUTPUT_DIR, filename)
        
    plt.savefig(filename)
    plt.close()
    print(f"Saved employment growth rate plot to {filename}")
def plot_efficiency_time_series(firms, time, economy_name, output_dir=None):
    """
    Plots the employment efficiency measure over time for each firm.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    # Limit to first 1000 steps for clarity, consistent with other time series plots
    llimit, ulimit = mid_run_plot_slice()
    
    # Extract C values for colormapping
    c_values = [firm.sensitivity_coefficient for firm in firms]
    min_c = min(c_values)
    max_c = max(c_values)
    
    # Use a continuous colormap
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=min_c, vmax=max_c)
    
    for i, firm in enumerate(firms):
        efficiency = np.array(firm.efficiency)
        c_val = firm.sensitivity_coefficient
        ax.plot(time[llimit:ulimit], efficiency[llimit:ulimit], 
                 color=cmap(norm(c_val)), linestyle='-', linewidth=1.5, alpha=0.8)
    
    # Add a horizontal line at 0 for reference (where e = e_hat)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel("Time")
    ax.set_ylabel(r"Employment Efficiency Gap $\epsilon = (e - \hat{e}) / \hat{e}$")
    ax.set_title(f"Employment Efficiency Over Time ({economy_name})")
    
    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Sensitivity Coefficient (C)')
    
    ax.set_ylim(-0.15, 0.05)
    ax.grid(True, alpha=0.3)
    
    filename = f"efficiency_timeseries_{economy_name}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved efficiency time series plot to {filepath}")

def plot_aggregate_overview(demand_signal, firms, unemployment, time, population, economy_name, output_dir=None):
    """
    Plots GDP signal, aggregate unemployment rate, and aggregate vacancy rate
    on a shared time axis. Two panels: signal on top, rates on bottom.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    llimit, ulimit = mid_run_plot_slice()

    signal = np.array(demand_signal)
    unemployment_arr = np.array(unemployment)

    aggregate_vacancies = np.sum([np.array(f.vacancies) for f in firms], axis=0)

    unemployment_rate = unemployment_arr / population
    vacancy_rate = aggregate_vacancy_rate(unemployment_arr, aggregate_vacancies, population)

    fig, (ax_signal, ax_rates) = plt.subplots(2, 1, figsize=(10, 6),
                                               constrained_layout=True, sharex=True)

    ax_signal.plot(time[llimit:ulimit], signal[llimit:ulimit], color='green', linewidth=1.5)
    ax_signal.set_ylabel("Signal G(t)")
    ax_signal.set_title(f"Aggregate Overview ({economy_name})")
    ax_signal.grid(True, alpha=0.3)

    ax_rates.plot(time[llimit:ulimit], unemployment_rate[llimit:ulimit],
                  label='Unemployment Rate', color='steelblue', linewidth=1.5)
    ax_rates.plot(time[llimit:ulimit], vacancy_rate[llimit:ulimit],
                  label='Vacancy Rate', color='darkorange', linewidth=1.5)
    ax_rates.set_xlabel("Time")
    ax_rates.set_ylabel("Rate")
    ax_rates.legend()
    ax_rates.grid(True, alpha=0.3)

    filename = f"aggregate_overview_{economy_name}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved aggregate overview plot to {filepath}")


def plot_multi_firm_signal_gap_scatters(
    demand_signal,
    firms,
    unemployment,
    aggregate_vacancies,
    population,
    economy_name,
    output_dir=None,
    burn_in=0,
):
    """
    Multipanel scatter of **rates** vs the GDP signal and normalized employment gaps
    (post–burn-in), with $L$ = labor force:

    - Unemployment rate $u = U/L$.
    - Employment rate $e = E/L$ with $E = \\sum_i e_i = L - U$.
    - Vacancy rate $v/(v+e)$ with aggregate $v,e$ (:func:`plotting.rates.aggregate_vacancy_rate`).
    - Target demand rate $\\hat{d} = D/L$ with $D = \\sum_i \\hat{e}_i$ from
      ``employment_demand``.

    Gap axes: $(e - \\hat{d}) = (E-D)/L$ and $(e - V/L) = E/L - V/L$.

    Panels: $v$ vs $G$, $u$ vs $G$, $v$ vs $(e-\\hat{d})$, $u$ vs $(e-\\hat{d})$,
    $v$ vs $(e - V/L)$, $u$ vs $(e - V/L)$.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    L = float(population)
    if L <= 0:
        return

    signal = np.asarray(demand_signal, dtype=float).ravel()
    U_lvl = np.asarray(unemployment, dtype=float).ravel()
    V_lvl = np.asarray(aggregate_vacancies, dtype=float).ravel()
    n = min(len(signal), len(U_lvl), len(V_lvl))
    if n == 0:
        return
    signal, U_lvl, V_lvl = signal[:n], U_lvl[:n], V_lvl[:n]

    emp_mat = np.array([np.asarray(f.employment, dtype=float)[:n] for f in firms])
    dem_mat = np.array([np.asarray(f.employment_demand, dtype=float)[:n] for f in firms])
    E_lvl = emp_mat.sum(axis=0)
    D_lvl = dem_mat.sum(axis=0)

    u_rate = U_lvl / L
    e_rate = E_lvl / L
    d_rate = D_lvl / L
    v_rate = aggregate_vacancy_rate(U_lvl, V_lvl, L)
    gap_ed_rate = e_rate - d_rate
    gap_ev_rate = e_rate - V_lvl / L

    bi = int(max(0, burn_in))
    g = signal[bi:]
    ur = u_rate[bi:]
    vr = v_rate[bi:]
    ger = gap_ed_rate[bi:]
    gvr = gap_ev_rate[bi:]

    if len(g) < 2:
        print("plot_multi_firm_signal_gap_scatters: insufficient post–burn-in points; skipping.")
        return

    fig, axs = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    fig.suptitle(
        rf"Unemployment / vacancy / employment-gap rates vs signal ({economy_name})",
        fontsize=12,
    )

    kw = dict(s=6, alpha=0.2, edgecolors="none")

    axs[0, 0].scatter(g, vr, c="C1", **kw)
    axs[0, 0].set_xlabel(r"Signal $G(t)$")
    axs[0, 0].set_ylabel(r"Vacancy rate $v/(v+e)$")
    axs[0, 0].set_title(r"$v$ vs $G$")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].scatter(g, ur, c="C0", **kw)
    axs[0, 1].set_xlabel(r"Signal $G(t)$")
    axs[0, 1].set_ylabel(r"Unemployment rate $u = U/L$")
    axs[0, 1].set_title(r"$u$ vs $G$")
    axs[0, 1].grid(True, alpha=0.3)

    axs[0, 2].scatter(ger, vr, c="C1", **kw)
    axs[0, 2].set_xlabel(r"$(E-D)/L = e - \hat{d}$")
    axs[0, 2].set_ylabel(r"Vacancy rate $v/(v+e)$")
    axs[0, 2].set_title(r"$v$ vs employment $-$ demand (rates)")
    axs[0, 2].grid(True, alpha=0.3)

    axs[1, 0].scatter(ger, ur, c="C0", **kw)
    axs[1, 0].set_xlabel(r"$(E-D)/L = e - \hat{d}$")
    axs[1, 0].set_ylabel(r"Unemployment rate $u$")
    axs[1, 0].set_title(r"$u$ vs employment $-$ demand (rates)")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].scatter(gvr, vr, c="C1", **kw)
    axs[1, 1].set_xlabel(r"$e - V/L = E/L - V/L$")
    axs[1, 1].set_ylabel(r"Vacancy rate $v/(v+e)$")
    axs[1, 1].set_title(r"$v$ vs $e - V/L$")
    axs[1, 1].grid(True, alpha=0.3)

    axs[1, 2].scatter(gvr, ur, c="C0", **kw)
    axs[1, 2].set_xlabel(r"$e - V/L = E/L - V/L$")
    axs[1, 2].set_ylabel(r"Unemployment rate $u$")
    axs[1, 2].set_title(r"$u$ vs $e - V/L$")
    axs[1, 2].grid(True, alpha=0.3)


    filename = f"signal_gap_scatters_{economy_name}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved signal / gap scatter figure to {filepath}")


def _skew_sample(x):
    """Third standardized moment γ₁; ``nan`` if fewer than 3 finite points."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    return float(stats.skew(x, bias=False))


def plot_multi_firm_skewness_bars(
    demand_signal,
    firms,
    unemployment,
    aggregate_vacancies,
    population,
    economy_name,
    output_dir=None,
    burn_in=0,
):
    """
    Bar chart of sample skewness (unbiased ``scipy.stats.skew``) on post–burn-in
    windows for:

    - Vacancy rate ``v/(v+e)`` (:func:`plotting.rates.aggregate_vacancy_rate`)
    - Unemployment rate ``u = U/L``
    - Raw GDP signal ``G(t)``
    - Aggregate demand ``D(t) = sum_i hat{e}_i(t)``
    - Log growth of demand ``ln(D_{t+1}/D_t)`` where ``D_t, D_{t+1} > 0``
    - Employment minus demand ``E(t) - D(t)`` (levels)
    - First difference ``Delta(E-D)_t`` (one-step change in the gap)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    L = float(population)
    if L <= 0:
        return

    signal = np.asarray(demand_signal, dtype=float).ravel()
    U_lvl = np.asarray(unemployment, dtype=float).ravel()
    V_lvl = np.asarray(aggregate_vacancies, dtype=float).ravel()
    n = min(len(signal), len(U_lvl), len(V_lvl))
    if n == 0:
        return
    signal, U_lvl, V_lvl = signal[:n], U_lvl[:n], V_lvl[:n]

    emp_mat = np.array([np.asarray(f.employment, dtype=float)[:n] for f in firms])
    dem_mat = np.array([np.asarray(f.employment_demand, dtype=float)[:n] for f in firms])
    E_lvl = emp_mat.sum(axis=0)
    D_lvl = dem_mat.sum(axis=0)

    u_rate = U_lvl / L
    v_rate = aggregate_vacancy_rate(U_lvl, V_lvl, L)

    bi = int(max(0, burn_in))
    g = signal[bi:]
    u_s = u_rate[bi:]
    v_s = v_rate[bi:]
    d_s = D_lvl[bi:]
    e_s = E_lvl[bi:]

    if len(g) < 3:
        print("plot_multi_firm_skewness_bars: insufficient post–burn-in points; skipping.")
        return

    d0, d1 = d_s[:-1], d_s[1:]
    valid_d = (d0 > 0) & (d1 > 0)
    d_log_growth = np.log(np.divide(d1, d0, out=np.full_like(d1, np.nan), where=valid_d))
    d_log_growth = d_log_growth[np.isfinite(d_log_growth)]

    gap_lvl = e_s - d_s
    gap_diff = np.diff(gap_lvl)

    labels = [
        r"GDP signal $G(t)$",
        r"Demand $D=\sum_i \hat{e}_i$",
        r"$\ln(D_{t+1}/D_t)$",
        r"$E-D$",
        r"$\Delta(E-D)$",
        r"Vacancy rate $v/(v+e)$",
        r"Unemployment rate $u=U/L$",

    ]
    skews = [
        _skew_sample(g),
        _skew_sample(d_s),
        _skew_sample(d_log_growth),
        _skew_sample(gap_lvl),
        _skew_sample(gap_diff),
        _skew_sample(v_s),
        _skew_sample(u_s),
    ]

    y = np.arange(len(labels))
    heights = np.array([s if np.isfinite(s) else 0.0 for s in skews], dtype=float)
    colors = ["C0" if np.isfinite(s) else "0.75" for s in skews]

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    bars = ax.barh(y, heights, color=colors, edgecolor="0.35", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"Sample skewness $\gamma_1$ (bias-corrected)")
    ax.axvline(0.0, color="0.45", linestyle="--", linewidth=0.9)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title(f"Post–burn-in skewness summary ({economy_name})")

    skews_arr = np.asarray(skews, dtype=float)
    finite_h = heights[np.isfinite(skews_arr)]
    span = float(np.nanmax(np.abs(finite_h))) if finite_h.size else 1.0
    pad = -1
    for bar, s in zip(bars, skews):
        yi = bar.get_y() + bar.get_height() * 0.5
        if np.isfinite(s):
            w = bar.get_width()
            txt = f"{s:.4f}"
            ax.text(pad, yi, txt, va="center", ha="right", fontsize=8, color="0.2")
        else:
            ax.text(pad, yi, "n/a", va="center", ha="left", fontsize=8, color="0.45")
    ax.margins(x=0.14)

    filename = f"skewness_bars_{economy_name}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved skewness bar chart to {filepath}")


def plot_vacancy_time_series(firms, time, economy_name, output_dir=None):
    """
    Plots individual firm vacancy levels over time, colored by sensitivity coefficient C.
    Analogous to plot_efficiency_time_series.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    llimit, ulimit = mid_run_plot_slice()

    c_values = [firm.sensitivity_coefficient for firm in firms]
    min_c = min(c_values)
    max_c = max(c_values)

    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=min_c, vmax=max_c)

    for firm in firms:
        vacancies = np.array(firm.vacancies)
        c_val = firm.sensitivity_coefficient
        ax.plot(time[llimit:ulimit], vacancies[llimit:ulimit],
                color=cmap(norm(c_val)), linestyle='-', linewidth=1.5, alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Vacancies")
    ax.set_title(f"Individual Firm Vacancies ({economy_name})")
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Sensitivity Coefficient (C)')

    filename = f"vacancy_timeseries_{economy_name}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved vacancy time series plot to {filepath}")
