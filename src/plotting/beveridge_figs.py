import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from plotting.paths import OUTPUT_DIR
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
    Plots unemployment and vacancy rates (normalized by population) as functions of the demand signal.

    Args:
        demand_signal (pd.Series): The time series of the economic signal (G(t)).
        aggregate_unemployment (list): The aggregate unemployment levels over time.
        aggregate_vacancies (list): The aggregate vacancy levels over time.
        economy_name (str): A name for the economy (e.g., "single_firm_ar2") for plot titles and filenames.
        population (float): The total population used to normalize unemployment and vacancies into rates.
        output_dir (str, optional): Directory to save the plot. Defaults to OUTPUT_DIR.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Convert to rates (divide by population)
    unemployment_rate = np.array(aggregate_unemployment) / population
    vacancy_rate = np.array(aggregate_vacancies) / population

    plt.figure(figsize=(12, 10), constrained_layout=True)

    # Plot Unemployment Rate vs. Demand
    plt.subplot(2, 1, 1) # 2 rows, 1 column, first plot
    plt.plot(demand_signal, unemployment_rate, label='Unemployment Rate', color='blue')
    plt.xlabel("Demand Signal (G(t))")
    plt.ylabel("Unemployment Rate (U / Population)")
    plt.title(f"Unemployment Rate vs. Demand Signal ({economy_name})")
    plt.grid(True)
    plt.legend()

    # Plot Vacancy Rate vs. Demand
    plt.subplot(2, 1, 2) # 2 rows, 1 column, second plot
    plt.plot(demand_signal, vacancy_rate, label='Vacancy Rate', color='orange')
    plt.xlabel("Demand Signal (G(t))")
    plt.ylabel("Vacancy Rate (V / Population)")
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
    aggregate_employment = np.sum([np.array(f.employment) for f in firms], axis=0)

    unemployment_rate = unemployment_arr / population
    vacancy_rate = aggregate_vacancies / (aggregate_vacancies + aggregate_employment)

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
