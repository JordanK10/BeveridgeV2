import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import skewnorm
import mpl_toolkits.axes_grid1.axes_divider as divider

from plotting.paths import OUTPUT_DIR
from time_grid import STEPS


def skew_normal_skewness_gamma1(alpha):
    """
    Third standardized moment γ₁ for X ~ SN(ξ, ω, α) (scipy ``skewnorm`` shape ``a`` = α).

    δ = α / √(1 + α²),
    γ₁ = ((4 - π) / 2) (δ √(2/π))³ / (1 - 2δ²/π)^(3/2).
    """
    alpha = float(alpha)
    delta = alpha / np.sqrt(1.0 + alpha**2)
    inner = 1.0 - 2.0 * delta**2 / np.pi
    if inner <= 0:
        return float("nan")
    return ((4.0 - np.pi) / 2.0) * (delta * np.sqrt(2.0 / np.pi)) ** 3 / (inner**1.5)


def plot_fluctuation_analysis(
    demand_signal,
    aggregate_unemployment,
    aggregate_vacancies,
    economy_name,
    population,
    output_dir=None,
    show_trace=False,
    burn_in=0,
    firms=None,
):
    """
    Plots histograms of unemployment rate u=U/L, vacancy rate V/(V+E), and log growth of
    aggregate labor demand D(t)=sum_i ê_i(t), plus 2D heatmaps (rates vs that growth).

    When ``firms`` is provided, third column and bottom-row x-axes use
    ln(D(t+1)) - ln(D(t)) with D(t) = sum of ``employment_demand`` across firms (interpretable
    common scale despite heterogeneous σ_i, C_i). If ``firms`` is None, falls back to
    ln(G(t+1)/G(t)) for the legacy signal-only definition.

    Args:
        demand_signal (pd.Series): The time series of the economic signal G(t) (used if firms is None).
        aggregate_unemployment (list): The aggregate unemployment levels over time.
        aggregate_vacancies (list): The aggregate vacancy levels over time.
        economy_name (str): A name for the economy (e.g., "single_firm_ar2") for plot titles and filenames.
        population (float): The total population, needed for rate calculations and normalization.
        output_dir (str, optional): Directory to save the plot. Defaults to OUTPUT_DIR.
        show_trace (bool, optional): If True, plots a white trace (alpha=0.5) over the heatmaps showing the time series path. Defaults to False.
        burn_in (int): Number of leading steps to discard before computing statistics.
        firms (list, optional): ``Firm`` instances; if set, aggregate labor demand D is summed from ``employment_demand``.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Convert to numpy arrays and drop burn-in period
    unemployment_np = np.array(aggregate_unemployment)[burn_in:]
    vacancies_np = np.array(aggregate_vacancies)[burn_in:]

    employment_np = population - unemployment_np
    unemployment_rate = unemployment_np / population
    denom_v = vacancies_np + employment_np
    vacancy_rate = np.divide(
        vacancies_np,
        denom_v,
        out=np.zeros_like(vacancies_np, dtype=float),
        where=denom_v > 0,
    )

    unemployment_rate_lag = unemployment_rate[:-1]
    vacancy_rate_lag = vacancy_rate[:-1]

    if firms is not None:
        n_steps = len(firms[0].employment_demand)
        D_full = np.array([sum(f.employment_demand[t] for f in firms) for t in range(n_steps)])
        D_np = D_full[burn_in:]
        d0 = D_np[:-1]
        d1 = D_np[1:]
        valid_growth = (d0 > 0) & (d1 > 0)
        demand_log_growth = np.full(len(d0), np.nan, dtype=float)
        demand_log_growth[valid_growth] = np.log(d1[valid_growth]) - np.log(d0[valid_growth])
        demand_log_growth_f = demand_log_growth[np.isfinite(demand_log_growth)]
        growth_xlabel = r"$\ln D(t+1) - \ln D(t)$"
        growth_title_hist = r"Histogram of $\ln D(t+1) - \ln D(t)$"
        growth_title_heat_u = r"Unemployment rate vs. $\ln D(t+1) - \ln D(t)$"
        growth_title_heat_v = r"Vacancy rate vs. $\ln D(t+1) - \ln D(t)$"
        empty_growth_msg = "No valid ln D(t+1) - ln D(t)\n(requires D(t), D(t+1) > 0)"
        drop_msg_fmt = (
            "fluctuation_analysis: dropped {n} steps with D(t)≤0 or D(t+1)≤0 "
            "(out of {m} aggregate-demand growth observations)"
        )
    else:
        demand_np = np.array(demand_signal)[burn_in:]
        g0 = demand_np[:-1]
        g1 = demand_np[1:]
        valid_growth = (g0 != 0) & (g1 != 0) & (g0 * g1 > 0)
        ratio = np.divide(g1, g0, out=np.full(len(g0), np.nan, dtype=float), where=valid_growth)
        demand_log_growth = np.log(ratio)
        demand_log_growth_f = demand_log_growth[np.isfinite(demand_log_growth)]
        growth_xlabel = r"$\ln(G(t+1)/G(t))$"
        growth_title_hist = r"Histogram of $\ln(G(t+1)/G(t))$"
        growth_title_heat_u = r"Unemployment rate vs. $\ln(G(t+1)/G(t))$"
        growth_title_heat_v = r"Vacancy rate vs. $\ln(G(t+1)/G(t))$"
        empty_growth_msg = "No valid ln(G(t+1)/G(t))\n(same-sign nonzero G required)"
        drop_msg_fmt = (
            "fluctuation_analysis: dropped {n} steps where G(t)G(t+1)≤0 or G=0 "
            "(out of {m} growth observations)"
        )

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(f"Fluctuation Analysis ({economy_name})", fontsize=16)

    # Row 0: All Histograms
    # Subplot 1: Histogram of unemployment rate u = U/L
    axs[0, 0].hist(unemployment_rate, bins=30, density=True, edgecolor='black', alpha=0.7, color='blue', label='Histogram')
    # Fit a Skew-Normal distribution
    a_u, loc_u, scale_u = skewnorm.fit(unemployment_rate)
    xmin_u, xmax_u = axs[0, 0].get_xlim()
    x_u = np.linspace(xmin_u, xmax_u, 100)
    p_u = skewnorm.pdf(x_u, a_u, loc_u, scale_u)
    g1_u = skew_normal_skewness_gamma1(a_u)
    axs[0, 0].plot(
        x_u,
        p_u,
        "k",
        linewidth=2,
        label=rf"Skew-Normal fit ($\gamma_1$={g1_u:.3f}, loc={loc_u:.2f}, scale={scale_u:.2f})",
    )
    axs[0, 0].set_title("Histogram of Unemployment Rate")
    axs[0, 0].set_xlabel("Unemployment rate u = U / L")
    axs[0, 0].set_ylabel("Density")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Subplot 2: Histogram of vacancy rate V/(V+E) (same as vacancy_and_unemployment_rates in rates.py)
    axs[0, 1].hist(vacancy_rate, bins=30, density=True, edgecolor='black', alpha=0.7, color='orange', label='Histogram')
    # Fit a Skew-Normal distribution
    a_v, loc_v, scale_v = skewnorm.fit(vacancy_rate)
    xmin_v, xmax_v = axs[0, 1].get_xlim()
    x_v = np.linspace(xmin_v, xmax_v, 100)
    p_v = skewnorm.pdf(x_v, a_v, loc_v, scale_v)
    g1_v = skew_normal_skewness_gamma1(a_v)
    axs[0, 1].plot(
        x_v,
        p_v,
        "k",
        linewidth=2,
        label=rf"Skew-Normal fit ($\gamma_1$={g1_v:.3f}, loc={loc_v:.2f}, scale={scale_v:.2f})",
    )
    axs[0, 1].set_title("Histogram of Vacancy Rate")
    axs[0, 1].set_xlabel("Vacancy rate V / (V + E)")
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Subplot 3: Histogram of log growth (aggregate D or legacy G)
    if len(demand_log_growth_f) > 0:
        axs[0, 2].hist(demand_log_growth_f, bins=30, density=True, edgecolor='black', alpha=0.7, color='green', label='Histogram')
    if len(demand_log_growth_f) > 2 and np.var(demand_log_growth_f) > 1e-10:
        try:
            a_d, loc_d, scale_d = skewnorm.fit(demand_log_growth_f)
            xmin_d, xmax_d = axs[0, 2].get_xlim()
            x_d = np.linspace(xmin_d, xmax_d, 100)
            p_d = skewnorm.pdf(x_d, a_d, loc_d, scale_d)
            g1_d = skew_normal_skewness_gamma1(a_d)
            axs[0, 2].plot(
                x_d,
                p_d,
                "k",
                linewidth=2,
                label=rf"Skew-Normal fit ($\gamma_1$={g1_d:.3f}, loc={loc_d:.2f}, scale={scale_d:.2f})",
            )
        except Exception as e:
            print(f"Could not fit skew-normal to log demand growth: {e}")
    elif len(demand_log_growth_f) == 0:
        axs[0, 2].text(0.5, 0.5, empty_growth_msg, ha="center", va="center", transform=axs[0, 2].transAxes)

    axs[0, 2].set_title(growth_title_hist)
    axs[0, 2].set_xlabel(growth_xlabel)
    axs[0, 2].set_ylabel("Density")
    axs[0, 2].grid(True)
    if len(demand_log_growth_f) > 0:
        axs[0, 2].legend()

    n_drop = len(demand_log_growth) - len(demand_log_growth_f)
    if n_drop > 0:
        print(drop_msg_fmt.format(n=n_drop, m=len(demand_log_growth)))

    # Row 1: All Heatmaps
    # Subplot 4: 2D Heatmap of Unemployment Rate vs. Vacancy Rate
    hb5 = axs[1, 0].hist2d(unemployment_rate, vacancy_rate, bins=50, cmap='viridis')
    if show_trace:
        axs[1, 0].plot(unemployment_rate, vacancy_rate, 'white', alpha=0.5, linewidth=1)
    axs[1, 0].set_title("Unemployment Rate vs. Vacancy Rate")
    axs[1, 0].set_xlabel("Unemployment rate u = U / L")
    axs[1, 0].set_ylabel("Vacancy rate V / (V + E)")
    fig.colorbar(hb5[3], ax=axs[1, 0], label='Density')
    axs[1, 0].grid(True)

    # Subplot 5: u(t) vs log growth (D or G) at time t (aligned lengths)
    mask_d = np.isfinite(demand_log_growth)
    hb3 = axs[1, 1].hist2d(
        demand_log_growth[mask_d],
        unemployment_rate_lag[mask_d],
        bins=50,
        cmap='viridis',
    )
    if show_trace:
        axs[1, 1].plot(
            demand_log_growth[mask_d],
            unemployment_rate_lag[mask_d],
            'white',
            alpha=0.5,
            linewidth=1,
        )
    axs[1, 1].set_title(growth_title_heat_u)
    axs[1, 1].set_xlabel(growth_xlabel)
    axs[1, 1].set_ylabel("Unemployment rate u = U / L")
    fig.colorbar(hb3[3], ax=axs[1, 1], label='Density')
    axs[1, 1].grid(True)

    # Subplot 6: v(t) vs log growth
    hb4 = axs[1, 2].hist2d(
        demand_log_growth[mask_d],
        vacancy_rate_lag[mask_d],
        bins=50,
        cmap='magma',
    )
    if show_trace:
        axs[1, 2].plot(
            demand_log_growth[mask_d],
            vacancy_rate_lag[mask_d],
            'white',
            alpha=0.5,
            linewidth=1,
        )
    axs[1, 2].set_title(growth_title_heat_v)
    axs[1, 2].set_xlabel(growth_xlabel)
    axs[1, 2].set_ylabel("Vacancy rate V / (V + E)")
    fig.colorbar(hb4[3], ax=axs[1, 2], label='Density')
    axs[1, 2].grid(True)

    filename = f"fluctuation_analysis_{economy_name}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved fluctuation analysis plot to {filepath}")

def plot_growth_rate_heatmaps(sigma, sensitivity_c, matching_rate, separation_rate, dt, economy_name, output_dir=None):
    """
    Generates two side-by-side heatmaps showing vacancy growth rate and employment growth rate
    as functions of current employment (e) and current signal (G).
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Define ranges for employment and signal
    e_center = sigma # Using passed sigma as the center for employment
    offset = 30
    e_range = 0.15 * e_center + offset 
    e_values = np.linspace(e_center - e_range, e_center + e_range, 50)

    g_values = np.linspace(-1, 1, 50) # +/- 1 for signal value

    E_grid, G_grid = np.meshgrid(e_values, g_values)
    
    # Initialize arrays for growth rates
    vacancy_growth_rate_grid = np.zeros_like(E_grid, dtype=float)
    employment_growth_rate_grid = np.zeros_like(E_grid, dtype=float)
    
    # Loop through each point in the grid to calculate growth rates
    for i in range(E_grid.shape[0]):
        for j in range(E_grid.shape[1]):
            e_t = E_grid[i, j]
            G_t = G_grid[i, j]
            
            # Calculate target employment at time t
            e_hat_t = sigma * (1 + sensitivity_c * G_t)
            
            # Calculate current vacancies (v_t) (from Firm.update_vacancies logic)
            v_t_calc = e_hat_t - e_t + e_t * separation_rate / matching_rate
            v_t = np.maximum(0, v_t_calc)
            
            # Calculate absolute number of workers fired (discrete, instantaneous adjustment)
            fired_t = np.maximum(0, e_t - e_hat_t)
            
            # Calculate hires and separations (flows over dt)
            hires_dt = v_t * matching_rate * dt
            separations_dt = e_t * separation_rate * dt
            
            # Predict employment at t+1 based on discrete update rule
            e_t_plus_1 = e_t + hires_dt - separations_dt - fired_t

            # Ensure e_t is not zero for log calculation, use a small epsilon if needed
            if e_t > np.finfo(float).eps:
                employment_growth_rate_grid[i, j] = (np.log(np.maximum(e_t_plus_1, np.finfo(float).eps)) - np.log(e_t)) / dt
            else:
                employment_growth_rate_grid[i, j] = 0.0 # Or NaN, depending on desired visual

            # Calculate equilibrium vacancies needed to just maintain current employment - BASELINE
            v_equilibrium_e_t = e_t * separation_rate / matching_rate

            # Calculate Vacancy Gap Growth Rate: (ln(V_demand) - ln(V_equilibrium)) / dt
            # Ensure v_t and v_equilibrium_e_t are positive for log calculation
            # v_equilibrium_e_t will always be positive if e_t > 0 (which it is in e_values)
            if v_t > np.finfo(float).eps:
                vacancy_growth_rate_grid[i, j] = (np.log(v_t) - np.log(np.maximum(v_equilibrium_e_t, np.finfo(float).eps))) / dt
            else:
                # If demand-induced vacancies are zero, and equilibrium vacancies are positive,
                # this implies a large negative "growth" (vacancies are far below equilibrium)
                vacancy_growth_rate_grid[i, j] = (np.log(np.finfo(float).eps) - np.log(np.maximum(v_equilibrium_e_t, np.finfo(float).eps))) / dt
    
    # Plotting
    
    # Use gridspec_kw to define width ratios: [Plot, Gap, Plot]
    # Disable constrained_layout as it conflicts with axes_divider colorbars
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.75), constrained_layout=False, 
                            gridspec_kw={'width_ratios': [1, 0.05, 1]})
    
    # Manually adjust margins to make room for right-side colorbars
    # right=0.85 leaves 15% of width for the colorbars on the right edge
    plt.subplots_adjust(left=0.1, right=0.85, bottom=0.08, top=0.9, wspace=0.3, hspace=0.3)
    
    fig.suptitle(f"Growth Rate Heatmaps ({economy_name})", fontsize=16)

    # Hide the middle spacer column
    for ax in axs[:, 1]:
        ax.axis('off')

    # Custom colormaps for positive (white to blue) and negative (white to red)
    cmap_pos = plt.cm.get_cmap('Blues')
    cmap_pos.set_bad('white', alpha=0) # Set NaN to transparent
    cmap_neg = plt.cm.get_cmap('Reds')
    cmap_neg.set_bad('white', alpha=0) # Set NaN to transparent

    # --- 1. Top-Left: Vacancy Gap Growth Rate ---
    ax0 = axs[0, 0]
    
    # Data for plotting
    positive_vacancy_growth = np.where(vacancy_growth_rate_grid > np.finfo(float).eps, vacancy_growth_rate_grid, np.nan)
    negative_vacancy_growth_abs = np.where(vacancy_growth_rate_grid < -np.finfo(float).eps, np.abs(vacancy_growth_rate_grid), np.nan)

    # Plot positive growth
    im0_pos = ax0.contourf(np.log(positive_vacancy_growth), origin='lower',
                         extent=[e_values.min(), e_values.max(), g_values.min() * sensitivity_c, g_values.max() * sensitivity_c],
                         aspect='auto', cmap=cmap_pos)
    # Plot negative growth
    im0_neg = ax0.contourf(np.log(negative_vacancy_growth_abs), origin='lower',
                         extent=[e_values.min(), e_values.max(), g_values.min() * sensitivity_c, g_values.max() * sensitivity_c],
                         aspect='auto', cmap=cmap_neg)

    ax0.set_title("Vacancy Gap Growth Rate")
    ax0.set_xlabel("Employment (e)")
    ax0.set_ylabel("Signal Growth (CG)")
    ax0.set_ylim(g_values.min() * sensitivity_c, g_values.max() * sensitivity_c)

    # Add colorbars for ax0
    divider0 = divider.make_axes_locatable(ax0)
    cax0_pos = divider0.append_axes("right", size="5%", pad="2%")
    cbar0_pos = fig.colorbar(im0_pos, cax=cax0_pos, label="Log(Vacancy Gap Growth Rate) > 0")
    
    cax0_neg = divider0.append_axes("right", size="5%", pad="20%") # Offset to not overlap
    cbar0_neg = fig.colorbar(im0_neg, cax=cax0_neg, label="Log(|Vacancy Gap Growth Rate|) < 0")
    
    # --- 2. Top-Right: Employment Growth Rate ---
    ax1 = axs[0, 2]
    
    # Data for plotting
    positive_employment_growth = np.where(employment_growth_rate_grid > np.finfo(float).eps, employment_growth_rate_grid, np.nan)
    negative_employment_growth_abs = np.where(employment_growth_rate_grid < -np.finfo(float).eps, np.abs(employment_growth_rate_grid), np.nan)

    # Plot positive growth
    im1_pos = ax1.contourf(np.log(positive_employment_growth), origin='lower',
                         extent=[e_values.min(), e_values.max(), g_values.min() * sensitivity_c, g_values.max() * sensitivity_c],
                         aspect='auto', cmap=cmap_pos)
    # Plot negative growth
    im1_neg = ax1.contourf(np.log(negative_employment_growth_abs), origin='lower',
                         extent=[e_values.min(), e_values.max(), g_values.min() * sensitivity_c, g_values.max() * sensitivity_c],
                         aspect='auto', cmap=cmap_neg)

    ax1.set_title("Employment Growth Rate")
    ax1.set_xlabel("Employment (e)")
    ax1.set_ylabel("Signal Growth (CG)")

    # Add colorbars for ax1
    divider1 = divider.make_axes_locatable(ax1)
    cax1_pos = divider1.append_axes("right", size="5%", pad="2%")
    cbar1_pos = fig.colorbar(im1_pos, cax=cax1_pos, label="Log(Employment Growth Rate) > 0")
    
    cax1_neg = divider1.append_axes("right", size="5%", pad="20%") # Offset to not overlap
    cbar1_neg = fig.colorbar(im1_neg, cax=cax1_neg, label="Log(|Employment Growth Rate|) < 0")
    
    # --- 3. Bottom-Left: Vacancies ---
    ax2 = axs[1, 0]
    
    # Reconstruct v_t_grid for plotting (need to recalculate or store it)
    v_t_grid = np.zeros_like(E_grid)
    e_change_grid = np.zeros_like(E_grid)
    
    for i in range(E_grid.shape[0]):
        for j in range(E_grid.shape[1]):
            e_t = E_grid[i, j]
            G_t = G_grid[i, j]
            e_hat_t = sigma * (1 + sensitivity_c * G_t)
            v_t_calc = e_hat_t - e_t + e_t * separation_rate / matching_rate
            v_t = np.maximum(0, v_t_calc)
            v_t_grid[i, j] = v_t
            
            fired_t = np.maximum(0, e_t - e_hat_t)
            hires_dt = v_t * matching_rate * dt
            separations_dt = e_t * separation_rate * dt
            e_t_plus_1 = e_t + hires_dt - separations_dt - fired_t
            e_change_grid[i, j] = e_t_plus_1 - e_t
            
    im2 = ax2.contourf(v_t_grid, origin='lower',
                     extent=[e_values.min(), e_values.max(), g_values.min(), g_values.max()],
                     aspect='auto', cmap='viridis')
    ax2.set_title("Vacancies (v_t)")
    ax2.set_xlabel("Employment (e)")
    ax2.set_ylabel("Signal (G)")
    divider2 = divider.make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad="2%")
    fig.colorbar(im2, cax=cax2, label="Vacancies")

    # --- 4. Bottom-Right: Employment Change ---
    ax3 = axs[1, 2]
    # Use diverging colormap centered at 0 with custom range
    norm = mcolors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=50)
    im3 = ax3.contourf(e_change_grid, origin='lower',
                     extent=[e_values.min(), e_values.max(), g_values.min(), g_values.max()],
                     aspect='auto', cmap='viridis', norm=norm)
    ax3.set_title("Employment Change (e_{t+1} - e_t)")
    ax3.set_xlabel("Employment (e)")
    ax3.set_ylabel("Signal (G)")
    divider3 = divider.make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad="2%")
    fig.colorbar(im3, cax=cax3, label="Change in Employment")

    
    # Plot the equilibrium employment curve on all subplots and grid lines
    equilibrium_employment = sigma * (1 + sensitivity_c * g_values)
    
    # Iterate over the top row axes (columns 0 and 2) - scaled by C
    top_axes = [axs[0, 0], axs[0, 2]]
    for ax in top_axes:
        ax.plot(equilibrium_employment, g_values * sensitivity_c, color='yellow', linestyle='-', linewidth=2, label='Equilibrium Employment')
        ax.axvline(x=e_center, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Iterate over the bottom row axes (columns 0 and 2) - original G scale
    bottom_axes = [axs[1, 0], axs[1, 2]]
    for ax in bottom_axes:
        ax.plot(equilibrium_employment, g_values, color='yellow', linestyle='-', linewidth=2, label='Equilibrium Employment')
        ax.axvline(x=e_center, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

    filename = f"growth_rate_heatmaps_{economy_name}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved growth rate heatmaps plot to {filepath}")
def plot_growth_rate_analysis(demand_signal, total_employment, aggregate_vacancies, economy_name, 
                              matching_rate, separation_rate, sigma, sensitivity_c, dt, output_dir=None):
    """
    Plots a 2D heatmap of employment growth rate vs. signal growth rate.
    Growth rates are computed as 1/(dt) * ln(x_{t+1}/x_t).
    Also plots the theoretical instantaneous growth rate for comparison.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Ensure inputs are numpy arrays
    signal = np.array(demand_signal)
    employment = np.array(total_employment)
    vacancies = np.array(aggregate_vacancies)
    
    # --- 1. Compute Empirical Growth Rates ---
    # Employment is strictly positive, so we use log returns: 1/dt * ln(E_{t+1}/E_t)
    with np.errstate(divide='ignore', invalid='ignore'):
         employment_growth = (np.log(employment[1:]) - np.log(employment[:-1])) / dt
    
    # Compute growth rate of the *effective demand factor* (1 + C*G)
    ed_factor = 1 + sensitivity_c * signal
    
    with np.errstate(divide='ignore', invalid='ignore'):
        signal_growth = (np.log(ed_factor[1:]) - np.log(ed_factor[:-1])) / dt
    
    signal_label = "Effective Demand Growth (1/dt * ln((1+CG)_{t+1}/(1+CG)_t))"
    
    # --- 2. Compute Theoretical Instantaneous Growth Rate ---
    v_t = vacancies[:-1]
    e_t = employment[:-1]
    G_t = signal[:-1]
    # Note: In the single firm experiment, matching_function is hardcoded to 1/6 in update_vacancies
    # but here we use the passed matching_rate which might be constant
    m_t = matching_rate
    
    # Calculate target employment level at time t
    e_hat_t = sigma * (1 + sensitivity_c * G_t)
    
    # Calculate absolute number of workers fired (discrete, instantaneous adjustment)
    fired_t = np.maximum(0, e_t - e_hat_t)
    
    # Calculate hires and separations (flows over dt)
    hires_dt = v_t * m_t * dt
    separations_dt = e_t * separation_rate * dt
    
    # Predict employment at t+1 based on discrete update rule
    e_t_plus_1_theoretical = e_t + hires_dt - separations_dt - fired_t
    
    # Compute logarithmic growth rate from e_t to e_t+1_theoretical
    # Use np.maximum with a small epsilon to prevent log(0) for extremely low employment
    theoretical_gamma_e = (np.log(np.maximum(e_t_plus_1_theoretical, np.finfo(float).eps)) - np.log(e_t)) / dt

    # --- 3. Filter and Plot ---
    # Filter out NaNs or Infs
    valid_mask = np.isfinite(signal_growth) & np.isfinite(employment_growth) & np.isfinite(theoretical_gamma_e)
    
    signal_growth = signal_growth[valid_mask]
    employment_growth = employment_growth[valid_mask]
    theoretical_gamma_e = theoretical_gamma_e[valid_mask]

    plt.figure(figsize=(8, 6.4))
    
    # Heatmap of empirical data
    plt.hist2d(signal_growth, employment_growth, bins=50, cmap='viridis', density=True, label='Empirical Density')
    plt.colorbar(label='Density')

    plt.plot(signal_growth, theoretical_gamma_e, color='red', linewidth=1, alpha=1, label='Theoretical Prediction')

    
    # Add a trace for visual clarity (Empirical Path)
    # Since we masked, order is preserved relative to each other but time gaps might exist.
    plt.plot(signal_growth, employment_growth, 'white', alpha=0.1, linewidth=0.25, linestyle='--',label='Empirical Trace')
    
    # Scatter plot of theoretical predictions
    # We plot them as small red dots to see how they align with the heatmap

    plt.title(f"Employment Growth vs. Signal Growth ({economy_name})")
    plt.xlabel(signal_label)
    plt.ylabel("Employment Growth Rate (1/dt * ln(E_{t+1}/E_t))")
    plt.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    filename = f"growth_rate_analysis_{economy_name}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved growth rate analysis plot to {filepath}")


def plot_growth_rates_time_series(demand_signal, total_employment, aggregate_vacancies, economy_name, 
                                  matching_rate, separation_rate, sigma, sensitivity_c, dt, time_array, output_dir=None):
    """
    Plots the time series of empirical signal growth, empirical employment growth, 
    and theoretical employment growth.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Ensure inputs are numpy arrays
    signal = np.array(demand_signal)
    employment = np.array(total_employment)
    vacancies = np.array(aggregate_vacancies)
    
    # --- 1. Compute Empirical Growth Rates ---
    # Employment log returns
    with np.errstate(divide='ignore', invalid='ignore'):
         emp_growth = (np.log(employment[1:]) - np.log(employment[:-1])) / dt
    
    # Robust Signal Growth: Growth of (1 + C*G)
    ed_factor = 1 + sensitivity_c * signal
    with np.errstate(divide='ignore', invalid='ignore'):
        sig_growth = (np.log(ed_factor[1:]) - np.log(ed_factor[:-1])) / dt
    
    # --- 2. Compute Theoretical Instantaneous Growth Rate ---
    v_t = vacancies[:-1]
    e_t = employment[:-1]
    G_t = signal[:-1]
    # Note: In the single firm experiment, matching_function is hardcoded to 1/6 in update_vacancies
    m_t = matching_rate 
    
    # Calculate target employment level at time t
    e_hat_t = sigma * (1 + sensitivity_c * G_t)
    
    # Calculate absolute number of workers fired (discrete, instantaneous adjustment)
    fired_t = np.maximum(0, e_t - e_hat_t)
    
    # Calculate hires and separations (flows over dt)
    hires_dt = v_t * m_t * dt
    separations_dt = e_t * separation_rate * dt
    
    # Predict employment at t+1 based on discrete update rule
    e_t_plus_1_theoretical = e_t + hires_dt - separations_dt - fired_t
    
    # Compute logarithmic growth rate from e_t to e_t+1_theoretical
    # Use np.maximum with a small epsilon to prevent log(0) for extremely low employment
    theo_emp_growth = (np.log(np.maximum(e_t_plus_1_theoretical, np.finfo(float).eps)) - np.log(e_t)) / dt

    # --- 3. Plotting ---
    # Create time array for the intervals (using start time of interval)
    time_points = time_array[:-1]
    
    # ~1000-step window near 24% through horizon (adapts to STEPS)
    llimit = min(max(0, int(STEPS * 0.24)), max(0, STEPS - 2))
    ulimit = min(STEPS, llimit + min(1000, STEPS - llimit))
    
    fig, axs = plt.subplots(2, 2, figsize=(9, 7.5), constrained_layout=True) # Two rows, two columns
    
    # Top-Left Panel: Raw Signal Input
    axs[0,0].plot(time_array[llimit:ulimit], signal[llimit:ulimit], label='Raw Signal (G)', color='green')
    axs[0,0].set_title(f"Raw Signal Time Series ({economy_name})")
    axs[0,0].set_xlabel("Time")
    axs[0,0].set_ylabel("Signal Value (G)")
    axs[0,0].legend()
    axs[0,0].grid(True, alpha=0.3)

    # Top-Right Panel: Raw Employment and Vacancies
    axs[0,1].plot(time_array[llimit:ulimit], employment[llimit:ulimit]-sigma, label=r'Raw Employment $(e-\sigma)$', color='darkblue')
    axs[0,1].plot(time_array[llimit:ulimit], vacancies[llimit:ulimit], label='Raw Vacancies (v)', color='darkred', linestyle='--')
    axs[0,1].set_title(f"Raw Employment and Vacancies Time Series ({economy_name})")
    axs[0,1].set_xlabel("Time")
    axs[0,1].set_ylabel("Count")
    axs[0,1].legend()
    axs[0,1].grid(True, alpha=0.3)

    # Bottom-Left Panel: Normalized Signal, Employment, and Vacancies
    # Normalize data by subtracting the temporal average, then dividing by the de-meaned max

    # Process signal
    signal_demeaned = signal[llimit:ulimit] - np.mean(signal[llimit:ulimit])
    normalized_signal = signal_demeaned / np.max(np.abs(signal_demeaned))

    # Process employment
    employment_demeaned = employment[llimit:ulimit] - np.mean(employment[llimit:ulimit])
    normalized_employment = employment_demeaned / np.max(np.abs(employment_demeaned))

    # Process vacancies
    vacancies_demeaned = vacancies[llimit:ulimit] - np.mean(vacancies[llimit:ulimit])
    normalized_vacancies = vacancies_demeaned / np.max(np.abs(vacancies_demeaned))
    
    axs[1,0].plot(time_array[llimit:ulimit], normalized_signal, label='Signal (G)', color='purple', linestyle='-.')
    axs[1,0].plot(time_array[llimit:ulimit], normalized_employment, label='Employment (e)', color='darkblue')
    axs[1,0].plot(time_array[llimit:ulimit], normalized_vacancies, label='Vacancies (v)', color='darkred', linestyle='--')
    
    axs[1,0].set_title(f"Normalized Levels Time Series ({economy_name})")
    axs[1,0].set_xlabel("Time")
    axs[1,0].set_ylabel("Normalized Value"+r'$(y - E_t[y]) / (|y - E_t[y]|)_{max}$')
    axs[1,0].legend()
    axs[1,0].grid(True, alpha=0.3)

    # Bottom-Right Panel: Growth Rates
    axs[1,1].plot(time_points[llimit:ulimit], sig_growth[llimit:ulimit], label='Signal Growth (1+CG)', color='green', alpha=0.5, linestyle='--')
    axs[1,1].plot(time_points[llimit:ulimit], emp_growth[llimit:ulimit], label='Empirical Emp. Growth', color='blue', linewidth=1.5)
    axs[1,1].plot(time_points[llimit:ulimit], theo_emp_growth[llimit:ulimit], label='Theoretical Emp. Growth', color='red', linestyle=':', linewidth=2)
    
    axs[1,1].set_title(f"Growth Rates Time Series ({economy_name})")
    axs[1,1].set_xlabel("Time")
    axs[1,1].set_ylabel("Growth Rate (1/dt * ln(x_{t+1}/x_t))")
    axs[1,1].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axs[1,1].legend()
    axs[1,1].grid(True, alpha=0.3)
    
    filename = f"growth_rates_timeseries_{economy_name}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close(fig) # Use fig.close() when using subplots
    print(f"Saved growth rates time series plot to {filepath}")
def plot_efficiency_heatmap(sigma, sensitivity_c, matching_rate, separation_rate, dt, economy_name, output_dir=None):
    """
    Generates a heatmap showing the employment efficiency measure as a function of current
    employment (e) and current signal (G).
    Efficiency is defined as (e - e_hat) / e_hat.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Define ranges for employment and signal - consistent with other heatmaps
    e_center = sigma
    offset = 30
    e_range = 0.15 * e_center + offset 
    e_values = np.linspace(e_center - e_range, e_center + e_range, 50)

    g_values = np.linspace(-1, 1, 50) 
    
    E_grid, G_grid = np.meshgrid(e_values, g_values)
    
    efficiency_grid = np.zeros_like(E_grid, dtype=float)
    
    for i in range(E_grid.shape[0]):
        for j in range(E_grid.shape[1]):
            e_t = E_grid[i, j]
            G_t = G_grid[i, j]
            
            e_hat_t = sigma * (1 + sensitivity_c * G_t)
            
            if e_hat_t != 0:
                efficiency_grid[i, j] = (e_t - e_hat_t) / e_hat_t
            else:
                efficiency_grid[i, j] = np.nan
                
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
    

    # Use a diverging colormap, centered at 0
    norm = mcolors.TwoSlopeNorm(vmin=efficiency_grid[np.isfinite(efficiency_grid)].min(), 
                                vcenter=0, 
                                vmax=efficiency_grid[np.isfinite(efficiency_grid)].max())
    
    im = ax.contourf(E_grid, G_grid, efficiency_grid, levels=50, cmap='RdBu_r', norm=norm)
    
    ax.set_xlabel("Employment (e)")
    ax.set_ylabel("Signal (G)")

    # Overlay equilibrium employment curve where efficiency is zero (e = e_hat)
    equilibrium_employment = sigma * (1 + sensitivity_c * g_values)
    ax.plot(equilibrium_employment, g_values, color='black', linestyle='--', linewidth=2, label='Equilibrium')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.colorbar(im, ax=ax, label=r"Efficiency Gap $\epsilon = (e - \hat{e}) / \hat{e}$", shrink=0.5)
    
    filename = f"efficiency_heatmap_{economy_name}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved efficiency heatmap plot to {filepath}")
def plot_matching_time_vs_K(K_current=None, u_min=0.05, u_max=0.15, output_dir=None):
    """
    Log-log plot of matching rate constant K (month⁻¹) vs average vacancy fill time,
    with one line per unemployment rate u.
    Rate-based matching: τ = 1/(K·u), independent of population L.
    Y-axis in nominal calendar units (1 time unit = 1 month).

    Overlays the operating range of the current configuration as a vertical
    bar at K = K_current spanning [τ(u_max), τ(u_min)].
    """
    if K_current is None:
        from beveridge import config as _cfg

        K_current = float(_cfg.MATCHING_RATE_CONSTANT[0])
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    K_values = np.logspace(np.log10(0.1), np.log10(100), 200)
    u_values = [0.02, 0.05, 0.10, 0.20]

    HOURS_PER_MONTH = 30.44 * 24
    DAYS_PER_MONTH  = 30.44
    WEEKS_PER_MONTH = 30.44 / 7

    named_ticks = [
        (1   / HOURS_PER_MONTH,  '1 hr'),
        (8   / HOURS_PER_MONTH,  '8 hrs'),
        (1   / DAYS_PER_MONTH,   '1 day'),
        (1   / WEEKS_PER_MONTH,  '1 wk'),
        (1,                       '1 mo'),
        (3,                       '3 mo'),
        (6,                       '6 mo'),
        (12,                      '1 yr'),
        (60,                      '5 yr'),
        (120,                     '10 yr'),
        (600,                     '50 yr'),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / (len(u_values) - 1)) for i in range(len(u_values))]

    y_min, y_max = np.inf, -np.inf
    for u, color in zip(u_values, colors):
        matching_time = 1.0 / (K_values * u)
        ax.plot(K_values, matching_time, color=color, linewidth=2,
                label=f'u = {u:.0%}')
        y_min = min(y_min, matching_time.min())
        y_max = max(y_max, matching_time.max())

    # Current config operating range
    tau_slow = 1.0 / (K_current * u_min)  # boom → slowest fill
    tau_fast = 1.0 / (K_current * u_max)  # recession → fastest fill
    ax.plot([K_current, K_current], [tau_fast, tau_slow],
            color='red', linewidth=2.5, zorder=5)
    ax.plot(K_current, tau_slow, 'o', color='red', markersize=8, zorder=6,
            label=f'Current cfg (u={u_min:.0%})')
    ax.plot(K_current, tau_fast, 's', color='red', markersize=8, zorder=6,
            label=f'Current cfg (u={u_max:.0%})')

    ax.set_xscale('log')
    ax.set_yscale('log')

    margin = 10
    visible = [(pos, lbl) for pos, lbl in named_ticks
               if y_min / margin <= pos <= y_max * margin]
    if visible:
        tick_pos, tick_lbl = zip(*visible)
        ax.set_yticks(list(tick_pos))
        ax.set_yticklabels(list(tick_lbl))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_xlabel('Matching Rate Constant  K  (month⁻¹)')
    ax.set_ylabel('Average Vacancy Fill Time')
    ax.set_title('Matching Time  τ = 1/(K·u)  vs  K')
    ax.legend(title='Unemployment Rate')
    ax.grid(True, which='both', alpha=0.3)

    filepath = os.path.join(output_dir, 'matching_time_vs_K.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved matching time plot to {filepath}")


def plot_matching_time_vs_urate(K_current=None, u_min=0.05, u_max=0.15, output_dir=None):
    """
    Log-scale plot of unemployment rate vs matching time for selected K values.
    Rate-based matching: τ = 1/(K·u), independent of population L.

    Overlays the operating range of the current configuration as two dots
    (at u_min and u_max) on the K_current line, connected by a bar.
    """
    if K_current is None:
        from beveridge import config as _cfg

        K_current = float(_cfg.MATCHING_RATE_CONSTANT[0])
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    u_rates = np.linspace(0.01, 0.20, 200)

    K_specs = [
        (1.0 / 6.0, "K = 1/6"),
        (K_current, f"K = {K_current:g} (current)"),
        (6.67, "K = 6.67 (3-mo fill @ u=5%)"),
        (20.0, "K = 20"),
    ]

    HOURS_PER_MONTH = 30.44 * 24
    DAYS_PER_MONTH  = 30.44
    WEEKS_PER_MONTH = 30.44 / 7

    named_ticks = [
        (1   / HOURS_PER_MONTH,  '1 hr'),
        (8   / HOURS_PER_MONTH,  '8 hrs'),
        (1   / DAYS_PER_MONTH,   '1 day'),
        (1   / WEEKS_PER_MONTH,  '1 wk'),
        (1,                       '1 mo'),
        (3,                       '3 mo'),
        (6,                       '6 mo'),
        (12,                      '1 yr'),
        (60,                      '5 yr'),
        (120,                     '10 yr'),
        (600,                     '50 yr'),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c']

    y_min, y_max = np.inf, -np.inf
    for (K, label), color in zip(K_specs, colors):
        tau = 1.0 / (K * u_rates)
        ax.plot(u_rates * 100, tau, color=color, linewidth=2, label=label)
        y_min = min(y_min, tau.min())
        y_max = max(y_max, tau.max())

    # Current config operating bounds on the K_current line
    tau_at_umin = 1.0 / (K_current * u_min)   # boom → slowest fill
    tau_at_umax = 1.0 / (K_current * u_max)   # recession → fastest fill
    ax.plot([u_min * 100, u_max * 100], [tau_at_umin, tau_at_umax],
            color='red', linewidth=2.5, zorder=5)
    ax.plot(u_min * 100, tau_at_umin, 'o', color='red', markersize=8, zorder=6,
            label=f'Boom bound (u={u_min:.0%})')
    ax.plot(u_max * 100, tau_at_umax, 's', color='red', markersize=8, zorder=6,
            label=f'Recession bound (u={u_max:.0%})')

    ax.set_yscale('log')
    ax.set_xlabel('Unemployment Rate  (%)')
    ax.set_ylabel('Average Vacancy Fill Time')
    ax.set_title('Matching Time  τ = 1/(K·u)  vs  Unemployment Rate')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    margin = 10
    visible = [(pos, lbl) for pos, lbl in named_ticks
               if y_min / margin <= pos <= y_max * margin]
    if visible:
        tick_pos, tick_lbl = zip(*visible)
        ax.set_yticks(list(tick_pos))
        ax.set_yticklabels(list(tick_lbl))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())

    filepath = os.path.join(output_dir, 'matching_time_vs_urate.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved matching time vs u-rate plot to {filepath}")
