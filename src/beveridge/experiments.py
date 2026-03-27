"""Simulation runners, sweeps, and experiment entrypoints used by main."""

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAX_LAG_STEPS = 50

from plotting.rates import compute_rates
from plotting.beveridge_figs import (
    plot_aggregate_overview,
    plot_beveridge_comparison,
    plot_efficiency_time_series,
    plot_multi_employment,
    plot_response_to_demand,
    plot_vacancy_time_series,
)
from plotting.gdp_shock_response import (
    default_shock_g_grid,
    fit_power_law_decay_window,
    half_life_time_in_window,
    plot_gdp_shock_response_figure,
)
from plotting.uv_crosscorr import (
    plot_uv_delta_levels_crosscorr_from_series,
    plot_uv_lag_experiment_from_series,
)
from plotting.c_sweep_figs import plot_firm_sensitivity_distribution
from plotting.diagnostics import (
    plot_efficiency_heatmap,
    plot_fluctuation_analysis,
    plot_growth_rate_analysis,
    plot_growth_rate_heatmaps,
    plot_growth_rates_time_series,
    plot_matching_time_vs_K,
    plot_matching_time_vs_urate,
)
from time_grid import DT, STEPS, TIME, mid_run_plot_slice

from . import config
from .economy import check_stability, initialize_economy, run_market
from .firm import Firm
from .signals import GDP


def run_simulation_for_signal(signal_name, economy_name_suffix, output_dir=None):
    print(f"Running single simulation for signal: {signal_name}...")

    firm_weights_k = [1]
    base_sigma = config.BASE_SIGMA[0]
    sigmas, population, _ = initialize_economy(
        GDP[signal_name],
        firm_weights_k,
        base_sigma,
        config.MATCHING_RATE_CONSTANT[0],
        config.PRODUCTIVITY_DENSITY[0],
        config.SENSITIVITY_COEFFICIENT[0],
    )

    initial_signal = GDP[signal_name].iloc[0]
    init_employment = sigmas[0] * (1 + config.SENSITIVITY_COEFFICIENT[0] * initial_signal)
    init_unemployment = population - init_employment

    firm = Firm(
        GDP[signal_name],
        sigmas[0],
        config.PRODUCTIVITY_DENSITY[0],
        init_employment,
        config.INIT_VACANCIES[0],
        config.MATCHING_RATE_CONSTANT[0],
        config.SENSITIVITY_COEFFICIENT[0],
        idio_std=0.0,
    )
    firm.set_time(TIME)

    unemployment, firms = run_market([firm], population, init_unemployment)

    shock_idx = 40
    debug_window = 5
    print("\n" + "=" * 60)
    print(f"DEBUG DATA AROUND SHOCK (t={shock_idx})")
    print("=" * 60)
    print(f"{'Time Step':<10} | {'Signal':<10} | {'Employment':<12} | {'Vacancies':<10} | {'Unemployment':<12} | {'Growth Rate %':<15}")
    print("-" * 85)

    for t in range(shock_idx - debug_window, shock_idx + debug_window + 1):
        if 0 <= t < len(TIME):
            emp = firms[0].employment[t]
            vac = firms[0].vacancies[t]
            sig = GDP[signal_name].iloc[t] if t < len(GDP[signal_name]) else 0
            unemp = unemployment[t]

            if t < len(TIME) - 1 and emp > 0:
                growth = (firms[0].employment[t + 1] - emp) / emp * 100
            else:
                growth = 0.0

            print(f"{t:<10} | {sig:<10.2f} | {emp:<12.2f} | {vac:<10.2f} | {unemp:<12.2f} | {growth:<15.2f}")

    print("=" * 60 + "\n")

    economy_name = f"single_firm_{economy_name_suffix}"
    plot_multi_employment(firms, unemployment, TIME, population, economy_name, output_dir=output_dir)
    vacancy_rate, unemployment_rate = compute_rates(
        firms, economy_name, population, plot=True, output_dir=output_dir, burn_in=config.BURN_IN
    )

    aggregate_vacancies = [np.sum([firm.vacancies[i] for firm in firms]) for i in range(len(TIME))]
    plot_response_to_demand(GDP[signal_name], unemployment, aggregate_vacancies, economy_name, population, output_dir=output_dir)

    show_trace = signal_name == "gdp_sine"
    plot_fluctuation_analysis(
        GDP[signal_name],
        unemployment,
        aggregate_vacancies,
        economy_name,
        population,
        output_dir=output_dir,
        show_trace=show_trace,
        burn_in=config.BURN_IN,
        firms=firms,
    )

    uv_ccf = plot_uv_lag_experiment_from_series(
        unemployment,
        aggregate_vacancies,
        population,
        burn_in=config.BURN_IN,
        max_lag_steps=MAX_LAG_STEPS,
        transform="demean_rates",
        output_dir=output_dir,
        economy_name=economy_name,
    )
    uv_ccf_d = plot_uv_delta_levels_crosscorr_from_series(
        unemployment,
        aggregate_vacancies,
        burn_in=config.BURN_IN,
        max_lag_steps=MAX_LAG_STEPS,
        output_dir=output_dir,
        economy_name=economy_name,
    )

    print(f"\nGenerated single-run plots for {signal_name}:")
    print(f"- employment_{economy_name}.pdf")
    print(f"- beveridge_{economy_name}.pdf")
    print(f"- response_to_demand_{economy_name}.pdf")
    print(f"- fluctuation_analysis_{economy_name}.pdf")
    print(f"- {os.path.basename(uv_ccf['output_path'])}")
    print(f"- {os.path.basename(uv_ccf_d['output_path'])}")

    return vacancy_rate, unemployment_rate


def run_single_timeseries(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, "single_timeseries_plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print("Running single simulation to generate time series plots...")

    if "gdp_ar2" in GDP.columns and not np.allclose(GDP["gdp_ar2"], 0):
        signal_name = "gdp_ar2"
        print("Using AR(2) signal for simulation.")
    else:
        signal_name = "gdp_constant"
        print("AR(2) signal not available or zero; using sine wave signal for simulation.")

    firm_weights_k = [1]
    base_sigma = config.BASE_SIGMA[0]
    sigmas, population, _ = initialize_economy(
        GDP[signal_name],
        firm_weights_k,
        base_sigma,
        config.MATCHING_RATE_CONSTANT[0],
        config.PRODUCTIVITY_DENSITY[0],
        config.SENSITIVITY_COEFFICIENT[0],
    )

    K = config.MATCHING_RATE_CONSTANT[0]
    C = config.SENSITIVITY_COEFFICIENT
    signal = GDP[signal_name]
    if not check_stability(K, population, sigmas, C, signal):
        return None, None

    initial_signal = GDP[signal_name].iloc[0]
    init_employment = sigmas[0] * (1 + config.SENSITIVITY_COEFFICIENT[0] * initial_signal)
    init_unemployment = population - init_employment

    config.INIT_UNEMPLOYMENT = init_unemployment
    config.POPULATION = population

    firm = Firm(
        GDP[signal_name],
        sigmas[0],
        config.PRODUCTIVITY_DENSITY[0],
        init_employment,
        0,
        config.MATCHING_RATE_CONSTANT[0],
        config.SENSITIVITY_COEFFICIENT[0],
        idio_std=0.0,
    )
    firm.set_time(TIME)

    unemployment, firms = run_market([firm], population, init_unemployment)

    plot_multi_employment(firms, unemployment, TIME, population, "single_firm", output_dir=output_dir)
    vacancy_rate, unemployment_rate = compute_rates(
        firms, "single", population, plot=True, output_dir=output_dir, burn_in=config.BURN_IN
    )

    aggregate_vacancies = [np.sum([firm.vacancies[i] for firm in firms]) for i in range(len(TIME))]

    total_employment = [np.sum([firm.employment[i] for firm in firms]) for i in range(len(TIME))]

    plot_response_to_demand(GDP[signal_name], unemployment, aggregate_vacancies, "single_firm", population, output_dir=output_dir)
    plot_fluctuation_analysis(
        GDP[signal_name],
        unemployment,
        aggregate_vacancies,
        "single_firm",
        population,
        output_dir=output_dir,
        show_trace=True,
        burn_in=config.BURN_IN,
        firms=firms,
    )

    uv_ccf = plot_uv_lag_experiment_from_series(
        unemployment,
        aggregate_vacancies,
        population,
        burn_in=config.BURN_IN,
        max_lag_steps=MAX_LAG_STEPS,
        transform="demean_rates",
        output_dir=output_dir,
        economy_name="single_firm",
    )
    uv_ccf_d = plot_uv_delta_levels_crosscorr_from_series(
        unemployment,
        aggregate_vacancies,
        burn_in=config.BURN_IN,
        max_lag_steps=MAX_LAG_STEPS,
        output_dir=output_dir,
        economy_name="single_firm",
    )

    plot_growth_rate_analysis(
        GDP[signal_name],
        total_employment,
        aggregate_vacancies,
        "single_firm",
        config.MATCHING_RATE_CONSTANT[0],
        config.SEPARATION_RATE,
        sigmas[0],
        config.SENSITIVITY_COEFFICIENT[0],
        dt=DT,
        output_dir=output_dir,
    )

    plot_growth_rates_time_series(
        GDP[signal_name],
        total_employment,
        aggregate_vacancies,
        "single_firm",
        config.MATCHING_RATE_CONSTANT[0],
        config.SEPARATION_RATE,
        sigmas[0],
        config.SENSITIVITY_COEFFICIENT[0],
        dt=DT,
        time_array=TIME,
        output_dir=output_dir,
    )

    plot_growth_rate_heatmaps(
        sigmas[0],
        config.SENSITIVITY_COEFFICIENT[0],
        config.MATCHING_RATE_CONSTANT[0],
        config.SEPARATION_RATE,
        DT,
        "single_firm",
        output_dir=output_dir,
    )

    plot_efficiency_heatmap(
        sigmas[0],
        config.SENSITIVITY_COEFFICIENT[0],
        config.MATCHING_RATE_CONSTANT[0],
        config.SEPARATION_RATE,
        DT,
        "single_firm",
        output_dir=output_dir,
    )

    plot_efficiency_time_series(firms, TIME, "single_firm", output_dir=output_dir)

    plot_aggregate_overview(GDP[signal_name], firms, unemployment, TIME, population, "single_firm", output_dir=output_dir)
    plot_vacancy_time_series(firms, TIME, "single_firm", output_dir=output_dir)

    print("\nGenerated single-run plots:")
    print(f"- {os.path.join(output_dir, 'employment_single_firm.pdf')}")
    print(f"- {os.path.join(output_dir, 'beveridgesingle.pdf')}")
    print(f"- {os.path.join(output_dir, 'response_to_demand_single_firm.pdf')}")
    print(f"- {os.path.join(output_dir, 'fluctuation_analysis_single_firm.pdf')}")
    print(f"- {uv_ccf['output_path']}")
    print(f"- {uv_ccf_d['output_path']}")
    print(f"- {os.path.join(output_dir, 'growth_rate_analysis_single_firm.pdf')}")
    print(f"- {os.path.join(output_dir, 'growth_rates_timeseries_single_firm.pdf')}")
    print(f"- {os.path.join(output_dir, 'growth_rate_heatmaps_single_firm.pdf')}")
    print(f"- {os.path.join(output_dir, 'efficiency_timeseries_single_firm.pdf')}")
    print(f"- {os.path.join(output_dir, 'efficiency_heatmap_single_firm.pdf')}")

    return vacancy_rate, unemployment_rate


def run_uv_crosscorr_experiment(output_dir=None):
    """
    Run a single-firm simulation and write only the U–V cross-correlation figure.

    For the full plot suite, use ``run_single_timeseries`` (which also calls the CCF).
    """
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, "uv_ccf")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    if "gdp_ar2" in GDP.columns and not np.allclose(GDP["gdp_ar2"], 0):
        signal_name = "gdp_ar2"
        print("Using AR(2) signal for UV cross-correlation run.")
    else:
        signal_name = "gdp_constant"
        print("AR(2) unavailable; using gdp_constant for UV cross-correlation run.")

    firm_weights_k = [1]
    base_sigma = config.BASE_SIGMA[0]
    sigmas, population, _ = initialize_economy(
        GDP[signal_name],
        firm_weights_k,
        base_sigma,
        config.MATCHING_RATE_CONSTANT[0],
        config.PRODUCTIVITY_DENSITY[0],
        config.SENSITIVITY_COEFFICIENT[0],
    )

    K = config.MATCHING_RATE_CONSTANT[0]
    C = config.SENSITIVITY_COEFFICIENT
    signal = GDP[signal_name]
    if not check_stability(K, population, sigmas, C, signal):
        print("Stability check failed; aborting UV cross-correlation run.")
        return None

    initial_signal = GDP[signal_name].iloc[0]
    init_employment = sigmas[0] * (1 + config.SENSITIVITY_COEFFICIENT[0] * initial_signal)
    init_unemployment = population - init_employment

    firm = Firm(
        GDP[signal_name],
        sigmas[0],
        config.PRODUCTIVITY_DENSITY[0],
        init_employment,
        0,
        config.MATCHING_RATE_CONSTANT[0],
        config.SENSITIVITY_COEFFICIENT[0],
        idio_std=0.0,
    )
    firm.set_time(TIME)

    unemployment, firms = run_market([firm], population, init_unemployment)
    aggregate_vacancies = [np.sum([firm.vacancies[i] for firm in firms]) for i in range(len(TIME))]

    uv_ccf = plot_uv_lag_experiment_from_series(
        unemployment,
        aggregate_vacancies,
        population,
        burn_in=config.BURN_IN,
        max_lag_steps=MAX_LAG_STEPS,
        transform="demean_rates",
        output_dir=output_dir,
        economy_name="single_firm",
    )
    uv_ccf_d = plot_uv_delta_levels_crosscorr_from_series(
        unemployment,
        aggregate_vacancies,
        burn_in=config.BURN_IN,
        max_lag_steps=MAX_LAG_STEPS,
        output_dir=output_dir,
        economy_name="single_firm",
    )
    print(f"Saved UV cross-correlation plot to {uv_ccf['output_path']}")
    print(f"Saved UV delta cross-correlation plot to {uv_ccf_d['output_path']}")
    return {"rates": uv_ccf, "deltas": uv_ccf_d}


def compare_sine_special(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, "sine_special_comparison")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print("Running single-firm simulations for signal comparison...")

    vr_custom, ur_custom = run_simulation_for_signal("gdp_custom", "custom_signal", output_dir=output_dir)

    vr_sine, ur_sine = run_simulation_for_signal("gdp_sine", "sine_signal", output_dir=output_dir)

    comparison_data = {
        "Custom Signal": (vr_custom, ur_custom),
        "Sine Wave Signal": (vr_sine, ur_sine),
    }
    plot_beveridge_comparison(comparison_data, filename=os.path.join(output_dir, "beveridge_signal_comparison.pdf"))


def run_ar2_experiment(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, "ar2_experiment_plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print("Running AR(2) experiment...")

    if "gdp_ar2" in GDP.columns and not np.all(GDP["gdp_ar2"] == 0):
        run_simulation_for_signal("gdp_ar2", "ar2", output_dir=output_dir)
    else:
        print("Skipping AR(2) experiment because the signal was not loaded.")


def run_sensitivity_sweep(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, "sensitivity_sweep_plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    c_values_sweep = [0.05, 0.1, 0.2, 0.5, 1.0]
    fixed_k = config.MATCHING_RATE_CONSTANT[0]

    print(f"Running Sensitivity (C) Sweep: {c_values_sweep}")

    results = []

    for c in c_values_sweep:
        print(f"  Running for C = {c}...")
        signal_series = GDP["gdp_custom"]

        firm_weights = [1]
        base_sigma = config.BASE_SIGMA[0]

        sigmas, population, _ = initialize_economy(
            signal_series, firm_weights, base_sigma, fixed_k, config.PRODUCTIVITY_DENSITY[0], [c]
        )

        initial_signal = signal_series.iloc[0]
        init_employment = sigmas[0] * (1 + c * initial_signal)
        init_unemployment = population - init_employment

        firm = Firm(signal_series, sigmas[0], config.PRODUCTIVITY_DENSITY[0], init_employment, 0, fixed_k, c, idio_std=0.0)
        firm.set_time(TIME)

        unemployment, firms = run_market([firm], population, init_unemployment)

        vacancy_rate, unemployment_rate = compute_rates(firms, "temp", population, plot=False)
        vr_window = vacancy_rate[STEPS // 2 :]
        ur_window = unemployment_rate[STEPS // 2 :]

        efficiency = np.array(firms[0].efficiency)

        results.append({"parameter": c, "efficiency": efficiency, "vr": vr_window, "ur": ur_window})

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    fig.suptitle(f"Parameter Sweep 1: Sensitivity Coefficient (C) (K={fixed_k:.2f})", fontsize=16)

    llimit, ulimit = mid_run_plot_slice()
    time_slice = TIME[llimit:ulimit]

    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=min(c_values_sweep), vmax=max(c_values_sweep) * 1.25)

    for res in results:
        c_val = res["parameter"]
        eff_slice = res["efficiency"][llimit:ulimit]
        axs[0].plot(time_slice, eff_slice, label=f"C={c_val}", color=cmap(norm(c_val)), linewidth=2)

    axs[0].set_title("Employment Efficiency Gap")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel(r"$\epsilon = (e - \hat{e}) / \hat{e}$")
    axs[0].set_ylim(-0.15, 0.05)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    for res in results:
        c_val = res["parameter"]
        axs[1].plot(res["ur"], res["vr"], label=f"C={c_val}", color=cmap(norm(c_val)))

    axs[1].set_title("Beveridge Curve (Steady State)")
    axs[1].set_xlabel("Unemployment Rate")
    axs[1].set_ylabel("Vacancy Rate")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    filename = os.path.join(output_dir, "sweep_1_sensitivity_C.pdf")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved Sweep 1 plot to {filename}")


def run_matching_rate_sweep(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, "matching_rate_sweep_plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Scale K with DT (same relative spread as pre–DT=1 sweep at DT=0.1)
    k_values_sweep = [0.1 * x for x in (1.0 / 20.0, 1.0 / 10.0, 1.0 / 6.0, 1.0 / 2.0, 1.0)]
    fixed_c = 0.1

    print(f"Running Matching Rate (K) Sweep: {k_values_sweep}")

    results = []

    for k in k_values_sweep:
        print(f"  Running for K = {k:.2f}...")
        signal_series = GDP["gdp_custom"]

        firm_weights = [1]
        base_sigma = config.BASE_SIGMA[0]

        sigmas, population, _ = initialize_economy(
            signal_series, firm_weights, base_sigma, k, config.PRODUCTIVITY_DENSITY[0], [fixed_c]
        )

        initial_signal = signal_series.iloc[0]
        init_employment = sigmas[0] * (1 + fixed_c * initial_signal)
        init_unemployment = population - init_employment

        firm = Firm(signal_series, sigmas[0], config.PRODUCTIVITY_DENSITY[0], init_employment, 0, k, fixed_c, idio_std=0.0)
        firm.set_time(TIME)

        unemployment, firms = run_market([firm], population, init_unemployment)

        vacancy_rate, unemployment_rate = compute_rates(firms, "temp", population, plot=False)
        vr_window = vacancy_rate[STEPS // 2 :]
        ur_window = unemployment_rate[STEPS // 2 :]

        efficiency = np.array(firms[0].efficiency)

        results.append({"parameter": k, "efficiency": efficiency, "vr": vr_window, "ur": ur_window})

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    fig.suptitle(f"Parameter Sweep 2: Matching Rate (K) (C={fixed_c})", fontsize=16)

    llimit, ulimit = mid_run_plot_slice()
    time_slice = TIME[llimit:ulimit]

    cmap = plt.get_cmap("plasma")
    norm = mcolors.LogNorm(vmin=min(k_values_sweep), vmax=max(k_values_sweep) * 1.25)

    for res in results:
        k_val = res["parameter"]
        eff_slice = res["efficiency"][llimit:ulimit]
        axs[0].plot(time_slice, eff_slice, label=f"K={k_val:.2f}", color=cmap(norm(k_val)), linewidth=2)

    axs[0].set_title("Employment Efficiency Gap")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel(r"$\epsilon = (e - \hat{e}) / \hat{e}$")
    axs[0].set_ylim(-0.15, 0.05)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    for res in results:
        k_val = res["parameter"]
        axs[1].plot(res["ur"], res["vr"], label=f"K={k_val:.2f}", color=cmap(norm(k_val)))

    axs[1].set_title("Beveridge Curve (Steady State)")
    axs[1].set_xlabel("Unemployment Rate")
    axs[1].set_ylabel("Vacancy Rate")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    filename = os.path.join(output_dir, "sweep_2_matching_K.pdf")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved Sweep 2 plot to {filename}")


def _resolve_sensitivity_coefficients(num_firms, c_distribution_method, c_params):
    if c_distribution_method == "uniform":
        return [c_params["value"]] * num_firms
    if c_distribution_method == "sampled":
        return np.random.uniform(c_params["min"], c_params["max"], num_firms).tolist()
    if c_distribution_method == "functional":
        if "values" in c_params and len(c_params["values"]) == num_firms:
            return c_params["values"]
        raise ValueError(
            f"For 'functional' C distribution, 'values' must be provided and match num_firms. "
            f"Provided: {c_params.get('values')} for {num_firms} firms"
        )
    if c_distribution_method == "power_law":
        c_max = c_params.get("c_max", 0.2)
        exponent = c_params.get("exponent", 2.0)
        zero_fraction = c_params.get("zero_fraction", 0.25)
        seed = c_params.get("seed", 42)
        flip = bool(c_params.get("power_law_flip", False))
        rng = np.random.default_rng(seed)

        max_zero = int(
            c_params.get("power_law_max_zero_firms", config.POWER_LAW_MAX_ZERO_FIRMS)
        )
        max_zero = max(0, min(max_zero, num_firms))
        n_zero = min(int(np.round(zero_fraction * num_firms)), max_zero)
        n_power = num_firms - n_zero

        u = rng.uniform(0, 1, n_power)
        t = u ** (1.0 / exponent)
        if flip:
            # F(c) = 1 - (1 - c/c_max)^α on (0,c_max]: density falls toward c_max for α > 1
            c_power = c_max * (1.0 - t)
        else:
            # F(c) = (c/c_max)^α: density rises toward c_max for α > 1
            c_power = c_max * t

        return np.concatenate([np.zeros(n_zero), np.sort(c_power)]).tolist()
    raise ValueError(f"Unknown C distribution method: {c_distribution_method}")


def resolve_sensitivity_coefficients(num_firms, c_distribution_method, c_params):
    """
    Firm-level sensitivity vector ``C`` (same construction as multi-firm simulation).

    For ``power_law``, optional ``c_params["power_law_flip"]`` (default False):
    if True, use ``C = c_max * (1 - U^{1/\\alpha})`` instead of ``c_max * U^{1/\\alpha}``.
    Optional ``c_params["power_law_max_zero_firms"]`` caps how many firms have
    ``C=0`` (default :data:`config.POWER_LAW_MAX_ZERO_FIRMS`).

    Returns a 1D ``numpy`` array (float).
    """
    return np.asarray(
        _resolve_sensitivity_coefficients(num_firms, c_distribution_method, c_params),
        dtype=float,
    )


def _simulate_multi_firm(num_firms, c_distribution_method, c_params, signal_name=None):
    """
    Build firms, run market dynamics, return aggregate series (no plotting).

    Returns dict with unemployment, aggregate_vacancies, population, firms,
    sensitivity_coefficients, signal_series.
    """
    if signal_name is None:
        signal_name = config.GDP_SIGNAL_NAME
    if signal_name not in GDP.columns:
        raise ValueError(f"Unknown GDP signal column {signal_name!r}; available: {list(GDP.columns)}")

    sensitivity_coefficients = resolve_sensitivity_coefficients(
        num_firms, c_distribution_method, c_params
    ).tolist()

    firm_weights_k = [1.0] * num_firms
    base_sigma = config.BASE_SIGMA[0]
    fixed_k = config.MATCHING_RATE_CONSTANT[0]

    signal_series = GDP[signal_name]
    sigmas, population, _ = initialize_economy(
        signal_series, firm_weights_k, base_sigma, fixed_k, config.PRODUCTIVITY_DENSITY[0], sensitivity_coefficients
    )

    initial_signal = signal_series.iloc[0]
    firms = []
    total_initial_employment = 0
    for i in range(num_firms):
        init_emp = sigmas[i] * (1 + sensitivity_coefficients[i] * initial_signal)
        total_initial_employment += init_emp
        firm = Firm(
            signal_series,
            sigmas[i],
            config.PRODUCTIVITY_DENSITY[0],
            init_emp,
            0,
            fixed_k,
            sensitivity_coefficients[i],
            seed=1000 + i,
        )
        firm.set_time(TIME)
        firms.append(firm)

    init_unemployment = population - total_initial_employment
    unemployment, firms = run_market(firms, population, init_unemployment)
    aggregate_vacancies = [np.sum([firm.vacancies[i] for firm in firms]) for i in range(len(TIME))]

    return {
        "unemployment": unemployment,
        "aggregate_vacancies": aggregate_vacancies,
        "population": population,
        "firms": firms,
        "sensitivity_coefficients": sensitivity_coefficients,
        "signal_series": signal_series,
    }


def _format_multi_firm_c_suptitle(num_firms, signal_name, c_distribution_method, c_params):
    parts = [f"{num_firms} firms", f"signal={signal_name}", str(c_distribution_method)]
    if c_distribution_method == "power_law":
        parts.extend(
            [
                f"c_max={c_params.get('c_max')}",
                f"exponent={c_params.get('exponent')}",
                f"zero_fraction={c_params.get('zero_fraction')}",
                f"seed={c_params.get('seed')}",
            ]
        )
        if c_params.get("power_law_flip"):
            parts.append("C=c_max(1-U^(1/α))")
        else:
            parts.append("C=c_max·U^(1/α)")
        mz = int(
            c_params.get("power_law_max_zero_firms", config.POWER_LAW_MAX_ZERO_FIRMS)
        )
        parts.append(f"C=0 cap={mz} firms")
    elif c_distribution_method == "uniform":
        parts.append(f"C={c_params.get('value')}")
    elif c_distribution_method == "sampled":
        parts.append(f"C~U[{c_params.get('min')},{c_params.get('max')}]")
    elif c_distribution_method == "functional":
        parts.append("C from provided values")
    return " · ".join(parts)


def run_multi_firm_simulation(
    num_firms,
    c_distribution_method,
    c_params,
    economy_name,
    output_dir=None,
    signal_name=None,
):
    """
    Run multi-firm dynamics and write plots under ``output_dir``.

    Args:
        signal_name: Column of ``GDP`` to use (e.g. ``gdp_sine``, ``gdp_ar2``).
            Defaults to ``config.GDP_SIGNAL_NAME``.
    """
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, "multi_firm_plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    if signal_name is None:
        signal_name = config.GDP_SIGNAL_NAME
    if signal_name not in GDP.columns:
        raise ValueError(f"Unknown GDP signal column {signal_name!r}; available: {list(GDP.columns)}")

    print(
        f"Running multi-firm simulation with {num_firms} firms, "
        f"C distribution: {c_distribution_method}, signal: {signal_name}..."
    )

    sim = _simulate_multi_firm(num_firms, c_distribution_method, c_params, signal_name=signal_name)
    unemployment = sim["unemployment"]
    firms = sim["firms"]
    population = sim["population"]
    aggregate_vacancies = sim["aggregate_vacancies"]
    signal_series = sim["signal_series"]

    plot_multi_employment(firms, unemployment, TIME, population, economy_name, output_dir=output_dir)
    compute_rates(firms, economy_name, population, plot=True, output_dir=output_dir, burn_in=config.BURN_IN)
    plot_efficiency_time_series(firms, TIME, economy_name, output_dir=output_dir)
    plot_aggregate_overview(signal_series, firms, unemployment, TIME, population, economy_name, output_dir=output_dir)
    plot_vacancy_time_series(firms, TIME, economy_name, output_dir=output_dir)

    plot_fluctuation_analysis(
        signal_series,
        unemployment,
        aggregate_vacancies,
        economy_name,
        population,
        output_dir=output_dir,
        show_trace=False,
        burn_in=config.BURN_IN,
        firms=firms,
    )

    c_fig_path = os.path.join(output_dir, f"c_distribution_{economy_name}.pdf")
    plot_firm_sensitivity_distribution(
        sim["sensitivity_coefficients"],
        c_fig_path,
        suptitle=(
            r"Cross-sectional firm sensitivity $C$"
            + "\n"
            + _format_multi_firm_c_suptitle(
                num_firms, signal_name, c_distribution_method, c_params
            )
        ),
    )

    print(f"\nGenerated multi-firm plots for {economy_name}:")
    print(f"- employment_{economy_name}.pdf")
    print(f"- beveridge{economy_name}.pdf")
    print(f"- efficiency_timeseries_{economy_name}.pdf")
    print(f"- aggregate_overview_{economy_name}.pdf")
    print(f"- vacancy_timeseries_{economy_name}.pdf")
    print(f"- fluctuation_analysis_{economy_name}.pdf")
    print(f"- c_distribution_{economy_name}.pdf")


def _piecewise_g_signal(post_shock_level, burn_in_steps, n_steps=STEPS):
    """G = 0 for indices < burn_in_steps, else constant post_shock_level."""
    g = np.zeros(n_steps, dtype=float)
    g[burn_in_steps:] = post_shock_level
    return g


def run_gdp_shock_response_experiment(
    output_dir=None,
    burn_in_steps=None,
    post_shock_steps=1000,
    g_values=None,
    base_seed=4242,
):
    """
    GDP_shock_response: burn-in at G=0, step to constant g, then analyze U,V.

    Uses an envelope signal with G=-1 everywhere for ``initialize_economy`` so
    stability checks use the worst-case g in the default grid. Each run uses a
    piecewise G(t) with the actual path.

    Power-law fit (see plan): for τ = 1 … post_shock_steps-1,
    log|U(τ)-U_end| vs log(τ+1); same for V. End of window is U at τ=post_shock_steps.
    Relaxation near a fixed point is often exponential; low R² is expected.

    Half-life: first time (in simulation time) the series crosses the midpoint between
    the first and last sample in the post-shock window (see ``half_life_time_in_window``).

    Returns dict with arrays: g_values, U_window, V_window, beta_U, beta_V, r2_U, r2_V,
    half_life_U, half_life_V.
    """
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, "gdp_shock_response")
    os.makedirs(output_dir, exist_ok=True)

    if burn_in_steps is None:
        burn_in_steps = config.BURN_IN

    if g_values is None:
        g_values = default_shock_g_grid()

    i_shock = int(burn_in_steps)
    window_len = int(post_shock_steps) + 1
    i_end = i_shock + post_shock_steps
    if i_end >= STEPS:
        raise ValueError(
            f"Shock window exceeds TIME: i_shock={i_shock}, post_shock_steps={post_shock_steps}, STEPS={STEPS}"
        )

    g_min_envelope = min(float(np.min(g_values)), -1.0)
    envelope = np.full(STEPS, g_min_envelope, dtype=float)
    envelope_series = pd.Series(envelope)

    firm_weights_k = [1]
    base_sigma = config.BASE_SIGMA[0]
    fixed_k = config.MATCHING_RATE_CONSTANT[0]
    c = config.SENSITIVITY_COEFFICIENT[0]

    sigmas, population, _ = initialize_economy(
        envelope_series,
        firm_weights_k,
        base_sigma,
        fixed_k,
        config.PRODUCTIVITY_DENSITY[0],
        [c],
    )

    piecewise_for_stability = pd.Series(_piecewise_g_signal(min(g_values), burn_in_steps))
    if not check_stability(fixed_k, population, sigmas, [c], piecewise_for_stability):
        print("Warning: stability check failed for minimum g; results may be unreliable.")

    U_by_g = []
    V_by_g = []
    beta_U = []
    beta_V = []
    r2_U = []
    r2_V = []
    half_life_U = []
    half_life_V = []

    for k, g in enumerate(g_values):
        g = float(g)
        G_arr = _piecewise_g_signal(g, burn_in_steps)
        signal_series = pd.Series(G_arr)

        init_employment = sigmas[0] * (1 + c * 0.0)
        init_unemployment = population - init_employment

        firm = Firm(
            signal_series,
            sigmas[0],
            config.PRODUCTIVITY_DENSITY[0],
            init_employment,
            config.INIT_VACANCIES[0],
            fixed_k,
            c,
            idio_std=0.0,
            seed=base_seed + k,
        )
        firm.set_time(TIME)

        unemployment, firms = run_market([firm], population, init_unemployment)

        U_win = np.asarray(unemployment[i_shock : i_shock + window_len], dtype=float)
        V_win = np.asarray(
            [sum(f.vacancies[i] for f in firms) for i in range(i_shock, i_shock + window_len)],
            dtype=float,
        )

        bu, ru = fit_power_law_decay_window(U_win, tau0=1.0, tau_min=1, tau_max_exclusive=window_len - 1)
        bv, rv = fit_power_law_decay_window(V_win, tau0=1.0, tau_min=1, tau_max_exclusive=window_len - 1)

        hlu = half_life_time_in_window(U_win, DT)
        hlv = half_life_time_in_window(V_win, DT)

        U_by_g.append(U_win)
        V_by_g.append(V_win)
        beta_U.append(bu)
        beta_V.append(bv)
        r2_U.append(ru)
        r2_V.append(rv)
        half_life_U.append(hlu)
        half_life_V.append(hlv)

    tau_axis = np.arange(window_len, dtype=float) * DT
    out_path = os.path.join(output_dir, "gdp_shock_response_U_V.pdf")

    plot_gdp_shock_response_figure(
        g_values,
        tau_axis,
        U_by_g,
        V_by_g,
        np.array(beta_U),
        np.array(beta_V),
        np.array(half_life_U),
        np.array(half_life_V),
        r2_U=np.array(r2_U),
        r2_V=np.array(r2_V),
        output_path=out_path,
        title="GDP shock response: aggregate U, V, power-law exponents, and half-lives",
    )

    print(f"Saved GDP shock response figure to {out_path}")

    return {
        "g_values": list(g_values),
        "U_window": U_by_g,
        "V_window": V_by_g,
        "beta_U": np.array(beta_U),
        "beta_V": np.array(beta_V),
        "r2_U": np.array(r2_U),
        "r2_V": np.array(r2_V),
        "half_life_U": np.array(half_life_U),
        "half_life_V": np.array(half_life_V),
        "tau_axis": tau_axis,
        "burn_in_steps": burn_in_steps,
        "post_shock_steps": post_shock_steps,
        "figure_path": out_path,
    }
