"""Market clearing, initialization, and stability checks."""

import numpy as np

from time_grid import DT, TIME


def compute_loop_area(x, y):
    """
    Calculates the area of a polygon using the Shoelace formula.
    Assumes the x and y arrays represent the vertices of a closed loop.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def run_market(firms, population, init_unemployment):
    unemployment = [init_unemployment for _ in TIME]
    for i, t in enumerate(TIME[:-1]):
        for firm in firms:
            firm.update_vacancies(i, unemployment[i], population)
            firm.update_employment(i, DT)

        unemployment[i + 1] = population - np.sum([firm.employment[i + 1] for firm in firms])

    final_idx = len(TIME) - 1
    for firm in firms:
        firm.update_vacancies(final_idx, unemployment[final_idx], population)

    return unemployment, firms


def initialize_economy(
    demand_signal,
    firm_weights_k,
    base_sigma,
    matching_rate_k,
    productivity_density,
    sensitivity_coefficients,
    target_u_rate=0.05,
):
    """
    Initializes the economy's structural parameters.

    Population L is set so that baseline unemployment (at G=0) equals target_u_rate.
    Stability is verified as a post-hoc check, not used to determine L.
    """
    num_firms_N = len(firm_weights_k)
    k = np.asarray(firm_weights_k)
    C = np.asarray(sensitivity_coefficients)

    sigmas = k * base_sigma

    total_baseline_employment = np.sum(sigmas * (1 + C * 0.0))

    population_L = total_baseline_employment / (1.0 - target_u_rate)

    g_min = demand_signal.min()
    total_employment_at_min = np.sum(sigmas * (1 + C * g_min))
    u_max_level = population_L - total_employment_at_min
    u_max_rate = u_max_level / population_L
    stability_param = matching_rate_k * u_max_rate * DT

    print(f"Population: {population_L:.1f}")
    print(f"Baseline employment: {total_baseline_employment:.1f}")
    print(f"Baseline u_rate: {target_u_rate:.1%}")
    print(f"Max u_rate (at G_min={g_min:.3f}): {u_max_rate:.3f} ({u_max_level:.1f} workers)")
    print(f"Stability check: K*u_max*dt = {matching_rate_k:.2f}*{u_max_rate:.3f}*{DT} = {stability_param:.4f} (must be < 1)")
    if stability_param >= 1.0:
        print("WARNING: Stability condition violated! Reduce K or dt.")

    return sigmas, population_L, num_firms_N


def check_stability(K, population, sigmas, sensitivity_coefficients, signal):
    """
    Checks stability for the rate-based matching function m = K * u  (u = U/L).
    Condition: K * u_max * dt < 1 for non-oscillatory convergence.
    """
    G_min = signal.min()
    C = np.asarray(sensitivity_coefficients)

    total_target_employment_at_min = np.sum(sigmas * (1 + C * G_min))
    u_max_level = population - total_target_employment_at_min
    u_max_rate = u_max_level / population

    if u_max_rate <= 0:
        print("Warning: Maximum unemployment rate (u_max) is zero or negative.")
        return True

    stability_param = K * u_max_rate * DT

    if stability_param >= 1.0:
        print("\n--- SIMULATION STABILITY WARNING ---")
        print(f"K * u_max * dt = {K:.2f} * {u_max_rate:.4f} * {DT} = {stability_param:.4f} >= 1")
        print("This will cause oscillatory instability. Reduce K or dt.")
        return False

    return True
