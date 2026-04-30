"""
PHASE 2: Lightweight simulator for market dynamics.

Mirrors firm.py and economy.py logic but works with arbitrary time grids
and doesn't depend on global TIME.
"""

import numpy as np


class SimpleCalibFirm:
    """Minimal firm with demand-responsive employment."""

    def __init__(self, sigma, c, productivity=1.0, idio_std=0.01, firing_threshold=0.0, seed=None):
        self.sigma = sigma
        self.c = c
        self.productivity = productivity
        self.idio_std = idio_std
        self.firing_threshold = firing_threshold  # α: fire only when (e - ê)/ê > α
        self.rng = np.random.default_rng(seed)

        self.employment = None
        self.vacancies = None
        self.employment_demand = None
        self.matching_function = None

    def compute_demand(self, signal_array):
        """
        Compute target employment over time.

        ê(t) = σ * (1 + C * G(t) + η(t))

        where η is iid N(0, idio_std) idiosyncratic noise (no persistence).
        """
        n = len(signal_array)
        eta = self.rng.normal(0, self.idio_std, size=n)

        demand = self.sigma * (1 + self.c * np.asarray(signal_array) + eta)
        return np.maximum(demand, 0.01)  # Ensure positive

    def initialize_state(self, time_len, init_employment):
        self.employment = np.zeros(time_len)
        self.vacancies = np.zeros(time_len)
        self.employment_demand = np.zeros(time_len)
        self.matching_function = np.zeros(time_len)
        self.employment[0] = init_employment

    def update_vacancies(self, t, m, s, dt):
        """Post vacancies to fill demand gap + account for separations."""
        if m > 0:
            v_t = (
                self.employment_demand[t]
                - self.employment[t]
                + self.employment[t] * s / m
            )
        else:
            v_t = self.employment_demand[t] - self.employment[t]

        self.vacancies[t] = max(0, v_t)
        self.matching_function[t] = m

    def update_employment(self, t, dt, m, s, hiring_efficiency=1.0):
        """Hire, separate, fire to match demand.

        Firing only triggers when the fractional overshoot exceeds firing_threshold α:
            fire if (e - ê) / ê > α
        This implements the [·]_θ threshold from discrete.txt with θ = α·ê.
        α=0 recovers the original instantaneous firing behaviour.

        hiring_efficiency φ ∈ (0,1] scales the fill rate of posted vacancies:
            hires = φ * v * m * dt
        Lower φ slows hiring (longer u recovery) without affecting vacancy posting.
        NOTE: hiring_efficiency is currently kept but not actively calibrated —
        see fit_report_updated.txt for rationale.
        """
        # hires = hiring_efficiency * self.vacancies[t] * m * dt  # φ-scaled (disabled)
        hires = self.vacancies[t] * m * dt
        separations = self.employment[t] * s * dt

        demand = self.employment_demand[t]
        excess = self.employment[t] - demand
        # Firing threshold θ = α * (e / ẽ): endogenous e scales the deadband,
        # exogenous ẽ (target demand) normalizes it — firm fires when excess > θ
        theta = self.firing_threshold * self.employment[t] / demand if demand > 0 else 0.0
        if excess > theta:
            firing = excess - theta
        else:
            firing = 0.0

        change = hires - separations - firing
        self.employment[t + 1] = max(0, self.employment[t] + change)


def simulate_market(
    time_array,
    gdp_signal,
    n_firms=250,
    c_exponent=.8,
    c_max=.3,
    zero_fraction=0.0625,
    K=3.5,
    s=0.03 ,
    target_u=0.05,
    seed=42,
    dt=1.0,
    burn_in_steps=5,
    firing_threshold=0.025,
    hiring_efficiency=1.0,
    idio_std=0.05,
):
    """
    Simulate market dynamics with power-law C distribution.

    Parameters
    ----------
    time_array : np.ndarray
        Time points (not necessarily evenly spaced; used for length only)
    gdp_signal : np.ndarray or pd.Series
        G(t) values aligned to time_array
    n_firms : int
        Total number of firms
    c_exponent : float
        Power-law exponent (α) for C distribution
    c_max : float
        Maximum sensitivity coefficient
    zero_fraction : float
        Fraction of firms with C=0
    K : float
        Matching rate constant (m = K * u)
    s : float
        Exogenous separation rate
    target_u : float
        Target baseline unemployment rate
    seed : int
        Random seed
    dt : float
        Time step

    Returns
    -------
    unemployment : np.ndarray
        U(t) / L (unemployment rate)
    vacancy_rate : np.ndarray
        V(t) / (V + E) (vacancy rate)
    """
    rng = np.random.default_rng(seed)
    T = len(time_array)
    gdp_signal = np.asarray(gdp_signal).ravel()

    if len(gdp_signal) != T:
        raise ValueError(f"gdp_signal length {len(gdp_signal)} != time_array length {T}")

    # Initialize C distribution (power-law)
    n_zero = int(n_firms * zero_fraction)
    n_power = n_firms - n_zero

    c_values = np.concatenate([
        np.zeros(n_zero),
        c_max * rng.uniform(0, 1, n_power) ** (1.0 / c_exponent),
    ])
    rng.shuffle(c_values)

    # Initialize firms
    base_sigma = 300.0
    sigmas = np.full(n_firms, base_sigma)
    firms = [
        SimpleCalibFirm(sigmas[i], c_values[i], idio_std=idio_std,
                        firing_threshold=firing_threshold, seed=seed + i)
        for i in range(n_firms)
    ]

    # Initialize labor force
    baseline_employment = np.sum(sigmas * (1 + c_values * 0.0))
    L = baseline_employment / (1.0 - target_u)

    # Compute target employment for actual time series
    for firm in firms:
        firm.initialize_state(T, 0)
        firm.employment_demand = firm.compute_demand(gdp_signal)

    # Initialize each firm at the steady-state employment level consistent with
    # target_u: L * (1 - target_u) workers spread evenly across all firms.
    # Since all firms share the same sigma this is already the symmetric equilibrium.
    init_emp_per_firm = L * (1.0 - target_u) / n_firms
    first_signal = gdp_signal[0]
    burn_in_target = np.array([
        max(firm.sigma * (1.0 + firm.c * first_signal), 0.01)
        for firm in firms
    ])
    for firm in firms:
        firm.employment[0] = init_emp_per_firm

    if burn_in_steps > 0:
        burn_emp = np.array([firm.employment[0] for firm in firms], dtype=float)
        burn_in_unemployment = max(0.0, (L - burn_emp.sum()) / L)

        for _ in range(burn_in_steps):
            m_t = K * burn_in_unemployment
            next_emp = burn_emp.copy()

            for i, firm in enumerate(firms):
                demand_t = burn_in_target[i]

                if m_t > 0:
                    v_t = max(0.0, demand_t - burn_emp[i] + burn_emp[i] * s / m_t)
                else:
                    v_t = max(0.0, demand_t - burn_emp[i])

                # Match main simulation logic exactly
                hires = v_t * m_t * dt
                separations = burn_emp[i] * s * dt

                excess = burn_emp[i] - demand_t
                theta = firing_threshold * burn_emp[i] / demand_t if demand_t > 0 else 0.0
                if excess > theta:
                    firing = excess - theta
                else:
                    firing = 0.0

                change = hires - separations - firing
                next_emp[i] = max(0.0, burn_emp[i] + change)

            burn_emp = next_emp
            burn_in_unemployment = max(0.0, (L - burn_emp.sum()) / L)

        # Use burn-in terminal state as t=0 initial employment
        for i, firm in enumerate(firms):
            firm.employment[0] = burn_emp[i]

    # Simulate market on actual time series
    unemployment = np.zeros(T)
    unemployment[0] = (L - sum(f.employment[0] for f in firms)) / L

    for t in range(T - 1):
        m_t = K * unemployment[t]

        # Update all firms
        for firm in firms:
            firm.update_vacancies(t, m_t, s, dt)

        for firm in firms:
            firm.update_employment(t, dt, m_t, s, hiring_efficiency=hiring_efficiency)

        # Market clearing: unemployment is residual
        total_employment = sum(f.employment[t + 1] for f in firms)
        unemployment[t + 1] = max(0, (L - total_employment) / L)

    # Compute vacancy rate
    total_vacancies = np.sum([f.vacancies for f in firms], axis=0)
    total_employment = np.sum([f.employment for f in firms], axis=0)
    vacancy_rate = total_vacancies / (total_vacancies + total_employment + 1e-10)

    return unemployment, vacancy_rate


def simulate_market_with_firms(**kwargs):
    """
    Same as simulate_market but also returns the list of SimpleCalibFirm objects,
    each with .c, .sigma, and .employment[T] populated.

    Returns
    -------
    unemployment : np.ndarray
    vacancy_rate : np.ndarray
    firms : list of SimpleCalibFirm
    """
    # Re-run the simulation by calling the internal logic directly.
    # We duplicate the call rather than modifying simulate_market's return
    # signature to avoid breaking the 12+ existing callers.
    rng = np.random.default_rng(kwargs.get("seed", 42))
    time_array  = kwargs["time_array"]
    gdp_signal  = np.asarray(kwargs["gdp_signal"]).ravel()
    n_firms     = kwargs.get("n_firms", 250)
    c_exponent  = kwargs.get("c_exponent", 0.8)
    c_max       = kwargs.get("c_max", 1.0)
    zero_fraction    = kwargs.get("zero_fraction", 0.5)
    K                = kwargs.get("K", 3.5)
    s                = kwargs.get("s", 0.01)
    target_u         = kwargs.get("target_u", 0.05)
    dt               = kwargs.get("dt", 1.0)
    burn_in_steps    = kwargs.get("burn_in_steps", 1000)
    firing_threshold = kwargs.get("firing_threshold", 0.0)
    hiring_efficiency= kwargs.get("hiring_efficiency", 1.0)
    idio_std         = kwargs.get("idio_std", 0.05)
    seed             = kwargs.get("seed", 42)

    T = len(time_array)
    n_zero  = int(n_firms * zero_fraction)
    n_power = n_firms - n_zero
    c_values = np.concatenate([
        np.zeros(n_zero),
        c_max * rng.uniform(0, 1, n_power) ** (1.0 / c_exponent),
    ])
    rng.shuffle(c_values)

    base_sigma = 300.0
    sigmas = np.full(n_firms, base_sigma)
    firms = [
        SimpleCalibFirm(sigmas[i], c_values[i], idio_std=idio_std,
                        firing_threshold=firing_threshold, seed=seed + i)
        for i in range(n_firms)
    ]

    baseline_employment = np.sum(sigmas * (1 + c_values * 0.0))
    L = baseline_employment / (1.0 - target_u)

    for firm in firms:
        firm.initialize_state(T, 0)
        firm.employment_demand = firm.compute_demand(gdp_signal)

    init_emp_per_firm = L * (1.0 - target_u) / n_firms
    first_signal = gdp_signal[0]
    burn_in_target = np.array([
        max(firm.sigma * (1.0 + firm.c * first_signal), 0.01) for firm in firms
    ])
    for firm in firms:
        firm.employment[0] = init_emp_per_firm

    if burn_in_steps > 0:
        burn_emp = np.array([firm.employment[0] for firm in firms], dtype=float)
        burn_u = max(0.0, (L - burn_emp.sum()) / L)
        for _ in range(burn_in_steps):
            m_t = K * burn_u
            next_emp = burn_emp.copy()
            for i, firm in enumerate(firms):
                d = burn_in_target[i]
                v_t = max(0.0, d - burn_emp[i] + burn_emp[i] * s / m_t) if m_t > 0 else max(0.0, d - burn_emp[i])
                hires = v_t * m_t * dt
                seps  = burn_emp[i] * s * dt
                excess = burn_emp[i] - d
                theta = firing_threshold * burn_emp[i] / d if d > 0 else 0.0
                firing = max(0.0, excess - theta) if excess > theta else 0.0
                next_emp[i] = max(0.0, burn_emp[i] + hires - seps - firing)
            burn_emp = next_emp
            burn_u = max(0.0, (L - burn_emp.sum()) / L)
        for i, firm in enumerate(firms):
            firm.employment[0] = burn_emp[i]

    unemployment = np.zeros(T)
    unemployment[0] = (L - sum(f.employment[0] for f in firms)) / L
    for t in range(T - 1):
        m_t = K * unemployment[t]
        for firm in firms:
            firm.update_vacancies(t, m_t, s, dt)
        for firm in firms:
            firm.update_employment(t, dt, m_t, s, hiring_efficiency=hiring_efficiency)
        unemployment[t + 1] = max(0, (L - sum(f.employment[t + 1] for f in firms)) / L)

    total_vacancies  = np.sum([f.vacancies   for f in firms], axis=0)
    total_employment = np.sum([f.employment  for f in firms], axis=0)
    vacancy_rate = total_vacancies / (total_vacancies + total_employment + 1e-10)

    return unemployment, vacancy_rate, firms


def simulate_market_with_sectors(
    time_array,
    gdp_signal,
    sector_specs,
    K=3.5,
    seed=42,
    dt=1.0,
):
    """
    Simulate market with sector heterogeneity.

    Parameters
    ----------
    time_array : np.ndarray
    gdp_signal : np.ndarray
    sector_specs : list of dict
        Each: {name, n_firms, c_exponent, c_max, zero_fraction, s}
    K : float
    seed : int
    dt : float

    Returns
    -------
    results : dict
        {sector_name: (u, v)}
    """
    results = {}
    for i, spec in enumerate(sector_specs):
        u, v = simulate_market(
            time_array,
            gdp_signal,
            n_firms=spec["n_firms"],
            c_exponent=spec["c_exponent"],
            c_max=spec["c_max"],
            zero_fraction=spec["zero_fraction"],
            K=K,
            s=spec["s"],
            seed=seed + i,
            dt=dt,
        )
        results[spec["name"]] = (u, v)

    return results
