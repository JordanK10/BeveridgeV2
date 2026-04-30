"""Aggregate vacancy / unemployment rates (no matplotlib at import time)."""

import numpy as np

from plotting.paths import OUTPUT_DIR


def aggregate_vacancy_rate(unemployment_levels, vacancy_levels, population):
    """
    Aggregate vacancy rate as **v / (v + e)** with:

    - ``v``: aggregate vacancy count (sum of firm vacancies),
    - ``e``: aggregate employment = ``L - U`` (``L`` = labor force, ``U`` = unemployment).

    Same as **V / (V + E)** when ``V, E`` denote aggregate vacancies and employment.
    """
    L = float(population)
    u = np.asarray(unemployment_levels, dtype=float)
    v = np.asarray(vacancy_levels, dtype=float)
    e = L - u
    denom = v + e
    return np.divide(v, denom, out=np.zeros_like(v, dtype=float), where=denom > 0)


def vacancy_and_unemployment_rates(firms, population):
    """
    Time series of vacancy rate v/(v+e) and unemployment rate U/L
    from a list of Firm instances sharing the same time index.
    """
    time_indices = range(len(firms[0].employment))
    vacancies = np.array([np.sum([firm.vacancies[i] for firm in firms]) for i in time_indices])
    employment = np.array([np.sum([firm.employment[i] for firm in firms]) for i in time_indices])
    unemployment = float(population) - employment

    vacancy_rate = aggregate_vacancy_rate(unemployment, vacancies, population)
    unemployment_rate = unemployment / float(population)

    return vacancy_rate.tolist(), unemployment_rate.tolist()


def compute_rates(firms, economy_name, population, plot=True, output_dir=None, burn_in=0):
    """
    Computes vacancy and unemployment rates and optionally plots the Beveridge curve.

    Args:
        burn_in (int): Leading steps discarded only for the Beveridge trajectory figure.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    vacancy_rate, unemployment_rate = vacancy_and_unemployment_rates(firms, population)

    if plot:
        from plotting.beveridge_figs import plot_beveridge_trajectory

        plot_beveridge_trajectory(
            economy_name, unemployment_rate, vacancy_rate, output_dir, burn_in=burn_in
        )

    return vacancy_rate, unemployment_rate
