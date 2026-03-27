"""Aggregate vacancy / unemployment rates (no matplotlib at import time)."""

import numpy as np

from plotting.paths import OUTPUT_DIR


def vacancy_and_unemployment_rates(firms, population):
    """
    Time series of vacancy rate V/(V+E) and unemployment rate 1 - E/population
    from a list of Firm instances sharing the same time index.
    """
    time_indices = range(len(firms[0].employment))
    vacancies = [np.sum([firm.vacancies[i] for firm in firms]) for i in time_indices]
    employment = [np.sum([firm.employment[i] for firm in firms]) for i in time_indices]

    vacancy_rate = [
        vacancies[i] / (vacancies[i] + employment[i]) if (vacancies[i] + employment[i]) > 0 else 0
        for i in time_indices
    ]
    unemployment_rate = [1 - employment[i] / population for i in time_indices]

    return vacancy_rate, unemployment_rate


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
