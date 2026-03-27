"""Single-firm state and dynamics."""

import os

import matplotlib.pyplot as plt
import numpy as np

from time_grid import TIME

from . import config


class Firm:
    def plot_demand(self):
        plt.plot(self.signal, label="signal")
        plt.plot(self.employment_demand, label="target employment")
        plt.legend()
        plt.savefig(os.path.join(config.OUTPUT_DIR, "demand.pdf"))
        plt.close()

    def update_vacancies(self, t, aggregate_unemployment_level, total_population):
        self.matching_function[t] = self.matching_rate_constant * (aggregate_unemployment_level / total_population)

        if self.matching_function[t] > 0:
            v_t = (
                self.employment_demand[t]
                - self.employment[t]
                + self.employment[t] * self.separation_rate[t] / self.matching_function[t]
            )
        else:
            v_t = self.employment_demand[t] - self.employment[t]

        if v_t < 0:
            v_t = 0

        floor_v = int(v_t)
        frac = v_t - floor_v
        self.vacancies[t] = floor_v + (1 if self._vacancy_rng.random() < frac else 0)

    def update_employment(self, t, dt):
        hires = self.vacancies[t] * self.matching_function[t] * dt
        separations = self.employment[t] * self.separation_rate[t] * dt

        excess_labor = self.employment[t] - self.employment_demand[t]
        if excess_labor > 0:
            firing = excess_labor
        else:
            firing = 0

        change = hires - separations - firing

        self.employment[t + 1] = self.employment[t] + change

        target = self.employment_demand[t]
        actual = self.employment[t + 1]

        if target <= 0:
            self.efficiency[t + 1] = 0.0
        else:
            self.efficiency[t + 1] = (actual - target) / target

    def computeDemand(self, signal, idio_persistence=0.9, idio_std=0.01, seed=None):
        """
        Compute target employment demand over the full time horizon.

        ê_i(t) = σ_i * (1 + C_i * G(t) + η_i(t))

        η_i is a zero-mean AR(1) idiosyncratic demand shock, uncorrelated across firms.
        """
        self.signal = signal
        aggregate_demand = self.firm_size * (1 + self.sensitivity_coefficient * np.asarray(signal))

        rng = np.random.default_rng(seed)
        n = len(aggregate_demand)
        eta = np.zeros(n)
        sigma_eps = idio_std
        for t in range(1, n):
            eta[t] = idio_persistence * eta[t - 1] + rng.normal(0, sigma_eps)

        return aggregate_demand + self.firm_size * eta

    def set_target(self):
        return self.employment_demand

    def __init__(
        self,
        signal,
        init_size,
        init_productivity_density,
        init_employment,
        init_vacancies,
        matching_rate_constant,
        sensitivity_coefficient,
        idio_persistence=0.9,
        idio_std=0.01,
        seed=None,
    ):
        self.sensitivity_coefficient = sensitivity_coefficient
        self.firm_size = init_size
        self.employment = [init_employment for _ in TIME]
        self.vacancies = [init_vacancies for _ in TIME]
        self.matching_function = [0 for _ in TIME]
        self.separation_rate = [config.SEPARATION_RATE for _ in TIME]
        self.efficiency = [0 for _ in TIME]
        vacancy_seed = (seed + 500) if seed is not None else None
        self._vacancy_rng = np.random.default_rng(vacancy_seed)
        self.time = None

        self.productivity_density = init_productivity_density
        self.employment_demand = self.computeDemand(signal, idio_persistence=idio_persistence, idio_std=idio_std, seed=seed)
        self.matching_rate_constant = matching_rate_constant
        self.target = self.set_target()

    def set_time(self, time):
        self.time = time
