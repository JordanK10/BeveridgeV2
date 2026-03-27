"""
Post-burn-in moments for DT migration checks.

Run from project: ``PYTHONPATH=src python -m beveridge.validate_moments``

Compare printed statistics to a baseline run (e.g. git checkout prior time_grid/config)
to verify similar u/v rate distributions after changing DT/K.
"""

import numpy as np

from plotting.uv_crosscorr import _rates_from_levels
from time_grid import DT, STEPS, T_TOTAL, TIME

from . import config
from .economy import check_stability, initialize_economy, run_market
from .firm import Firm
from .signals import GDP


def run_and_print_moments():
    if "gdp_ar2" in GDP.columns and not np.allclose(GDP["gdp_ar2"], 0):
        signal_name = "gdp_ar2"
    else:
        signal_name = "gdp_constant"

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

    if not check_stability(
        config.MATCHING_RATE_CONSTANT[0],
        population,
        sigmas,
        config.SENSITIVITY_COEFFICIENT[0],
        GDP[signal_name],
    ):
        print("Stability check failed.")

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
    aggregate_vacancies = [sum(f.vacancies[i] for f in firms) for i in range(len(TIME))]

    ur, vr = _rates_from_levels(unemployment, aggregate_vacancies, population, config.BURN_IN)

    print("--- validate_moments (post burn-in) ---")
    print(f"signal={signal_name}  STEPS={STEPS}  DT={DT}  T_TOTAL={T_TOTAL}")
    print(f"K={config.MATCHING_RATE_CONSTANT[0]}  BURN_IN={config.BURN_IN}")
    print(f"mean u_rate={np.mean(ur):.6f}  std={np.std(ur):.6f}")
    print(f"mean v_rate={np.mean(vr):.6f}  std={np.std(vr):.6f}")
    print(f"corr(u,v)={np.corrcoef(ur, vr)[0, 1]:.6f}  n={len(ur)}")


if __name__ == "__main__":
    run_and_print_moments()
