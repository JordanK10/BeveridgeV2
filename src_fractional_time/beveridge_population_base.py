"""
Backward-compatible entry point for the Beveridge simulation.

Implementation lives in the ``beveridge`` package (``src/beveridge/``).
"""

import os
import sys

# Ensure imports like ``beveridge`` and ``plotting`` resolve (package root is this directory).
_src = os.path.dirname(os.path.abspath(__file__))
if _src not in sys.path:
    sys.path.insert(0, _src)

from beveridge.config import (
    BASE_SIGMA,
    BURN_IN,
    DATA_DIR,
    GDP_SIGNAL_NAME,
    INIT_UNEMPLOYMENT,
    INIT_VACANCIES,
    MATCHING_RATE_CONSTANT,
    OUTPUT_DIR,
    POPULATION,
    PRODUCTIVITY_DENSITY,
    PROJECT_ROOT,
    SEPARATION_RATE,
    SENSITIVITY_COEFFICIENT,
    c_values,
    k_values_sweep,
    sigmas,
)
from beveridge.economy import check_stability, compute_loop_area, initialize_economy, run_market
from beveridge.experiments import (
    compare_sine_special,
    run_ar2_experiment,
    run_matching_rate_sweep,
    run_multi_firm_simulation,
    run_sensitivity_sweep,
    run_simulation_for_signal,
    run_single_timeseries,
)
from beveridge.firm import Firm
from beveridge.main import main
from beveridge.signals import GDP, RAW_GDP

__all__ = [
    "BASE_SIGMA",
    "BURN_IN",
    "DATA_DIR",
    "GDP",
    "GDP_SIGNAL_NAME",
    "INIT_UNEMPLOYMENT",
    "INIT_VACANCIES",
    "MATCHING_RATE_CONSTANT",
    "OUTPUT_DIR",
    "POPULATION",
    "PRODUCTIVITY_DENSITY",
    "PROJECT_ROOT",
    "RAW_GDP",
    "SEPARATION_RATE",
    "SENSITIVITY_COEFFICIENT",
    "Firm",
    "c_values",
    "check_stability",
    "compare_sine_special",
    "compute_loop_area",
    "initialize_economy",
    "k_values_sweep",
    "main",
    "run_ar2_experiment",
    "run_market",
    "run_matching_rate_sweep",
    "run_multi_firm_simulation",
    "run_sensitivity_sweep",
    "run_simulation_for_signal",
    "run_single_timeseries",
    "sigmas",
]

if __name__ == "__main__":
    main()
