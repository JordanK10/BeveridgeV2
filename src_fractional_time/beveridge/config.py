"""Simulation parameters and paths."""

import os

import numpy as np

# Package lives at src/beveridge/; project root is two levels up from this file.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

SENSITIVITY_COEFFICIENT = [0.1]  # C
BASE_SIGMA = [300]  # sigma
POPULATION = 1
PRODUCTIVITY_DENSITY = [1000]  # pi
MATCHING_RATE_CONSTANT = [5.4]  # K (month⁻¹)
INIT_VACANCIES = [0]  # V0
SEPARATION_RATE = 0.01  # s

BURN_IN = 2000

sigmas = np.linspace(250, 1900, 20)
c_values = [0.25, 0.5, 0.75, 1]
k_values_sweep = np.logspace(np.log10(5e-5), np.log10(5e-4), 4)

# Default GDP column for multi-firm and related runs
GDP_SIGNAL_NAME = "gdp_ar2"

# Set by run_single_timeseries when population is recomputed
INIT_UNEMPLOYMENT = None
