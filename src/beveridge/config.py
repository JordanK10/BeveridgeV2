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
# K scaled with DT so K*u_max*DT matches prior 5.4×0.1 (see time_grid.DT, economy stability check)
MATCHING_RATE_CONSTANT = [3.5]  # K (per unit time); was 0.54 before calibration sweep
INIT_VACANCIES = [0]  # V0
SEPARATION_RATE = 0.01  # s

# Calendar-equivalent to old 2000 steps × 0.1 = 200 time units
BURN_IN = 200

sigmas = np.linspace(250, 1900, 20)
c_values = [0.25, 0.5, 0.75, 1]
k_values_sweep = np.logspace(np.log10(5e-5), np.log10(5e-4), 4)

# Default GDP column for multi-firm and related runs
GDP_SIGNAL_NAME = "gdp_ar2"

# Multi-firm: each firm uses weight 1 on ``BASE_SIGMA``; ``initialize_economy`` sets
# labor force L = sum(sigmas)/(1 - target_u_rate), so L scales linearly with this count.
MULTI_FIRM_COUNT = 250

# Power-law C: at most this many firms at C=0 (minimum sensitivity); excess
# implied by ``zero_fraction`` becomes positive-C draws (same RNG usage order).
POWER_LAW_MAX_ZERO_FIRMS = 75

# Bins for firm-level C histograms (multi-firm + sweep distribution panels).
C_DISTRIBUTION_HIST_BINS = 15

# Set by run_single_timeseries when population is recomputed
INIT_UNEMPLOYMENT = None
