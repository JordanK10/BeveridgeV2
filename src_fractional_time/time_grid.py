"""Canonical simulation time grid and signal interpolation onto that grid."""

import numpy as np
from scipy.interpolate import interp1d

DT = 0.1
STEPS = 50000
# Time grid: 0, 0.1, ..., (STEPS - 1) * DT  (matches typical 50k-length pickles)
TIME = np.arange(0, STEPS * DT, DT)


def interpolate_series(raw_series, time_array):
    """Linearly interpolate a series indexed at t = 0, 1, 2, ... onto ``time_array``."""
    t_raw = np.arange(len(raw_series))
    f = interp1d(t_raw, raw_series, kind="linear", fill_value="extrapolate")
    return f(time_array)
