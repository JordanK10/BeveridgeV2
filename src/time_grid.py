"""Canonical simulation time grid and calendar-consistent signal interpolation.

Design (DT=1 migration):
- **T_TOTAL** = STEPS * DT is the simulated calendar span (unchanged from the old
  50000×0.1 = 5000 time units).
- **TIME** = ``np.arange(STEPS) * DT`` gives model times τ ∈ {0, …, T_TOTAL − DT}.
- Raw GDP pickles are **N** samples spanning **[0, T_TOTAL)** (left-inclusive grid);
  ``interpolate_series`` maps model **TIME** onto that calendar so changing **DT**
  only changes sampling density along the **same** underlying path (for fixed raw).

Validation: compare post–burn-in moments of unemployment/vacancy **rates** (mean, std,
correlation) to a baseline run with the previous (DT, STEPS, K); paths are not
required to match bit-for-bit.
"""

import numpy as np
from scipy.interpolate import interp1d

DT = 1.0
STEPS = 5000
T_TOTAL = float(STEPS * DT)
# Model clock: 0, DT, 2*DT, …, (STEPS-1)*DT  (same span as old 0 … 4999.9 by 0.1)
TIME = np.arange(STEPS, dtype=float) * DT


def mid_run_plot_slice(start_frac=0.22, end_frac=0.24):
    """
    Integer indices ``ll:ul`` for legacy figures that zoomed ~22–24% through the run.
    Works for any ``STEPS`` (e.g. 1100:1200 when ``STEPS=5000``).
    """
    ll = max(0, int(STEPS * start_frac))
    ul = min(STEPS, max(ll + 1, int(STEPS * end_frac)))
    return ll, ul


def interpolate_series(raw_series, time_array, t_total=None):
    """
    Linearly interpolate ``raw_series`` onto ``time_array`` using a **calendar** axis.

    Knot times are ``linspace(0, t_total, len(raw), endpoint=False)``, i.e. **N**
    evenly spaced samples over **[0, t_total)**. Defaults ``t_total`` to
    :data:`T_TOTAL` from this module so pickles of length **N** align with the
    simulated horizon regardless of **N** (e.g. 50k knots over 5000 calendar units).
    """
    raw = np.asarray(raw_series, dtype=float).ravel()
    n = len(raw)
    if n == 0:
        raise ValueError("raw_series is empty")
    if t_total is None:
        t_total = T_TOTAL
    t_total = float(t_total)
    if n == 1:
        t_raw = np.array([0.0])
    else:
        t_raw = np.linspace(0.0, t_total, n, endpoint=False)
    f = interp1d(t_raw, raw, kind="linear", fill_value="extrapolate")
    return f(np.asarray(time_array, dtype=float))
