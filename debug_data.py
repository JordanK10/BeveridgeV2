import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from time_grid import DT, TIME, interpolate_series

PROJECT_ROOT = _ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Load dummy_gdp
try:
    dummy_gdp = pd.read_pickle(os.path.join(DATA_DIR, 'dummy_gdp.pkl'))
    print("Dummy GDP loaded.")
    print(dummy_gdp.head())
    print("Min:", dummy_gdp['gdp'].min(), "Max:", dummy_gdp['gdp'].max())
except Exception as e:
    print(f"Error loading dummy_gdp: {e}")

if 'dummy_gdp' in locals() and 'gdp' in dummy_gdp.columns:
    interpolated = interpolate_series(dummy_gdp['gdp'], TIME)
    print("\nInterpolated Sine Signal:")
    print(interpolated[:10])
    print("Min:", interpolated.min(), "Max:", interpolated.max())

    # Calculate growth rates
    signal_growth = (np.log(interpolated[1:]) - np.log(interpolated[:-1])) / DT
    print("\nSignal Growth Rate Stats:")
    print("Min:", signal_growth.min(), "Max:", signal_growth.max(), "Mean:", signal_growth.mean())
    print("Sample:", signal_growth[:10])

    # Plot first 100 points of signal and growth
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(TIME[:100], interpolated[:100])
    plt.title("Interpolated Signal")
    plt.subplot(2, 1, 2)
    plt.plot(TIME[:99], signal_growth[:99])
    plt.title("Signal Growth")
    plt.savefig("debug_signal.png")
    print("Saved debug_signal.png")
