"""Load pickle signals and build the in-memory GDP DataFrame aligned to TIME."""

import os
import pickle

import numpy as np
import pandas as pd

from time_grid import TIME, interpolate_series

from . import config

# Read in GDP data from the constant/sine signal in dummy_gdp.pkl
RAW_GDP = pd.read_pickle(os.path.join(config.DATA_DIR, "dummy_gdp.pkl"))

GDP = pd.DataFrame(index=range(len(TIME)))

if "gdp" in RAW_GDP.columns:
    gdp_interp = interpolate_series(RAW_GDP["gdp"], TIME)
    GDP["gdp_sine"] = gdp_interp
    GDP["gdp_custom"] = gdp_interp

try:
    with open(os.path.join(config.DATA_DIR, "ar2_signal.pkl"), "rb") as f:
        ar2_data = pickle.load(f)
    ar2_growth_signal = ar2_data["ar2_growth"]
    ar2_interp = interpolate_series(ar2_growth_signal, TIME)
    ar2_centered = ar2_interp - np.mean(ar2_interp)
    ar2_amplified = ar2_centered / np.std(ar2_centered) * 0.25
    GDP["gdp_ar2"] = ar2_amplified
    print("Successfully loaded and processed ar2_signal.pkl (using amplified growth rates)")
except (FileNotFoundError, KeyError):
    print("ar2_signal.pkl not found or invalid. Skipping AR2 experiment.")
    GDP["gdp_ar2"] = np.zeros(len(TIME))

GDP["gdp_constant"] = GDP["gdp_sine"][0] * 100
print("Added constant (flat) signal: gdp_constant")
