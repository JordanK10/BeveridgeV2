#!/usr/bin/env python3
"""
Simulate a stochastic GDP time series (GBM) with a recession window:
- Positive drift from t=0 to t1 = 1/3 T
- Negative drift from t1 to t2 = 2/5 T
- Positive drift from t2 to T
Saves a pandas DataFrame with columns ["time", "gdp"] to dummy_demand.pkl.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')


def simulate_gdp(
    T=20.0,              # total (normalized) time
    N=500,              # number of steps
    S0=500.0,           # initial GDP level (index)
    mu_pos=0.03,        # positive drift (before t1 and after t2)
    mu_neg=-0.12,       # negative drift (recession window)
    sigma=0.02,         # volatility
    seed=2,            # RNG seed for reproducibility
):
    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0.0, T, N + 1)

    # Interfaces
    t1 = 1.0 / 3.0 * T   # drop onset
    t2 = 2.0 / 5.0 * T   # recovery onset

    # Piecewise drift over time grid (use left-point value per interval)
    mu = np.where(t < t1, mu_pos, np.where(t < t2, mu_neg, mu_pos))

    # Simulate GBM with Euler–Maruyama on levels
    S = np.empty_like(t)
    S[0] = S0
    dW = rng.normal(loc=0.0, scale=np.sqrt(dt), size=N)

    for i in range(N):
        S[i + 1] = S[i] + mu[i] * S[i] * dt + sigma * S[i] * dW[i]
        if S[i + 1] <= 0:  # numerical guard
            S[i + 1] = np.finfo(float).eps

    return t, S, t1, t2

def main():
    # --- parameters for sine wave signal ---
    signal_steps = 50000 # Match STEPS in beveridge.py or n_periods in ar2generator.py
    amplitude = .25
    frequency = 0.01 # Controls how many cycles over the signal_steps
    offset = 0.0   # Baseline for the sine wave

    # --- generate data ---
    t = np.arange(signal_steps)
    S = offset + amplitude * np.cos(2 * np.pi * frequency * t)

    # --- save ---
    df = pd.DataFrame({"time": t, "gdp": S})
    df.to_pickle(os.path.join(DATA_DIR, "dummy_gdp.pkl"))
    print("Saved sine wave GDP series to dummy_gdp.pkl")
    print(df)

    # --- quick plot ---
    plt.figure()
    plt.plot(t[:100], S[:100], 'o-', label=f"Sine Wave GDP (Amp={amplitude}, Freq={frequency})")
    plt.title("Sine Wave GDP Signal")
    plt.xlabel("Time (steps)")
    plt.ylabel("GDP (index)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dummy_gdp.png'))

    # --- Growth Rate Fluctuations Plot ---
    # Calculate log returns: ln(S_{t+1} / S_t)
    # Using np.diff(np.log(S)) gives us r_t for t=0 to T-2
    log_returns = np.diff(np.log(S))
    
    # Calculate fluctuations around the mean growth rate
    mean_growth = np.mean(log_returns)
    growth_fluctuations = log_returns - mean_growth
    
    # Time array for growth rates (length T-1)
    t_growth = t[:-1]

    plt.figure()
    plt.plot(t_growth[:100], growth_fluctuations[:100], 'r-', label="Growth Rate Fluctuations")
    plt.title("GDP Growth Rate Fluctuations (Log Returns - Mean)")
    plt.xlabel("Time (steps)")
    plt.ylabel("Fluctuation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dummy_gdp_growth_fluctuations.png'))
    print(f"Saved growth fluctuations plot to {os.path.join(OUTPUT_DIR, 'dummy_gdp_growth_fluctuations.png')}")


if __name__ == "__main__":
    main()