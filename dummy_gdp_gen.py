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


def simulate_gdp(
    T=20.0,              # total (normalized) time
    N=500,              # number of steps
    S0=100.0,           # initial GDP level (index)
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
    # --- simulate ---
    t, S, t1, t2 = simulate_gdp()

    # --- save ---
    df = pd.DataFrame({"time": t, "gdp": S})
    df.to_pickle("dummy_gdp.pkl")
    print("Saved GDP series to dummy_demand.pkl")
    print(df.head())

    # --- quick plot ---
    plt.figure()
    plt.plot(t, S, label="Simulated GDP (GBM, piecewise drift)")
    plt.axvline(t1, linestyle="--", label="drop start (t=1/3)")
    plt.axvline(t2, linestyle="--", label="recovery start (t=2/5)")
    plt.title("Stochastic GDP with Recession Window")
    plt.xlabel("Time (normalized)")
    plt.ylabel("GDP (index)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('dummy_gdp.png')


if __name__ == "__main__":
    main()