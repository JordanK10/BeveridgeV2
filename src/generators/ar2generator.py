import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from scipy.stats import skew, skewnorm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')


gdp0 = 1000.0

# User-facing parameters (change these)
# Long AR sample; pickle is interpolated onto model TIME via time_grid.interpolate_series (calendar [0, T_TOTAL)).
n_periods = 10000
avg_growth = 0.02          # Mean growth rate per period (~2% quarterly)
volatility = 0.01          # Innovation std dev before AR filtering
target_skewness = -1.0     # Mild negative skew: sharp recessions, gradual recoveries (US data ~ -0.5 to -1.0)
phi1 = 0.8                 # AR(1) persistence (stationarity requires phi1 + phi2 < 1)
phi2 = 0.1                 # AR(2) coefficient (adds cycle-like memory)
seed = 35

if __name__ == "__main__":
    # --- 1. Generate skewed innovations (the random shocks) ---
    rng = np.random.default_rng(seed)
    alpha = target_skewness / 6.0
    eps = rng.normal(size=n_periods)
    z = eps + alpha * (eps**2 - 1.0)
    z = (z - z.mean()) / z.std() # Standardize
    shocks = volatility * z

    # --- 3. Run the AR(2) recursion using the modified shocks ---
    ar2_series = np.zeros(n_periods)
    ar2_series[0] = avg_growth
    ar2_series[1] = avg_growth

    for t in range(2, n_periods):
        ar2_series[t] = (
            avg_growth
            + phi1 * (ar2_series[t-1] - avg_growth)
            + phi2 * (ar2_series[t-2] - avg_growth)
            + shocks[t]
        )
    
    # Use the AR(2) growth rate series directly (no detrending)
    gdp = gdp0 * np.cumprod(1 + ar2_series)
    
    # --- Check skewness ---
    shock_skew = skew(shocks)
    output_skew = skew(ar2_series)
    print(f"Target Skewness (parameter): {target_skewness}")
    print(f"Empirical Skewness of Shocks (Input): {shock_skew:.4f}")
    print(f"Empirical Skewness of AR(2) Series (Output): {output_skew:.4f}")

    # --- Create an informative plot with subplots ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
    
    # Plot 1: Log GDP Level (to handle large magnitudes)
    ax1.plot(np.log10(gdp), label="Log10 GDP Level")
    ax1.set_title("Engineered AR(2) Demand Signal")
    ax1.set_ylabel("Log10 Demand Level")
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Growth Rate
    ax2.plot(ar2_series, label="GDP Growth Rate", color='orange')
    ax2.set_ylabel("Growth Rate")
    ax2.set_xlabel("Time")
    ax2.grid(True)
    ax2.axhline(avg_growth, color='blue', linestyle='--', alpha=0.7, label=f'Average Growth: {avg_growth:.4f}')
    ax2.legend()

    # Plot 3: Histogram of Growth Rate Fluctuations
    fluctuations = ar2_series - avg_growth
    ax3.hist(fluctuations, bins=50, density=True, color='green', alpha=0.7, label='Fluctuations')
    
    # Fit and plot Skew-Normal distribution
    a, loc, scale = skewnorm.fit(fluctuations)
    x = np.linspace(fluctuations.min(), fluctuations.max(), 100)
    p = skewnorm.pdf(x, a, loc, scale)
    ax3.plot(x, p, 'k', linewidth=2, label=f'Skew-Normal Fit (α={a:.2f})')

    ax3.set_title(f"Growth Rate Fluctuations (Empirical Skewness: {output_skew:.4f})")
    ax3.set_xlabel("Deviation from Mean Growth")
    ax3.set_ylabel("Density")
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ar2_signal_detailed.png"))
    print("Saved detailed AR(2) signal plot to ar2_signal_detailed.png")

    # Save output
    result = {"ar2_level": gdp, "ar2_growth": ar2_series}

    with open(os.path.join(DATA_DIR, "ar2_signal.pkl"), "wb") as f:
        pickle.dump(result, f)

    print("Saved AR(2) series with skewed shocks → ar2_signal.pkl")
