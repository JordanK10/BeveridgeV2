import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')


gdp0 = 1000.0


if __name__ == "__main__":
    # User-facing parameters (change these)
    n_periods = 80          # 20 years * 4 quarters/year
    avg_growth = 0.02
    volatility = 0.02
    target_skewness = -0.5   # Negative skew for sharp recessions
    phi1 = 0.8
    phi2 = 0.1
    seed = 42

    # --- 1. Generate skewed innovations (the random shocks) ---
    rng = np.random.default_rng(seed)
    alpha = target_skewness / 6.0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    eps = rng.normal(size=n_periods)
    z = eps + alpha * (eps**2 - 1.0)
    z = (z - z.mean()) / z.std() # Standardize
    shocks = volatility * z

    # --- 2. Engineer the large, negative shock directly into the innovations ---
    shock_start = 40  # Year 11 * 4 quarters/year
    shock_end = 44    # Year 12 * 4 quarters/year
    # A shock to the growth rate is an external shock to the innovations.
    # A -5% growth rate is a shock of -0.07 relative to the mean growth of 0.02.
    shock_value = -0.07
    shocks[shock_start:shock_end] = shock_value

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
    
    # --- Detrend the final signal to ensure it's centered on zero ---
    # This makes it a pure fluctuation signal for the firm model.
    ar2_series_detrended = ar2_series - np.mean(ar2_series)
    
    gdp = gdp0 * np.cumprod(1 + ar2_series_detrended)
    
    # --- Create an informative plot with subplots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: GDP Level
    ax1.plot(gdp, label="GDP Level (AR2 process)")
    ax1.set_title("Engineered AR(2) Demand Signal")
    ax1.set_ylabel("Demand Level")
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Growth Rate
    ax2.plot(ar2_series_detrended, label="GDP Growth Rate (Detrended)", color='orange')
    ax2.axvspan(shock_start, shock_end, color='red', alpha=0.2, label='Engineered Shock')
    ax2.set_ylabel("Growth Rate")
    ax2.set_xlabel("Time")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ar2_signal_detailed.png"))
    print("Saved detailed AR(2) signal plot to ar2_signal_detailed.png")

    # Save output
    result = {"ar2_level": gdp, "ar2_growth": ar2_series_detrended}

    with open(os.path.join(DATA_DIR, "ar2_signal.pkl"), "wb") as f:
        pickle.dump(result, f)

    print("Saved AR(2) series with skewed shocks → ar2_signal.pkl")
