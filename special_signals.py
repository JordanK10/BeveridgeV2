#!/usr/bin/env python3
"""
Auto-generate signals including a calculated shock signal.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===========================
# ======= PARAMETERS ========
# ===========================

# Global time grid
T = 1.0               # total normalized time
N = 1999              # number of intervals; series has N+1 points
SEED = 123            # RNG seed for reproducibility

# Noise (applied additively, same schedule for all three signals)
NOISE_SCHEDULE = "constant"   # {"constant", "sin", "piecewise"}
SIGMA0 = 0.0000000000000001                 # baseline noise scale
SIGMA1 = 0.0000000000000001                  # amplitude (sin) or bump (piecewise)
RHO = 0.35                     # AR(1) persistence in noise ([-0.95, 0.95] recommended)

# Shock Signal Parameters
# Calculated for s=0.005, C=0.25, X=0.10
SHOCK_G_PEAK = 100.0
SHOCK_G_DROP = 90.12
T_SHOCK = 0.5

# Positive step parameters: low -> high at t_step_pos
POS_G_LOW = 75.0
POS_G_HIGH = 120.0
T_STEP_POS = 1.0 / 3.0

# Negative step parameters: high -> low at t_step_neg
NEG_G_HIGH = 120.0
NEG_G_LOW = 75.0
T_STEP_NEG = 1.0 / 3.0

# Sine wave parameters: baseline + amp * sin(2π f t + phase)
SINE_BASELINE = 1000.0
SINE_AMP = 100.0          # keep baseline > amp to ensure positivity in deterministic part
SINE_FREQ = 2.0         # cycles over [0, T]
SINE_PHASE = 0.0        # radians

# Custom periodic signal parameters
CUSTOM_BASELINE = 1000.0
CUSTOM_AMP = 100.0
CUSTOM_FREQ = 2.0      # cycles over [0, T]

# Output
OUTFILE = "dummy_demand.pkl"
SHOW_PLOT = True


# ===========================
# ===== IMPLEMENTATION ======
# ===========================

def sigma_t(t, schedule="constant", sigma0=0.2, sigma1=0.0, t1=0.33, t2=0.40):
    """
    Time-varying noise scale σ(t).
    schedule ∈ {"constant", "sin", "piecewise"}.
    """
    if schedule == "constant":
        return np.full_like(t, sigma0, dtype=float)
    if schedule == "sin":
        # smooth oscillation around sigma0
        return np.maximum(0.0, sigma0 + sigma1 * np.sin(2 * np.pi * t / (t[-1] if t[-1] > 0 else 1.0)))
    if schedule == "piecewise":
        out = np.full_like(t, sigma0, dtype=float)
        out[(t >= t1) & (t < t2)] = sigma0 + abs(sigma1)
        return out
    raise ValueError(f"Unknown noise schedule: {schedule}")

def ar1_noise(t, rng, schedule, sigma0, sigma1, rho, t1=0.33, t2=0.40):
    """
    AR(1) noise with time-varying innovation scale σ(t):
        eps_i = rho * eps_{i-1} + σ(t_i) * η_i,  η_i ~ N(0,1)
    eps[0] = 0 by convention.
    """
    sig = sigma_t(t, schedule=schedule, sigma0=sigma0, sigma1=sigma1, t1=t1, t2=t2)
    eps = np.zeros_like(t, dtype=float)
    for i in range(1, len(t)):
        eps[i] = rho * eps[i - 1] + sig[i] * rng.normal()
    return eps

def gdp_positive_step(t, rng):
    H = (t >= T_STEP_POS).astype(float)  # Heaviside step
    deterministic = POS_G_LOW + (POS_G_HIGH - POS_G_LOW) * H
    eps = ar1_noise(t, rng, NOISE_SCHEDULE, SIGMA0, SIGMA1, RHO, t1=T_STEP_POS, t2=T_STEP_POS + 1e-9)
    G = deterministic + eps
    return np.maximum(np.finfo(float).eps, G)

def gdp_negative_step(t, rng):
    H = (t >= T_STEP_NEG).astype(float)
    deterministic = NEG_G_HIGH - (NEG_G_HIGH - NEG_G_LOW) * H
    eps = ar1_noise(t, rng, NOISE_SCHEDULE, SIGMA0, SIGMA1, RHO, t1=T_STEP_NEG, t2=T_STEP_NEG + 1e-9)
    G = deterministic + eps
    return np.maximum(np.finfo(float).eps, G)

def gdp_sine(t, rng):
    if SINE_BASELINE <= abs(SINE_AMP):
        raise ValueError("Require SINE_BASELINE > |SINE_AMP| to keep GDP positive in deterministic part.")
    deterministic = SINE_BASELINE + SINE_AMP * np.sin(2 * np.pi * SINE_FREQ * (t / T) + SINE_PHASE)
    eps = ar1_noise(t, rng, NOISE_SCHEDULE, SIGMA0, SIGMA1, RHO)
    G = deterministic + eps
    return np.maximum(np.finfo(float).eps, G)

def gdp_custom(t, rng):
    """
    Generates a custom periodic signal with two parts per cycle:
    1. Sine wave rise (7/8 cycle)
    2. Exponential decay (1/8 cycle)
    """
    if CUSTOM_BASELINE <= abs(CUSTOM_AMP):
        raise ValueError("Require CUSTOM_BASELINE > |CUSTOM_AMP| to keep GDP positive in deterministic part.")

    T_cycle = T / CUSTOM_FREQ
    t_in_cycle = t % T_cycle
    
    deterministic = np.zeros_like(t, dtype=float)

    # Define phase transition point
    p1_end = 7 * T_cycle / 8

    # Part 1: Sine wave rise (0 to 7/8 of cycle)
    # This is a cosine that goes from trough to peak over the first 7/8 of the cycle.
    # The full period of this cosine would be 2 * p1_end = 1.75 * T_cycle.
    mask1 = t_in_cycle < p1_end
    t1 = t_in_cycle[mask1]
    T_rise = 1.75 * T_cycle 
    deterministic[mask1] = CUSTOM_BASELINE - CUSTOM_AMP * np.cos(2 * np.pi * t1 / T_rise)

    # Part 2: Exponential decay (7/8 to end of cycle)
    mask2 = t_in_cycle >= p1_end
    t2 = t_in_cycle[mask2] - p1_end # time relative to start of decay
    
    peak = CUSTOM_BASELINE + CUSTOM_AMP
    trough = CUSTOM_BASELINE - CUSTOM_AMP
    decay_duration = T_cycle / 8

    # We model the decay as y(t) = C1*exp(-k*t) + C2, solving for C1 and C2
    # such that y(0) = peak and y(decay_duration) = trough.
    # The decay constant 'k' is chosen so the curve has a sharp, natural shape.
    # Let's set k so the exponential term decays by 95% over the duration.
    decay_constant = 3.0 / decay_duration # k*D = 3 -> exp(-3) is ~5%
    
    C1 = (peak - trough) / (1 - np.exp(-3.0))
    C2 = peak - C1
    
    deterministic[mask2] = C1 * np.exp(-decay_constant * t2) + C2

    eps = ar1_noise(t, rng, NOISE_SCHEDULE, SIGMA0, SIGMA1, RHO)
    G = deterministic + eps
    return np.maximum(np.finfo(float).eps, G)

def gdp_shock_experiment(t, rng):
    """
    A constant high signal that drops discontinuously at t=0.5
    to test a specific employment shock size.
    """
    deterministic = np.where(t < T_SHOCK, SHOCK_G_PEAK, SHOCK_G_DROP)
    eps = ar1_noise(t, rng, NOISE_SCHEDULE, SIGMA0, SIGMA1, RHO)
    G = deterministic + eps
    return np.maximum(np.finfo(float).eps, G)

def main():
    # Time grid
    t = np.linspace(0.0, T, N + 1)
    rng = np.random.default_rng(SEED)

    # Generate three signals
    g_pos = gdp_positive_step(t, rng)
    g_neg = gdp_negative_step(t, rng)
    g_sin = gdp_sine(t, rng)
    g_custom = gdp_custom(t, rng)
    g_shock = gdp_shock_experiment(t, rng)

    # Save
    df = pd.DataFrame({
        "time": t,
        "gdp_pos_step": g_pos,
        "gdp_neg_step": g_neg,
        "gdp_sine": g_sin,
        "gdp_custom": g_custom,
        "gdp_shock": g_shock,
    })
    df.to_pickle(OUTFILE)
    print(f"Saved signals to {OUTFILE}")
    print(df.head())

    # Plot
    if SHOW_PLOT:
        plt.figure()
        plt.plot(t, g_pos, label="GDP – positive step")
        # plt.axvline(T_STEP_POS, linestyle="--", linewidth=1)
        # plt.plot(t, g_neg, label="GDP – negative step")
        # plt.axvline(T_STEP_NEG, linestyle="--", linewidth=1)
        # plt.plot(t, g_sin, label="GDP – sine")
        plt.plot(t, g_custom, label="GDP - custom")
        plt.plot(t, g_shock, label="GDP - shock", linewidth=2, color='red')
        plt.xlabel("time (normalized)")
        plt.ylabel("GDP (level)")
        plt.title("Synthetic GDP (linear scale) with additive AR(1) noise")
        plt.legend()
        plt.tight_layout()
        plt.savefig("special_signals.png")

if __name__ == "__main__":
    main()
