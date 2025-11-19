import numpy as np
import pickle
import matplotlib.pyplot as plt


gdp0 = 1000.0

def simulate_ar2_skewed(
    n_periods=400,
    avg_growth=0.02,          # mean growth rate
    volatility=0.02,          # std of innovations BEFORE skew transformation
    phi1=0.5,
    phi2=0.2,
    target_skewness=0.5,      # ~0.4–0.6 typical for GDP growth
    seed=42
):
    """
    Simulate an AR(2) process with skewed innovations.
    
    The innovations use the transformation:
        z = eps + alpha*(eps**2 - 1)
    where eps ~ N(0,1). For alpha > 0 this produces right-skewed noise.
    
    Empirically (checked via simulation):
        skew(z) ≈ 6 * alpha     for small alpha
    
    Parameters
    ----------
    target_skewness : float
        Desired skewness of the shock distribution (approximate).
    """
    
    rng = np.random.default_rng(seed)

    # --- approximate alpha from desired skewness ---
    alpha = target_skewness / 6.0
    alpha = float(np.clip(alpha, 0.0, 1.0))  # safety bounds

    # --- generate skewed shocks ---
    eps = rng.normal(size=n_periods)
    z = eps + alpha * (eps**2 - 1.0)

    # standardize to mean 0, variance 1
    z = (z - z.mean()) / z.std()

    # scale by desired volatility
    shocks = volatility * z

    # --- AR(2) recursion ---
    g = np.zeros(n_periods)
    g[0] = avg_growth
    g[1] = avg_growth

    for t in range(2, n_periods):
        g[t] = (
            avg_growth
            + phi1 * (g[t-1] - avg_growth)
            + phi2 * (g[t-2] - avg_growth)
            + shocks[t]
        )

    return g


if __name__ == "__main__":
    # User-facing parameters (change these)
    n_periods = 400
    avg_growth = 0.02        # 2% mean growth
    volatility = 0.02        # std dev of shocks
    target_skewness = 0.5    # GDP-like
    phi1 = 0.5
    phi2 = 0.2
    seed = 42

    # Generate skewed AR(2)
    ar2_series = simulate_ar2_skewed(
        n_periods=n_periods,
        avg_growth=avg_growth,
        volatility=volatility,
        phi1=phi1,
        phi2=phi2,
        target_skewness=target_skewness,
        seed=seed
    )

    gdp =gdp0*np.cumprod(1+ar2_series)

    plt.plot(np.log(gdp))
    plt.savefig("ar2_signal.png")

    # Save output
    result = {"ar2": gdp}

    with open("ar2_signal.pkl", "wb") as f:
        pickle.dump(result, f)

    print("Saved AR(2) series with skewed shocks → ar2_signal.pkl")
