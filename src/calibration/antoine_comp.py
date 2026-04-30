"""
Compare empirical tail exponents and skewness with Antoine's reference period (2000-2018).

Computes upper and lower tail exponents (via Hill estimator) and skewness for
u, v, and GDP growth over the reference period used in Antoine (2021).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew


def _hill_tail_exponent(data, tail_fraction=0.1):
    """
    Compute tail exponent using Hill's estimator.

    Parameters
    ----------
    data : array-like
        1D array of observations
    tail_fraction : float
        Fraction of largest/smallest values to use (default: 0.1 = 10%)

    Returns
    -------
    tuple
        (upper_alpha, lower_alpha) - estimated tail exponents
    """
    data = np.asarray(data)
    n_tail = max(1, int(len(data) * tail_fraction))

    # Sort data
    sorted_data = np.sort(data)

    # Upper tail: largest values (all positive)
    upper_tail = sorted_data[-n_tail:]
    if len(upper_tail) > 1 and np.all(upper_tail > 0):
        upper_log_ratio = np.log(upper_tail[-1] / upper_tail[:-1]).sum()
        upper_alpha = (len(upper_tail) - 1) / upper_log_ratio if upper_log_ratio > 0 else np.nan
    else:
        upper_alpha = np.nan

    # Lower tail: smallest values (all negative for meaningful tail analysis)
    # Use absolute values of the smallest (most negative) elements
    lower_tail = sorted_data[:n_tail]
    if len(lower_tail) > 1 and np.all(lower_tail < 0):
        lower_tail_abs = np.abs(lower_tail[::-1])  # Reverse and take absolute values
        lower_log_ratio = np.log(lower_tail_abs[-1] / lower_tail_abs[:-1]).sum()
        lower_alpha = (len(lower_tail) - 1) / lower_log_ratio if lower_log_ratio > 0 else np.nan
    else:
        lower_alpha = np.nan

    return upper_alpha, lower_alpha


def compute_antoine_comparison(
    start_date="2000-01-01",
    end_date="2018-12-31",
    output_dir="output/antoine_comparison",
    exclude_ranges=None,
):
    """
    Compute tail exponents and skewness for a date range, with optional exclusions.

    Parameters
    ----------
    start_date : str
        Start of reference window (ISO format). Default: 2000-01-01
    end_date : str
        End of reference window (ISO format). Default: 2018-12-31
    output_dir : str
        Where to save comparison results
    exclude_ranges : list of tuple, optional
        List of (start_date, end_date) tuples to exclude from analysis.
        Example: [("2020-03-01", "2020-06-01"), ("2020-03-01", "2021-12-01")]

    Returns
    -------
    dict
        Dictionary with computed statistics for Δu, Δv, and GDP
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading empirical data for period {start_date} to {end_date}...")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from calibration.empirical import load_empirical_data

    # Load empirical data using the standard pipeline
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from calibration.empirical import load_empirical_data

    # Load full empirical data
    empirical_data = load_empirical_data()

    # Filter to Antoine's reference period
    df = empirical_data.loc[start_date:end_date].copy()

    # Apply exclusion ranges if specified
    if exclude_ranges:
        mask = pd.Series(True, index=df.index)
        for excl_start, excl_end in exclude_ranges:
            mask = mask & ~df.index.to_series().between(
                pd.Timestamp(excl_start),
                pd.Timestamp(excl_end)
            )
        df = df[mask]

    # Extract data and compute changes
    emp_u_levels = df["u_obs"].dropna().values.astype(float)
    emp_v_levels = df["v_obs"].dropna().values.astype(float)
    gdp_signal = df["G"].dropna().values.astype(float)

    # Compute first differences (changes)
    emp_u = np.diff(emp_u_levels)
    emp_v = np.diff(emp_v_levels)

    print(f"\nPeriod: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Observations: {len(emp_u)} (monthly changes)")

    # Compute statistics
    results = {}

    # === UNEMPLOYMENT CHANGE (Δu) ===
    print("\n" + "="*70)
    print("UNEMPLOYMENT CHANGE (Δu)")
    print("="*70)
    u_upper_alpha, u_lower_alpha = _hill_tail_exponent(emp_u, tail_fraction=0.1)
    u_skew = scipy_skew(emp_u)

    results["du"] = {
        "mean": float(np.mean(emp_u)),
        "std": float(np.std(emp_u)),
        "min": float(np.min(emp_u)),
        "max": float(np.max(emp_u)),
        "skew": float(u_skew),
        "upper_tail_exponent": float(u_upper_alpha) if not np.isnan(u_upper_alpha) else None,
        "lower_tail_exponent": float(u_lower_alpha) if not np.isnan(u_lower_alpha) else None,
    }

    print(f"  Mean: {np.mean(emp_u):.6f}")
    print(f"  Std:  {np.std(emp_u):.6f}")
    print(f"  Range: [{np.min(emp_u):.6f}, {np.max(emp_u):.6f}]")
    print(f"  Skewness: {u_skew:.4f}")
    print(f"  Upper tail exponent (α): {u_upper_alpha:.4f}" if not np.isnan(u_upper_alpha) else "  Upper tail exponent: N/A")
    print(f"  Lower tail exponent (α): {u_lower_alpha:.4f}" if not np.isnan(u_lower_alpha) else "  Lower tail exponent: N/A")

    # === VACANCY CHANGE (Δv) ===
    print("\n" + "="*70)
    print("VACANCY CHANGE (Δv)")
    print("="*70)
    v_upper_alpha, v_lower_alpha = _hill_tail_exponent(emp_v, tail_fraction=0.1)
    v_skew = scipy_skew(emp_v)

    results["dv"] = {
        "mean": float(np.mean(emp_v)),
        "std": float(np.std(emp_v)),
        "min": float(np.min(emp_v)),
        "max": float(np.max(emp_v)),
        "skew": float(v_skew),
        "upper_tail_exponent": float(v_upper_alpha) if not np.isnan(v_upper_alpha) else None,
        "lower_tail_exponent": float(v_lower_alpha) if not np.isnan(v_lower_alpha) else None,
    }

    print(f"  Mean: {np.mean(emp_v):.6f}")
    print(f"  Std:  {np.std(emp_v):.6f}")
    print(f"  Range: [{np.min(emp_v):.6f}, {np.max(emp_v):.6f}]")
    print(f"  Skewness: {v_skew:.4f}")
    print(f"  Upper tail exponent (α): {v_upper_alpha:.4f}" if not np.isnan(v_upper_alpha) else "  Upper tail exponent: N/A")
    print(f"  Lower tail exponent (α): {v_lower_alpha:.4f}" if not np.isnan(v_lower_alpha) else "  Lower tail exponent: N/A")

    # === GDP LOG-GROWTH (G) ===
    print("\n" + "="*70)
    print("GDP LOG-GROWTH (G)")
    print("="*70)
    g_upper_alpha, g_lower_alpha = _hill_tail_exponent(gdp_signal, tail_fraction=0.1)
    g_skew = scipy_skew(gdp_signal)

    results["gdp"] = {
        "mean": float(np.mean(gdp_signal)),
        "std": float(np.std(gdp_signal)),
        "min": float(np.min(gdp_signal)),
        "max": float(np.max(gdp_signal)),
        "skew": float(g_skew),
        "upper_tail_exponent": float(g_upper_alpha) if not np.isnan(g_upper_alpha) else None,
        "lower_tail_exponent": float(g_lower_alpha) if not np.isnan(g_lower_alpha) else None,
    }

    print(f"  Mean: {np.mean(gdp_signal):.6f}")
    print(f"  Std:  {np.std(gdp_signal):.6f}")
    print(f"  Range: [{np.min(gdp_signal):.6f}, {np.max(gdp_signal):.6f}]")
    print(f"  Skewness: {g_skew:.4f}")
    print(f"  Upper tail exponent (α): {g_upper_alpha:.4f}" if not np.isnan(g_upper_alpha) else "  Upper tail exponent: N/A")
    print(f"  Lower tail exponent (α): {g_lower_alpha:.4f}" if not np.isnan(g_lower_alpha) else "  Lower tail exponent: N/A")

    # Save results to text file
    summary_path = os.path.join(output_dir, "antoine_comparison.txt")
    with open(summary_path, "w") as f:
        f.write("ANTOINE REFERENCE PERIOD COMPARISON (2000-2018)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Data period: {df.index[0].date()} to {df.index[-1].date()} ({len(emp_u)} monthly changes)\n\n")

        f.write("UNEMPLOYMENT CHANGE (Δu)\n")
        f.write(f"  Mean: {results['du']['mean']:.6f}\n")
        f.write(f"  Std:  {results['du']['std']:.6f}\n")
        f.write(f"  Range: [{results['du']['min']:.6f}, {results['du']['max']:.6f}]\n")
        f.write(f"  Skewness: {results['du']['skew']:.4f}\n")
        if results['du']['upper_tail_exponent'] is not None:
            f.write(f"  Upper tail exponent: {results['du']['upper_tail_exponent']:.4f}\n")
        if results['du']['lower_tail_exponent'] is not None:
            f.write(f"  Lower tail exponent: {results['du']['lower_tail_exponent']:.4f}\n")
        f.write("\n")

        f.write("VACANCY CHANGE (Δv)\n")
        f.write(f"  Mean: {results['dv']['mean']:.6f}\n")
        f.write(f"  Std:  {results['dv']['std']:.6f}\n")
        f.write(f"  Range: [{results['dv']['min']:.6f}, {results['dv']['max']:.6f}]\n")
        f.write(f"  Skewness: {results['dv']['skew']:.4f}\n")
        if results['dv']['upper_tail_exponent'] is not None:
            f.write(f"  Upper tail exponent: {results['dv']['upper_tail_exponent']:.4f}\n")
        if results['dv']['lower_tail_exponent'] is not None:
            f.write(f"  Lower tail exponent: {results['dv']['lower_tail_exponent']:.4f}\n")
        f.write("\n")

        f.write("GDP LOG-GROWTH (G)\n")
        f.write(f"  Mean: {results['gdp']['mean']:.6f}\n")
        f.write(f"  Std:  {results['gdp']['std']:.6f}\n")
        f.write(f"  Range: [{results['gdp']['min']:.6f}, {results['gdp']['max']:.6f}]\n")
        f.write(f"  Skewness: {results['gdp']['skew']:.4f}\n")
        if results['gdp']['upper_tail_exponent'] is not None:
            f.write(f"  Upper tail exponent: {results['gdp']['upper_tail_exponent']:.4f}\n")
        if results['gdp']['lower_tail_exponent'] is not None:
            f.write(f"  Lower tail exponent: {results['gdp']['lower_tail_exponent']:.4f}\n")

    print(f"\nResults saved to {summary_path}")
    return results


if __name__ == "__main__":
    # Compute three scenarios
    print("=" * 70)
    print("Full data range (2000-present)")
    print("=" * 70)
    compute_antoine_comparison(
        start_date="2000-01-01",
        end_date="2026-12-31",
        output_dir="output/antoine_comparison/full_data"
    )

    print("\n" + "=" * 70)
    print("Data range to acute COVID (2000-2020-06)")
    print("=" * 70)
    compute_antoine_comparison(
        start_date="2000-01-01",
        end_date="2020-06-30",
        output_dir="output/antoine_comparison/to_acute_covid"
    )

    print("\n" + "=" * 70)
    print("Full data range minus acute COVID (2000-present, excluding 2020-03 to 2020-06)")
    print("=" * 70)
    compute_antoine_comparison(
        start_date="2000-01-01",
        end_date="2026-12-31",
        exclude_ranges=[("2020-03-01", "2020-06-01")],
        output_dir="output/antoine_comparison/full_minus_acute_covid"
    )
