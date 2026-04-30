"""
Load, align, and cache all macro data needed for Beveridge calibration.

Fetches BLS (CPS + CES), FRED (GDP), and JOLTS (vacancy) data, aligns them to a common
monthly DatetimeIndex, and saves the result to data/macro_data.csv.

Usage
-----
    from data_fetch.loader import load_macro_data
    df = load_macro_data()          # fetches live and caches
    df = load_macro_data(cache=True) # re-uses cache if present
"""

import glob
import os

import numpy as np
import pandas as pd

from .bls import fetch_bls_series
from .fred import fetch_fred_series

_DEFAULT_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "macro_data.csv"
)

def _load_jolts_vacancy(series_id="JTS000000000000000JOR"):
    """
    Load JOLTS vacancy rate from Excel files.

    Parameters
    ----------
    series_id : str
        Series ID for job openings rate (default: JTS000000000000000JOR)

    Returns
    -------
    pd.Series
        Monthly vacancy rate (fraction, 0-1)
    """
    frames = []

    for f in sorted(glob.glob("data/jolts/*.xlsx")):
        try:
            meta = pd.read_excel(f, nrows=12, header=None, engine="openpyxl")
            sid = str(meta.iloc[3, 1]).strip()
        except Exception:
            continue

        if sid != series_id:
            continue

        try:
            df = pd.read_excel(f, skiprows=12, header=None, engine="openpyxl")
            df.columns = [
                "year", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            ]
            # Skip the header row ("Year", "Jan", ...) and empty rows, keep only numeric years
            df = df[df["year"].apply(lambda x: isinstance(x, (int, float)) and 1999 < x < 2027)]

            melted = df.melt(id_vars="year", var_name="month", value_name="vacancy_rate")
            melted["date"] = pd.to_datetime(
                melted["year"].astype(str) + "-" + melted["month"],
                format="%Y-%b",
                errors="coerce"
            )
            melted = melted.dropna(subset=["vacancy_rate"])
            melted = melted.set_index("date").sort_index()[["vacancy_rate"]]
            frames.append(melted)
        except Exception as e:
            print(f"Warning: failed to load JOLTS from {f}: {e}")
            continue

    if frames:
        return pd.concat(frames, axis=0)["vacancy_rate"]
    else:
        return pd.Series(dtype=float)



def load_macro_data(
    cache_path=None,
    cache=False,
    start_year=1951,
    end_year=2026,
    bls_api_key=None,
    fred_api_key=None,
):
    """
    Fetch BLS and FRED data, interpolate quarterly GDP to monthly, and return
    a single aligned DataFrame.

    Parameters
    ----------
    cache_path : str, optional
        Path to the cache CSV. Defaults to data/macro_data.csv.
    cache : bool
        If True and the cache file exists, load from cache instead of fetching.
    start_year : int
        Year to start data collection. Defaults to 1951 for long-term GDP statistics.
    end_year : int
    bls_api_key : str, optional
    fred_api_key : str, optional

    Returns
    -------
    pd.DataFrame
        Monthly DatetimeIndex. Columns include all BLS CPS/CES series plus
        gdp_real_level and gdp_real_growth_rate (forward-filled to monthly),
        and vacancy_rate from JOLTS (as fraction, not percent).
        Note: BLS data typically starts 2000; GDP data available from 1951;
        JOLTS vacancy data typically starts 2000.
    """
    if cache_path is None:
        cache_path = _DEFAULT_CACHE

    if cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df

    # --- BLS ------------------------------------------------------------------
    # BLS API has a 240-observation limit per series, but fetch_bls_series now batches
    # requests across overlapping year windows to capture the full range.
    bls_kwargs = dict(start_year=max(start_year, 2000), end_year=end_year)
    if bls_api_key is not None:
        bls_kwargs["api_key"] = bls_api_key
    bls_df = fetch_bls_series(**bls_kwargs)

    # --- FRED -----------------------------------------------------------------
    fred_kwargs = dict(start_date=f"{start_year}-01-01", end_date=f"{end_year}-12-31")
    if fred_api_key is not None:
        fred_kwargs["api_key"] = fred_api_key
    fred_df = fetch_fred_series(**fred_kwargs)

    # True monthly index
    monthly_idx = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-01",
        freq="MS",
    )

    # Preserve quarterly GDP source for Antoine signal
    gdp_q = fred_df["gdp_real_level"].dropna().copy()

    # Linear interpolation of FRED quarterly data to monthly
    fred_monthly = fred_df.reindex(monthly_idx).interpolate(method="time")

    # Align BLS to same monthly index
    bls_monthly = bls_df.reindex(monthly_idx)

    # --- Load JOLTS vacancy data ------------------------------------------------
    jolts_vacancy = _load_jolts_vacancy()
    if not jolts_vacancy.empty:
        jolts_vacancy_monthly = jolts_vacancy.reindex(monthly_idx)
    else:
        jolts_vacancy_monthly = pd.Series(np.nan, index=monthly_idx)

    # --- Merge ----------------------------------------------------------------
    df = fred_monthly.join(bls_monthly, how="left")

    # Add vacancy rate as fraction
    df["v"] = pd.to_numeric(jolts_vacancy_monthly, errors="coerce") / 100.0

    # Antoine GDP signal
    df["gdp_real_level_monthly_linear"] = df["gdp_real_level"]
    df["gdp_real_growth_rate_monthly_linear"] = df["gdp_real_growth_rate"]
    
    df["gdp_log_growth_linear"] = np.log(df["gdp_real_level_monthly_linear"]).diff()
    # Derived columns useful for calibration
    # u = unemployment_rate / 100  (fraction, not percent)
    if "unemployment_rate" in df.columns:
        df["u"] = df["unemployment_rate"] / 100.0



    df.sort_index(inplace=True)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    df.to_csv(cache_path)
    print(f"Saved macro data to {cache_path}")

    return df


if __name__ == "__main__":
    df = load_macro_data()
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nColumns: {df.columns.tolist()}")
