"""
Fetch FRED series via the FRED REST API.

A FRED API key is required. Set the environment variable FRED_API_KEY,
or pass api_key= directly.  Register at: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
import time

import pandas as pd
import requests

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "gdp_real_level":       "GDPC1",             # Real GDP, billions chained 2017 USD, quarterly SA
    "gdp_real_growth_rate": "A191RL1Q225SBEA",   # Real GDP growth rate, quarterly SA, %
}


def fetch_fred_series(
    series_dict=None,
    start_date="2000-01-01",
    end_date="2026-12-31",
    api_key= "211112645b4ec0743e021c94cd7f8bde",
    retries=3,
    retry_delay=5.0,
):
    """
    Fetch one or more FRED series and return a DataFrame.

    Parameters
    ----------
    series_dict : dict, optional
        Mapping of {column_name: fred_series_id}. Defaults to FRED_SERIES.
    start_date : str
        ISO date string for the start of the observation window.
    end_date : str
        ISO date string for the end of the observation window.
    api_key : str, optional
        FRED API key. Falls back to FRED_API_KEY env var.
    retries : int
        Number of retry attempts on transient errors.
    retry_delay : float
        Seconds between retries.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex at the native frequency of each series (quarterly for GDP).
        One column per series; missing observations are NaN.
    """
    if series_dict is None:
        series_dict = FRED_SERIES

    key = api_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        raise ValueError(
            "FRED API key required. Set the FRED_API_KEY environment variable "
            "or pass api_key= to fetch_fred_series()."
        )


    frames = {}

    for col_name, series_id in series_dict.items():
        params = {
            "series_id": series_id,
            "observation_start": start_date,
            "observation_end": end_date,
            "api_key": key,
            "file_type": "json",
        }

        for attempt in range(retries):
            try:
                resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt < retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"FRED API request failed for {series_id} after {retries} attempts: {exc}"
                    ) from exc

        body = resp.json()
        if "observations" not in body:
            raise RuntimeError(
                f"Unexpected FRED response for {series_id}: {body.get('error_message', body)}"
            )

        records = {}
        for obs in body["observations"]:
            try:
                val = float(obs["value"])
            except (ValueError, TypeError):
                val = float("nan")
            records[pd.Timestamp(obs["date"])] = val

        frames[col_name] = pd.Series(records, name=col_name)

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df.sort_index(inplace=True)
    return df


def save_fred_data(output_path, **kwargs):
    """Fetch FRED data and save to CSV at output_path."""
    df = fetch_fred_series(**kwargs)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path)
    return df
