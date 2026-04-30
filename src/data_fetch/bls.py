"""
Fetch BLS time series via the BLS Public Data API v2.

API key is optional but raises the rate limit from 25 to 500 queries/day.
Set the environment variable BLS_API_KEY to use a registered key.

Reference: https://www.bls.gov/developers/api_signature_v2.htm
"""

import json
import os
import time
import signal

import pandas as pd
import requests

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# CPS (unemployment / labor force)
CPS_SERIES = {
    "unemployment_rate":   "LNS14000000",  # U-3 unemployment rate, SA
    "employment_level":    "LNS12000000",  # Civilian employment level, SA (thousands)
    "labor_force_level":   "LNS11000000",  # Civilian labor force level, SA (thousands)
}

# CES (nonfarm payroll employment by sector, SA, thousands)
CES_SERIES = {
    "emp_total_nonfarm":          "CES0000000001",
    "emp_total_private":          "CES0500000001",
    "emp_construction":           "CES2000000001",
    "emp_manufacturing":          "CES3000000001",
    "emp_trade_transport_util":   "CES4000000001",
    "emp_retail":                 "CES4200000001",
    "emp_prof_business":          "CES6000000001",
    "emp_edu_health":             "CES6500000001",
    "emp_leisure_hospitality":    "CES7000000001",
    "emp_government":             "CES9000000001",
}

ALL_BLS_SERIES = {**CPS_SERIES, **CES_SERIES}

# BLS API allows at most 50 series per POST request.
_BATCH_SIZE = 50


def _timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("BLS fetch operation exceeded 10 second timeout")


def fetch_bls_series(
    series_dict=None,
    start_year=2000,
    end_year=2026,
    api_key="a877d8fd5c454ab5a608cafdf01cc1f1",
    retries=3,
    retry_delay=5.0,
    timeout_seconds=10,
):
    """
    Fetch one or more BLS series and return a monthly DatetimeIndex DataFrame.

    Parameters
    ----------
    series_dict : dict, optional
        Mapping of {column_name: bls_series_id}. Defaults to ALL_BLS_SERIES.
    start_year : int
        First year to request.
    end_year : int
        Last year to request (inclusive).
    api_key : str, optional
        BLS registered API key. Falls back to BLS_API_KEY env var, then unauthenticated.
    retries : int
        Number of retry attempts on transient HTTP errors.
    retry_delay : float
        Seconds to wait between retries.

    Returns
    -------
    pd.DataFrame
        Monthly frequency, DatetimeIndex (month-start), one column per series.
        Values are float; missing months are NaN.
    """
    if series_dict is None:
        series_dict = ALL_BLS_SERIES

    key = api_key or os.environ.get("BLS_API_KEY", "")
    name_by_id = {v: k for k, v in series_dict.items()}
    series_ids = list(series_dict.values())

    # Split series into batches of ≤50
    series_batches = [series_ids[i: i + _BATCH_SIZE] for i in range(0, len(series_ids), _BATCH_SIZE)]

    # BLS API returns ~240 observations per series. Split year range into non-overlapping
    # contiguous 5-year windows to ensure we capture the full range without gaps or duplication.
    year_ranges = []
    current_start = start_year
    while current_start < end_year:
        window_end = min(current_start + 5, end_year)
        year_ranges.append((current_start, window_end))
        current_start = window_end

    all_records = {}  # {series_id: {period_str: value}}

    # Set timeout handler
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        overall_start = time.time()
        print(f"[BLS] Starting fetch: {len(series_batches)} series batches × {len(year_ranges)} year ranges = {len(series_batches) * len(year_ranges)} requests")

        for batch_num, series_batch in enumerate(series_batches):
            for year_num, (batch_start_year, batch_end_year) in enumerate(year_ranges):
                request_start = time.time()
                request_desc = f"Batch {batch_num+1}/{len(series_batches)}, Years {batch_start_year}-{batch_end_year}"
                payload = {
                    "seriesid": series_batch,
                    "startyear": str(batch_start_year),
                    "endyear": str(batch_end_year),
                }
                if key:
                    payload["registrationkey"] = key

                api_start = time.time()
                for attempt in range(retries):
                    try:
                        resp = requests.post(
                            BLS_API_URL,
                            data=json.dumps(payload),
                            headers={"Content-type": "application/json"},
                            timeout=30,
                        )
                        resp.raise_for_status()
                        break
                    except requests.RequestException as exc:
                        if attempt < retries - 1:
                            time.sleep(retry_delay)
                        else:
                            raise RuntimeError(f"BLS API request failed after {retries} attempts: {exc}") from exc
                api_elapsed = time.time() - api_start

                body = resp.json()
                if body.get("status") != "REQUEST_SUCCEEDED":
                    message = body.get("message", body.get("status", "unknown error"))
                    raise RuntimeError(f"BLS API returned non-success status: {message}")

                parse_start = time.time()
                for series in body["Results"]["series"]:
                    sid = series["seriesID"]
                    if sid not in all_records:
                        all_records[sid] = {}
                    records = all_records[sid]
                    for obs in series["data"]:
                        year = obs["year"]
                        period = obs["period"]          # e.g. "M01" … "M12"
                        if not period.startswith("M"):
                            continue                     # skip annual rows
                        month = period[1:]              # "01" … "12"
                        try:
                            val = float(obs["value"].replace(",", ""))
                        except (ValueError, AttributeError):
                            val = float("nan")
                        # No overlaps, so just add the data point
                        period_str = f"{year}-{month}"
                        records[period_str] = val
                parse_elapsed = time.time() - parse_start
                request_elapsed = time.time() - request_start

                print(f"  [BLS] {request_desc}: API={api_elapsed:.2f}s, Parse={parse_elapsed:.2f}s, Total={request_elapsed:.2f}s")

        overall_elapsed = time.time() - overall_start
        print(f"[BLS] Completed in {overall_elapsed:.2f}s\n")
    except TimeoutError as e:
        print(f"[BLS] ERROR: {e}")
        raise
    finally:
        signal.alarm(0)  # Cancel alarm

    # Build a unified monthly DatetimeIndex spanning the full requested range
    idx = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-01",
        freq="MS",
    )
    df = pd.DataFrame(index=idx)
    df.index.name = "date"

    for sid, records in all_records.items():
        col = name_by_id.get(sid, sid)
        col_data = {}
        for period_str, val in records.items():
            try:
                col_data[pd.Timestamp(period_str)] = val
            except Exception:
                pass
        series_obj = pd.Series(col_data, name=col)
        df[col] = series_obj.reindex(idx)

    return df


def save_bls_data(output_path, **kwargs):
    """Fetch BLS data and save to CSV at output_path."""
    df = fetch_bls_series(**kwargs)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path)
    return df
