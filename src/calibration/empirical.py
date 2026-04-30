"""
PHASE 1: Load and align empirical data (u, v, G).

Loads JOLTS vacancy rate, BLS unemployment, and macro data (with pre-computed GDP signal),
and returns one aligned monthly DataFrame.
"""

import glob
import os

import numpy as np
import pandas as pd


def _load_jolts_series(series_ids_map):
    """
    Load JOLTS Excel files and parse into a DataFrame.

    Parameters
    ----------
    series_ids_map : dict
        Mapping of {series_id: column_name}

    Returns
    -------
    pd.DataFrame
        Monthly index, columns for each requested series.
    """
    frames = []

    for f in sorted(glob.glob("data/jolts/*.xlsx")):
        try:
            meta = pd.read_excel(f, nrows=12, header=None, engine="openpyxl")
            sid = str(meta.iloc[3, 1]).strip()
        except Exception:
            continue

        if sid not in series_ids_map:
            continue

        col_name = series_ids_map[sid]

        try:
            df = pd.read_excel(f, skiprows=12, header=None, engine="openpyxl")
            df.columns = [
                "year", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            ]
            df = df[df["year"].apply(lambda x: isinstance(x, (int, float)) and 1999 < x < 2027)]

            melted = df.melt(id_vars="year", var_name="month", value_name=col_name)
            melted["date"] = pd.to_datetime(
                melted["year"].astype(str) + "-" + melted["month"],
                format="%Y-%b",
                errors="coerce"
            )
            melted = melted.dropna(subset=[col_name])
            melted = melted.set_index("date").sort_index()[[col_name]]
            frames.append(melted)
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}")
            continue

    if not frames:
        raise ValueError("No JOLTS data loaded")

    jolts = pd.concat(frames, axis=1)
    jolts.index.name = "date"
    return jolts


def load_antoine_uv(
    path="data/us_bev_curve_antoine.csv",
    start_date="1951-01-01",
    end_date=None,
):
    """
    Load Antoine's historical u/v series (1951-01 to 2023-12, monthly).

    The CSV has no date column; rows are assigned a monthly DatetimeIndex
    starting 1951-01-01. Values are in percent and are divided by 100.

    Returns
    -------
    pd.DataFrame with columns u_obs, v_obs (fractions) and a monthly DatetimeIndex.
    """
    df = pd.read_csv(path, header=0)
    df.columns = ["u_obs", "v_obs"]
    df["u_obs"] /= 100.0
    df["v_obs"] /= 100.0
    df.index = pd.date_range("1951-01-01", periods=len(df), freq="MS")
    df.index.name = "date"
    df = df.loc[start_date:]
    if end_date is not None:
        df = df.loc[:end_date]
    return df


DEFAULT_CYCLE_BREAKPOINTS = ["2008-09-01", "2020-03-01"]


def load_empirical_data(
    start_date="2000-01-01",
    end_date="2026-12-31",
    output_path="data/empirical_data.csv",
    gdp_lag_months=0,
    use_antoine_uv=False,
    antoine_uv_path="data/us_bev_curve_antoine.csv",
    gdp_signal_type="cum_demeaned_piecewise",
    cycle_breakpoints=None,
):
    """
    Load and align empirical u, v, G from JOLTS, BLS, FRED.

    Parameters
    ----------
    start_date : str
        Start of window (ISO format)
    end_date : str
        End of window (ISO format)
    output_path : str
        Path to save aligned CSV
    gdp_lag_months : int
        Months to lag the GDP signal before passing to the model.
        Antoine (beveridgev0.pdf §5.4) finds an 18-month backward lag
        gives the best fit: G(t) reflects GDP conditions from t-18.
        Set to 0 to disable.
    use_antoine_uv : bool
        If True, replace u_obs / v_obs with Antoine's historical series
        (data/us_bev_curve_antoine.csv, 1951-01 to 2023-12).  The GDP
        signal G(t) is still taken from FRED.  Useful for extending the
        sample back before JOLTS begins in 2001.
    antoine_uv_path : str
        Path to Antoine's u/v CSV (only used when use_antoine_uv=True).
    gdp_signal_type : str
        Which GDP transformation to assign to column G:
          - "growth": monthly log-growth (the original signal).
          - "cum_demeaned": cumulative integral of de-meaned monthly log-growth
              over the returned window. Equivalent to log-GDP minus a linear
              trend. Mean-reverting; no permanent drift in long-run u.
          - "cum_demeaned_piecewise": same as cum_demeaned but applied
              independently to each segment delimited by cycle_breakpoints.
              Each segment has its own mean subtracted, then integrated, and
              the integral resets to zero at each breakpoint. Treats each
              economic cycle as a separate episode.
    cycle_breakpoints : list of ISO date strings, optional
        Breakpoints between economic cycles (only used when gdp_signal_type
        is "cum_demeaned_piecewise"). Each is the time-midpoint between a
        bust's peak and trough. Defaults to ["2008-09-01", "2020-03-01"].

    Returns
    -------
    pd.DataFrame
        Columns: u_obs, v_obs, G, plus diagnostic columns
    """
    # Load macro data (BLS + FRED)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data_fetch.loader import load_macro_data

    macro = load_macro_data(cache=True)

    # Load JOLTS rates
    jolts_map = {
        "JTS000000000000000JOR": "v_obs",  # Job openings rate
        "JTS000000000000000TSR": "sep_rate",  # Separation rate
    }
    jolts = _load_jolts_series(jolts_map)

    # Merge - keep gdp_log_growth_linear for GDP signal
    df = macro[["unemployment_rate", "gdp_real_level", "gdp_log_growth_linear"]].join(jolts, how="left")
    df["u_obs"] = pd.to_numeric(df["unemployment_rate"], errors="coerce") / 100.0
    df["v_obs"] = pd.to_numeric(df["v_obs"], errors="coerce") / 100.0
    df["s_obs"] = pd.to_numeric(df["sep_rate"], errors="coerce") / 100.0

    if use_antoine_uv:
        ant = load_antoine_uv(path=antoine_uv_path, start_date=start_date, end_date=end_date)
        df["u_obs"] = ant["u_obs"].reindex(df.index)
        df["v_obs"] = ant["v_obs"].reindex(df.index)

    # Use pre-computed GDP log-growth from loader
    # gdp_log_growth_linear is monthly log-growth with AR(2) noise
    g_growth = df["gdp_log_growth_linear"].copy()
    if gdp_lag_months > 0:
        g_growth = g_growth.shift(gdp_lag_months)
    df["G_growth"] = g_growth

    # Trim to window, hard-capped at Oct 2025 (last month with real GDP data)
    end_date = min(end_date, "2025-10-31")
    df = df.loc[start_date:end_date]

    # Build the chosen GDP signal on the windowed data. De-meaning has to be
    # done over the fit window itself so the integral starts and ends near
    # zero on the sample being fit.
    g_window = df["G_growth"]
    if gdp_signal_type == "growth":
        df["G"] = g_window
    elif gdp_signal_type == "cum_demeaned":
        demeaned = g_window - g_window.mean()
        df["G"] = demeaned.fillna(0).cumsum()
    elif gdp_signal_type == "cum_demeaned_piecewise":
        if cycle_breakpoints is None:
            cycle_breakpoints = DEFAULT_CYCLE_BREAKPOINTS
        breakpoints = [pd.Timestamp(b) for b in cycle_breakpoints]
        # Build segment edges that bracket the data: [start, bp1, bp2, ..., end+1]
        segment_edges = (
            [df.index[0]]
            + [b for b in breakpoints if df.index[0] < b <= df.index[-1]]
            + [df.index[-1] + pd.Timedelta(days=1)]
        )
        g_combined = pd.Series(0.0, index=df.index)
        for left, right in zip(segment_edges[:-1], segment_edges[1:]):
            mask = (df.index >= left) & (df.index < right)
            seg = g_window.loc[mask]
            if len(seg) == 0:
                continue
            demeaned = (seg - seg.mean()).fillna(0)
            g_combined.loc[mask] = demeaned.cumsum()
        df["G"] = g_combined
    else:
        raise ValueError(f"Unknown gdp_signal_type: {gdp_signal_type!r}")

    # Keep only essential columns
    df_out = df[["u_obs", "v_obs", "G", "s_obs", "gdp_real_level"]].copy()
    df_out.index.name = "date"

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    df_out.to_csv(output_path)
    print(f"Saved empirical data to {output_path}")
    print(f"  Shape: {df_out.shape}")
    print(f"  Date range: {df_out.index[0]} to {df_out.index[-1]}")
    print(f"  GDP lag: {gdp_lag_months} months")
    print(f"\nSummary statistics:")
    print(df_out.describe())

    return df_out


if __name__ == "__main__":
    df = load_empirical_data()
