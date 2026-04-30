# skewculator_appendix_match.py

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew
from statsmodels.tsa.filters.hp_filter import hpfilter

GDP_PATH = Path("GDP_us_long.csv")
BEV_PATH = Path("us_bev_curve.csv")


def one_sided_hp_cycle(series, lamb=1600, min_obs=20):
    """
    One-sided HP cycle:
    at time t, fit HP filter using data up to t only, and keep the last cycle value.
    """
    x = pd.Series(series).astype(float).reset_index(drop=True)
    out = np.full(len(x), np.nan)

    for t in range(min_obs - 1, len(x)):
        y = x.iloc[: t + 1]
        cycle, trend = hpfilter(y, lamb=lamb)
        out[t] = cycle.iloc[-1]

    return pd.Series(out)


def load_inputs():
    gdp = pd.read_csv(GDP_PATH)
    gdp["observation_date"] = pd.to_datetime(gdp["observation_date"])
    gdp = gdp.sort_values("observation_date").reset_index(drop=True)

    bev = pd.read_csv(BEV_PATH).rename(columns={
        "unemployment rate": "u",
        "vacancy rate": "v",
    })

    if "date" in bev.columns:
        bev["date"] = pd.to_datetime(bev["date"])
        bev = bev.set_index("date").sort_index()
    else:
        bev.index = pd.date_range("1951-01-01", periods=len(bev), freq="MS")

    return gdp, bev


def build_appendix_gdp_signal(gdp):
    q = gdp.copy()
    if "GDP" not in q.columns:
        non_date_cols = [c for c in q.columns if c != "observation_date"]
        q = q.rename(columns={non_date_cols[0]: "GDP"})

    q = q[["observation_date", "GDP"]].dropna().copy()
    q["GDP"] = q["GDP"].astype(float)

    # Appendix: quarterly cyclical component by one-sided HP filter
    q["cycle_q"] = one_sided_hp_cycle(q["GDP"], lamb=1600, min_obs=20)
    q = q.set_index("observation_date")

    # Linearly interpolate quarterly cycle to monthly
    monthly_index = pd.date_range(q.index.min(), q.index.max(), freq="MS")
    monthly_cycle = (
        q["cycle_q"]
        .reindex(q.index.union(monthly_index))
        .interpolate(method="time")
        .reindex(monthly_index)
    )

    # Appendix: standardized with 120-month rolling std
    rolling_std = monthly_cycle.rolling(120, min_periods=120).std()
    g_month = monthly_cycle / rolling_std

    # Appendix: lagged one month
    g_month = g_month.shift(1)

    return g_month


def compute_skews(bev, g_month):
    df = bev.copy()

    # Appendix: vacancies floored at 1e-4
    df["v"] = np.maximum(df["v"].astype(float), 1e-4)
    df["u"] = df["u"].astype(float)
    df["g"] = g_month.reindex(df.index)

    df = df.dropna(subset=["u", "v", "g"]).copy()

    du = np.diff(df["u"].to_numpy())
    dv = np.diff(df["v"].to_numpy())
    dg = np.diff(df["g"].to_numpy())

    out = pd.DataFrame({
        "series": ["Δu", "Δv", "Δ transformed GDP"],
        "n": [len(du), len(dv), len(dg)],
        "skew_bias_false": [
            skew(du, bias=False, nan_policy="omit"),
            skew(dv, bias=False, nan_policy="omit"),
            skew(dg, bias=False, nan_policy="omit"),
        ],
        "skew_bias_true": [
            skew(du, bias=True, nan_policy="omit"),
            skew(dv, bias=True, nan_policy="omit"),
            skew(dg, bias=True, nan_policy="omit"),
        ],
    })

    return df, out


def diff_series(df):
    du = pd.Series(np.diff(df["u"].to_numpy()), index=df.index[1:], name="du")
    dv = pd.Series(np.diff(df["v"].to_numpy()), index=df.index[1:], name="dv")
    dg = pd.Series(np.diff(df["g"].to_numpy()), index=df.index[1:], name="dg")
    return pd.concat([du, dv, dg], axis=1)


def skew_table_from_diffs(diffs, label):
    out = pd.DataFrame({
        "sample": [label, label, label],
        "series": ["Δu", "Δv", "Δ transformed GDP"],
        "n": [len(diffs["du"]), len(diffs["dv"]), len(diffs["dg"])],
        "skew_bias_false": [
            skew(diffs["du"], bias=False, nan_policy="omit"),
            skew(diffs["dv"], bias=False, nan_policy="omit"),
            skew(diffs["dg"], bias=False, nan_policy="omit"),
        ],
        "skew_bias_true": [
            skew(diffs["du"], bias=True, nan_policy="omit"),
            skew(diffs["dv"], bias=True, nan_policy="omit"),
            skew(diffs["dg"], bias=True, nan_policy="omit"),
        ],
    })
    return out


def main():
    gdp, bev = load_inputs()
    g_month = build_appendix_gdp_signal(gdp)
    df, _ = compute_skews(bev, g_month)

    diffs = diff_series(df)

    print("=== Full sample ===")
    print(skew_table_from_diffs(diffs, "Full sample").to_string(index=False))

    mask_no_covid = (diffs.index < "2020-03-01") | (diffs.index > "2020-06-01")
    print("\n=== Excluding 2020-03 through 2020-06 ===")
    print(skew_table_from_diffs(diffs.loc[mask_no_covid], "Exclude 2020-03 to 2020-06").to_string(index=False))

    mask_no_broad = (diffs.index < "2020-03-01") | (diffs.index > "2021-12-01")
    print("\n=== Excluding 2020-03 through 2021-12 ===")
    print(skew_table_from_diffs(diffs.loc[mask_no_broad], "Exclude 2020-03 to 2021-12").to_string(index=False))

    print("\nTop 10 positive Δu:")
    print(diffs["du"].sort_values(ascending=False).head(10).to_string())
if __name__ == "__main__":
    main()