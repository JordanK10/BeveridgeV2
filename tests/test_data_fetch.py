"""
Unit and integration tests for data_fetch.

Unit tests mock the HTTP layer and validate parsing logic.
Integration tests hit live APIs — they are skipped unless the relevant
environment variables (BLS_API_KEY, FRED_API_KEY) are set.

Run all:
    cd /Users/kempj/Documents/BeveridgeV2
    python -m pytest tests/test_data_fetch.py -v

Run only unit tests (no network):
    python -m pytest tests/test_data_fetch.py -v -m "not integration"
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# Make src importable from the tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_fetch.bls import (
    ALL_BLS_SERIES,
    CES_SERIES,
    CPS_SERIES,
    fetch_bls_series,
)
from data_fetch.fred import FRED_SERIES, fetch_fred_series
from data_fetch.loader import load_macro_data


# ---------------------------------------------------------------------------
# Helpers: minimal fake API responses
# ---------------------------------------------------------------------------

def _bls_response(series_ids, year="2024", months=("M01", "M02", "M03"), value="1234.5"):
    """Build a minimal BLS API v2 JSON response body."""
    return {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": sid,
                    "data": [
                        {"year": year, "period": m, "value": value, "footnotes": []}
                        for m in months
                    ],
                }
                for sid in series_ids
            ]
        },
    }


def _fred_response(dates_values):
    """Build a minimal FRED observations JSON response body."""
    return {
        "observations": [
            {"date": d, "value": str(v)} for d, v in dates_values
        ]
    }


# ---------------------------------------------------------------------------
# BLS unit tests
# ---------------------------------------------------------------------------

class TestBLSSeriesRegistry(unittest.TestCase):
    """Validate that the series dictionaries have the right shape."""

    def test_cps_has_required_keys(self):
        for key in ("unemployment_rate", "employment_level", "labor_force_level"):
            self.assertIn(key, CPS_SERIES, f"Missing CPS key: {key}")

    def test_ces_has_required_keys(self):
        expected = [
            "emp_total_nonfarm", "emp_construction", "emp_manufacturing",
            "emp_retail", "emp_leisure_hospitality", "emp_government",
        ]
        for key in expected:
            self.assertIn(key, CES_SERIES, f"Missing CES key: {key}")

    def test_all_series_ids_are_strings(self):
        for name, sid in ALL_BLS_SERIES.items():
            self.assertIsInstance(sid, str, f"{name} series ID is not a string")
            self.assertTrue(len(sid) > 0, f"{name} series ID is empty")

    def test_no_duplicate_series_ids(self):
        ids = list(ALL_BLS_SERIES.values())
        self.assertEqual(len(ids), len(set(ids)), "Duplicate BLS series IDs found")


class TestFetchBLSSeries(unittest.TestCase):
    """Test BLS fetch + parse logic with mocked HTTP."""

    def _mock_post(self, series_ids, **kwargs):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _bls_response(
            series_ids, year="2024", months=("M01", "M06", "M12")
        )
        return mock_resp

    @patch("data_fetch.bls.requests.post")
    def test_returns_dataframe(self, mock_post):
        mock_post.side_effect = lambda *a, **kw: self._mock_post(
            [CPS_SERIES["unemployment_rate"]]
        )
        df = fetch_bls_series(
            series_dict={"unemployment_rate": CPS_SERIES["unemployment_rate"]},
            start_year=2024,
            end_year=2024,
        )
        self.assertIsInstance(df, pd.DataFrame)

    @patch("data_fetch.bls.requests.post")
    def test_column_names_match_series_dict_keys(self, mock_post):
        series_dict = {
            "unemployment_rate": CPS_SERIES["unemployment_rate"],
            "employment_level":  CPS_SERIES["employment_level"],
        }
        mock_post.side_effect = lambda *a, **kw: self._mock_post(
            list(series_dict.values())
        )
        df = fetch_bls_series(series_dict=series_dict, start_year=2024, end_year=2024)
        for col in series_dict:
            self.assertIn(col, df.columns, f"Column {col!r} missing from result")

    @patch("data_fetch.bls.requests.post")
    def test_index_is_monthly_datetimeindex(self, mock_post):
        mock_post.side_effect = lambda *a, **kw: self._mock_post(
            [CPS_SERIES["unemployment_rate"]]
        )
        df = fetch_bls_series(
            series_dict={"unemployment_rate": CPS_SERIES["unemployment_rate"]},
            start_year=2024,
            end_year=2024,
        )
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        # Monthly frequency — consecutive dates differ by ~1 month
        diffs = df.index[1:] - df.index[:-1]
        for d in diffs:
            self.assertLessEqual(d.days, 31)
            self.assertGreaterEqual(d.days, 28)

    @patch("data_fetch.bls.requests.post")
    def test_parsed_values_are_float(self, mock_post):
        mock_post.side_effect = lambda *a, **kw: self._mock_post(
            [CPS_SERIES["unemployment_rate"]]
        )
        df = fetch_bls_series(
            series_dict={"unemployment_rate": CPS_SERIES["unemployment_rate"]},
            start_year=2024,
            end_year=2024,
        )
        finite = df["unemployment_rate"].dropna()
        self.assertTrue(len(finite) > 0, "No non-NaN values parsed")
        self.assertTrue(
            all(isinstance(v, float) for v in finite),
            "Not all parsed values are float",
        )

    @patch("data_fetch.bls.requests.post")
    def test_api_error_status_raises(self, mock_post):
        bad_resp = MagicMock()
        bad_resp.raise_for_status.return_value = None
        bad_resp.json.return_value = {"status": "REQUEST_FAILED", "message": ["Bad series"]}
        mock_post.return_value = bad_resp
        with self.assertRaises(RuntimeError):
            fetch_bls_series(
                series_dict={"unemployment_rate": CPS_SERIES["unemployment_rate"]},
                start_year=2024,
                end_year=2024,
            )

    @patch("data_fetch.bls.requests.post")
    def test_annual_period_rows_are_skipped(self, mock_post):
        """Rows with period='M13' (annual average) must not appear in output."""
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [{
                    "seriesID": CPS_SERIES["unemployment_rate"],
                    "data": [
                        {"year": "2024", "period": "M01", "value": "4.0", "footnotes": []},
                        {"year": "2024", "period": "M13", "value": "4.1", "footnotes": []},
                    ],
                }]
            },
        }
        mock_post.return_value = resp
        df = fetch_bls_series(
            series_dict={"unemployment_rate": CPS_SERIES["unemployment_rate"]},
            start_year=2024,
            end_year=2024,
        )
        # Only Jan 2024 should be non-NaN (M13 skipped)
        non_nan = df["unemployment_rate"].dropna()
        self.assertEqual(len(non_nan), 1)
        self.assertEqual(non_nan.index[0], pd.Timestamp("2024-01-01"))


# ---------------------------------------------------------------------------
# FRED unit tests
# ---------------------------------------------------------------------------

class TestFetchFREDSeries(unittest.TestCase):
    """Test FRED fetch + parse logic with mocked HTTP."""

    @patch("data_fetch.fred.requests.get")
    def test_returns_dataframe(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = _fred_response([
            ("2024-01-01", 22000.0), ("2024-04-01", 22200.0),
        ])
        df = fetch_fred_series(
            series_dict={"gdp_real_level": "GDPC1"},
            api_key="test_key",
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("gdp_real_level", df.columns)

    @patch("data_fetch.fred.requests.get")
    def test_index_is_datetimeindex(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = _fred_response([
            ("2024-01-01", 22000.0), ("2024-04-01", 22200.0),
        ])
        df = fetch_fred_series(
            series_dict={"gdp_real_level": "GDPC1"}, api_key="test_key"
        )
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    @patch("data_fetch.fred.requests.get")
    def test_values_are_float(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = _fred_response([
            ("2024-01-01", 22000.0),
        ])
        df = fetch_fred_series(
            series_dict={"gdp_real_level": "GDPC1"}, api_key="test_key"
        )
        val = df["gdp_real_level"].dropna().iloc[0]
        self.assertIsInstance(val, float)
        self.assertAlmostEqual(val, 22000.0)

    def test_missing_api_key_raises(self):
        old = os.environ.pop("FRED_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                fetch_fred_series(api_key=None)
        finally:
            if old is not None:
                os.environ["FRED_API_KEY"] = old

    @patch("data_fetch.fred.requests.get")
    def test_non_numeric_value_becomes_nan(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = _fred_response([
            ("2024-01-01", "."),   # FRED uses "." for missing
        ])
        import math
        df = fetch_fred_series(
            series_dict={"gdp_real_level": "GDPC1"}, api_key="test_key"
        )
        val = df["gdp_real_level"].iloc[0]
        self.assertTrue(math.isnan(val))

    @patch("data_fetch.fred.requests.get")
    def test_missing_observations_key_raises(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {"error_message": "Bad API key"}
        with self.assertRaises(RuntimeError):
            fetch_fred_series(
                series_dict={"gdp_real_level": "GDPC1"}, api_key="bad_key"
            )


# ---------------------------------------------------------------------------
# Loader unit tests
# ---------------------------------------------------------------------------

class TestLoadMacroData(unittest.TestCase):
    """Test loader logic with mocked fetch functions."""

    def _make_bls_df(self):
        idx = pd.date_range("2024-01-01", periods=12, freq="MS")
        return pd.DataFrame(
            {
                "unemployment_rate": [4.0] * 12,
                "employment_level":  [160000.0] * 12,
                "labor_force_level": [167000.0] * 12,
            },
            index=idx,
        )

    def _make_fred_df(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="QS")
        return pd.DataFrame(
            {"gdp_real_level": [22000.0, 22200.0, 22400.0, 22600.0],
             "gdp_real_growth_rate": [2.5, 2.8, 3.0, 2.9]},
            index=idx,
        )

    @patch("data_fetch.loader.fetch_fred_series")
    @patch("data_fetch.loader.fetch_bls_series")
    def test_returns_dataframe_with_expected_columns(self, mock_bls, mock_fred):
        mock_bls.return_value = self._make_bls_df()
        mock_fred.return_value = self._make_fred_df()
        df = load_macro_data(cache_path="/tmp/_test_macro.csv", fred_api_key="x")
        for col in ("unemployment_rate", "employment_level", "labor_force_level",
                    "gdp_real_level", "u"):
            self.assertIn(col, df.columns, f"Missing column: {col}")
        if os.path.exists("/tmp/_test_macro.csv"):
            os.remove("/tmp/_test_macro.csv")

    @patch("data_fetch.loader.fetch_fred_series")
    @patch("data_fetch.loader.fetch_bls_series")
    def test_u_column_is_rate_not_percent(self, mock_bls, mock_fred):
        mock_bls.return_value = self._make_bls_df()
        mock_fred.return_value = self._make_fred_df()
        df = load_macro_data(cache_path="/tmp/_test_macro2.csv", fred_api_key="x")
        self.assertTrue((df["u"].dropna() < 1.0).all(), "u column should be a fraction < 1")
        if os.path.exists("/tmp/_test_macro2.csv"):
            os.remove("/tmp/_test_macro2.csv")

    @patch("data_fetch.loader.fetch_fred_series")
    @patch("data_fetch.loader.fetch_bls_series")
    def test_cache_is_written(self, mock_bls, mock_fred):
        mock_bls.return_value = self._make_bls_df()
        mock_fred.return_value = self._make_fred_df()
        path = "/tmp/_test_macro_cache.csv"
        if os.path.exists(path):
            os.remove(path)
        load_macro_data(cache_path=path, fred_api_key="x")
        self.assertTrue(os.path.exists(path), "Cache file was not written")
        os.remove(path)

    @patch("data_fetch.loader.fetch_fred_series")
    @patch("data_fetch.loader.fetch_bls_series")
    def test_cache_is_loaded_without_fetching(self, mock_bls, mock_fred):
        mock_bls.return_value = self._make_bls_df()
        mock_fred.return_value = self._make_fred_df()
        path = "/tmp/_test_macro_cache2.csv"
        load_macro_data(cache_path=path, fred_api_key="x")
        # Reset mocks — second call should not invoke them
        mock_bls.reset_mock()
        mock_fred.reset_mock()
        load_macro_data(cache_path=path, cache=True, fred_api_key="x")
        mock_bls.assert_not_called()
        mock_fred.assert_not_called()
        os.remove(path)


# ---------------------------------------------------------------------------
# Integration tests (skipped unless API keys are set)
# ---------------------------------------------------------------------------

_HAS_BLS_KEY  = bool(os.environ.get("BLS_API_KEY"))
_HAS_FRED_KEY = bool(os.environ.get("FRED_API_KEY"))


@unittest.skipUnless(_HAS_BLS_KEY, "BLS_API_KEY not set — skipping integration test")
class TestBLSIntegration(unittest.TestCase):
    """Live BLS API call — validates real data shape and plausible values."""

    @classmethod
    def setUpClass(cls):
        cls.df = fetch_bls_series(
            series_dict={
                "unemployment_rate": CPS_SERIES["unemployment_rate"],
                "labor_force_level": CPS_SERIES["labor_force_level"],
            },
            start_year=2020,
            end_year=2023,
        )

    def test_shape(self):
        self.assertGreater(len(self.df), 0, "DataFrame is empty")
        self.assertIn("unemployment_rate", self.df.columns)

    def test_unemployment_rate_in_plausible_range(self):
        u = self.df["unemployment_rate"].dropna()
        self.assertTrue((u >= 1.0).all(), "Some unemployment rates below 1%")
        self.assertTrue((u <= 20.0).all(), "Some unemployment rates above 20%")

    def test_no_all_nan_columns(self):
        for col in self.df.columns:
            self.assertFalse(
                self.df[col].isna().all(),
                f"Column {col!r} is entirely NaN",
            )

    def test_date_range_coverage(self):
        self.assertIn(pd.Timestamp("2020-01-01"), self.df.index)
        self.assertIn(pd.Timestamp("2023-12-01"), self.df.index)


@unittest.skipUnless(_HAS_FRED_KEY, "FRED_API_KEY not set — skipping integration test")
class TestFREDIntegration(unittest.TestCase):
    """Live FRED API call — validates real data shape and plausible values."""

    @classmethod
    def setUpClass(cls):
        cls.df = fetch_fred_series(start_date="2010-01-01", end_date="2024-12-31")

    def test_shape(self):
        self.assertGreater(len(self.df), 0, "DataFrame is empty")

    def test_expected_columns_present(self):
        for col in ("gdp_real_level", "gdp_real_growth_rate"):
            self.assertIn(col, self.df.columns)

    def test_gdp_level_positive(self):
        gdp = self.df["gdp_real_level"].dropna()
        self.assertTrue((gdp > 0).all(), "GDP level contains non-positive values")

    def test_no_all_nan_columns(self):
        for col in self.df.columns:
            self.assertFalse(
                self.df[col].isna().all(),
                f"Column {col!r} is entirely NaN",
            )

    def test_gdp_quarterly_spacing(self):
        """Real GDP (GDPC1) is quarterly; consecutive observations ~90 days apart."""
        gdp_idx = self.df["gdp_real_level"].dropna().index
        diffs = [(gdp_idx[i+1] - gdp_idx[i]).days for i in range(len(gdp_idx)-1)]
        for d in diffs:
            self.assertGreaterEqual(d, 28)
            self.assertLessEqual(d, 95)


if __name__ == "__main__":
    unittest.main()
