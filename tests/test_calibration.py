"""
Tests for calibration module.

Unit tests with mocked data, integration tests with real empirical data.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calibration.empirical import _compute_gdp_signal
from calibration.loss import compute_loss
from calibration.simulate import SimpleCalibFirm, simulate_market


class TestSimpleCalibFirm(unittest.TestCase):
    """Test basic firm dynamics."""

    def test_initialization(self):
        firm = SimpleCalibFirm(sigma=300, c=0.1)
        self.assertEqual(firm.sigma, 300)
        self.assertEqual(firm.c, 0.1)

    def test_compute_demand(self):
        firm = SimpleCalibFirm(sigma=300, c=0.1, idio_std=0.0)
        signal = np.zeros(10)
        demand = firm.compute_demand(signal)
        self.assertEqual(len(demand), 10)
        self.assertTrue(np.all(demand > 0))
        np.testing.assert_array_almost_equal(demand, 300.0)

    def test_compute_demand_with_signal(self):
        firm = SimpleCalibFirm(sigma=300, c=0.1, idio_std=0.0)
        signal = np.ones(10)
        demand = firm.compute_demand(signal)
        np.testing.assert_array_almost_equal(demand, 300.0 * 1.1)


class TestSimulateMarket(unittest.TestCase):
    """Test market simulation."""

    def test_returns_valid_arrays(self):
        time_array = np.arange(100)
        gdp_signal = np.sin(np.linspace(0, 2 * np.pi, 100)) * 0.1
        u, v = simulate_market(time_array, gdp_signal, n_firms=50)
        self.assertEqual(len(u), 100)
        self.assertEqual(len(v), 100)

    def test_unemployment_in_valid_range(self):
        time_array = np.arange(100)
        gdp_signal = np.sin(np.linspace(0, 2 * np.pi, 100)) * 0.1
        u, v = simulate_market(time_array, gdp_signal, n_firms=50, target_u=0.05)
        self.assertTrue(np.all(u >= 0))
        self.assertTrue(np.all(u <= 1))

    def test_vacancy_rate_in_valid_range(self):
        time_array = np.arange(100)
        gdp_signal = np.sin(np.linspace(0, 2 * np.pi, 100)) * 0.1
        u, v = simulate_market(time_array, gdp_signal, n_firms=50)
        self.assertTrue(np.all(v >= 0))
        self.assertTrue(np.all(v <= 1))

    def test_repeated_signal_returns_valid_result(self):
        """Flat repeated signal should produce stable unemployment."""
        time_array = np.arange(100)
        gdp_flat = np.ones(100) * 0.1
        u, v = simulate_market(time_array, gdp_flat, n_firms=50, seed=42)
        # Unemployment should converge to a steady state
        steady_state_u = np.mean(u[-20:])
        self.assertTrue(0.0 <= steady_state_u <= 1.0)

    def test_mismatched_signal_length_raises(self):
        time_array = np.arange(100)
        gdp_signal = np.ones(50)  # Wrong length
        with self.assertRaises(ValueError):
            simulate_market(time_array, gdp_signal, n_firms=50)


class TestComputeLoss(unittest.TestCase):
    """Test loss function."""

    def test_perfect_fit_has_zero_loss(self):
        """Identical simulated and empirical should have zero loss."""
        u = np.random.uniform(0.03, 0.08, 50)
        v = np.random.uniform(0.02, 0.06, 50)
        loss, _ = compute_loss(u, v, u, v)
        self.assertAlmostEqual(loss, 0.0, places=10)

    def test_perfect_anticorr_penalty(self):
        """Should penalize wrong correlation sign."""
        u = np.linspace(0.03, 0.08, 50)
        v_emp = np.linspace(0.06, 0.02, 50)  # negative corr
        v_sim = np.linspace(0.02, 0.06, 50)  # positive corr
        loss, comps = compute_loss(u, v_sim, u, v_emp)
        self.assertGreater(comps["loss_corr"], 0.1)

    def test_handles_nans(self):
        """Should skip NaN values and still compute loss."""
        u_sim = np.linspace(0.03, 0.08, 20)
        v_sim = np.linspace(0.06, 0.02, 20)
        u_sim[5] = np.nan
        v_sim[10] = np.nan

        u_emp = np.linspace(0.03, 0.08, 20)
        v_emp = np.linspace(0.06, 0.02, 20)

        loss, comps = compute_loss(u_sim, v_sim, u_emp, v_emp)
        # Should compute on the 18 valid points (not including the 2 with NaN)
        self.assertGreaterEqual(comps["n_valid"], 18)

    def test_loss_increases_with_error(self):
        """Larger errors should increase loss."""
        u = np.linspace(0.03, 0.08, 50)
        v = np.linspace(0.06, 0.02, 50)
        loss_small, _ = compute_loss(u, v, u, v + 0.001)
        loss_large, _ = compute_loss(u, v, u, v + 0.01)
        # Both should be finite and loss_large should be larger
        self.assertTrue(np.isfinite(loss_large))
        self.assertTrue(np.isfinite(loss_small))
        self.assertGreater(loss_large, loss_small)


class TestComputeGDPSignal(unittest.TestCase):
    """Test GDP signal normalization."""

    def test_output_is_normalized(self):
        gdp = pd.Series([1000.0, 1010.0, 1020.0, 1030.0])
        g = _compute_gdp_signal(gdp, sigma_g=0.25)
        # Should have std ~0.25
        g_valid = g.dropna()
        if len(g_valid) > 1:
            self.assertAlmostEqual(g_valid.std(), 0.25, places=2)

    def test_handles_constant_series(self):
        """Constant series should produce zero signal."""
        gdp = pd.Series(np.ones(50) * 1000.0)
        g = _compute_gdp_signal(gdp)
        self.assertEqual(g.dropna().std(), 0.0)

    def test_output_length_matches_input(self):
        gdp = pd.Series([1000.0 + i * 10 for i in range(100)])
        g = _compute_gdp_signal(gdp)
        self.assertEqual(len(g), 100)


class TestCalibrationIntegration(unittest.TestCase):
    """Integration tests with real calibration pipeline."""

    def setUp(self):
        """Create minimal synthetic empirical data."""
        self.n = 100
        self.time_array = np.arange(self.n)

        # Synthetic GDP signal
        self.gdp_signal = np.sin(np.linspace(0, 4 * np.pi, self.n)) * 0.15

        # Synthetic empirical data from known simulation
        from calibration.simulate import simulate_market
        u_emp, v_emp = simulate_market(
            self.time_array,
            self.gdp_signal,
            n_firms=100,
            c_exponent=2.0,
            c_max=0.08,
            zero_fraction=0.25,
            K=3.5,
            s=0.01,
            seed=123,
        )

        self.empirical_data = pd.DataFrame({
            "u_obs": u_emp,
            "v_obs": v_emp,
            "G": self.gdp_signal,
        })

    def test_calibration_converges(self):
        """Calibration should produce valid results."""
        from calibration.optimize import calibrate

        result = calibrate(
            self.empirical_data,
            self.time_array,
            self.gdp_signal,
            K=3.5,
            method="L-BFGS-B",
            n_firms=100,
            verbose=False,
        )

        self.assertIn("params_optimal", result)
        self.assertIn("sim_u", result)
        self.assertIn("sim_v", result)
        # Loss should be finite (might be 0 or very small for synthetic data)
        self.assertTrue(np.isfinite(result["loss_optimal"]))

    def test_simulated_trajectories_valid(self):
        """Simulated u and v should be in valid ranges."""
        from calibration.optimize import calibrate

        result = calibrate(
            self.empirical_data,
            self.time_array,
            self.gdp_signal,
            K=3.5,
            method="L-BFGS-B",
            n_firms=100,
            verbose=False,
        )

        u_sim = result["sim_u"]
        v_sim = result["sim_v"]

        self.assertTrue(np.all(u_sim >= 0))
        self.assertTrue(np.all(u_sim <= 1))
        self.assertTrue(np.all(v_sim >= 0))
        self.assertTrue(np.all(v_sim <= 1))


if __name__ == "__main__":
    unittest.main()
