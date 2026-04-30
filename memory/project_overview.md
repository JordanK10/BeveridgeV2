---
name: BeveridgeV2 Project Overview
description: High-level structure of the BeveridgeV2 agent-based macro labor market model and its calibration pipeline
type: project
---

Agent-based model of the US labor market that reproduces the Beveridge curve (u-v tradeoff).
Firms post vacancies and hire based on a matching function; the calibration fits simulated aggregate u(t), v(t) to empirical JOLTS/BLS data driven by a GDP signal G(t).

**Why:** Research project (likely academic/Antoine collaboration) exploring how firm-level heterogeneity generates Beveridge curve dynamics.

**How to apply:** Calibration improvements should preserve the pipeline structure and keep simulate_market as the inner loop.

## Key modules

### src/beveridge/
Full ABM with TIME grid dependency — the production model.
- firm.py: Firm class (employment, vacancies, efficiency, firing)
- economy.py: aggregate market clearing
- config.py, signals.py, experiments.py, main.py

### src/calibration/
Lightweight standalone simulator + fitting — works without global TIME.
- empirical.py: loads JOLTS + BLS + FRED, aligns monthly u_obs/v_obs/G
- simulate.py: SimpleCalibFirm + simulate_market() — core inner loop
- loss.py: compute_loss() — level MSE + correlation + Beveridge shape
- optimize.py: calibrate() — L-BFGS-B or differential_evolution
- fit.py: multi-basin OLS pipeline — LHS scan → basin ID → Nelder-Mead
- fit_smm.py: SMM alternative — matches moments (mean_u, std_u, corr_uv, skew_du, etc.)
- parameter_scan.py: 2D grid scans (c_max×c_exp, idio_std×firing_threshold, K×s) with NPZ cache
- plotting.py, diagnostic_plot.py, fit_plots.py, fit_smm_plots.py: visualization
- main.py: orchestrates full pipeline

### src/data_fetch/
- fred.py, bls.py, loader.py: pull macro data, cache locally
- GDP signal: monthly log-growth with AR(2) noise (gdp_log_growth_linear)

### src/generators/
- ar2generator.py, dummy_gdp_gen.py: synthetic GDP signal generators

## Parameters calibrated (fit.py)
PARAM_NAMES = [K, s, c_exponent, zero_fraction, firing_threshold, idio_std]

Bounds:
  K:                (1.00, 6.00)   — matching rate constant (m = K*u)
  s:                (0.02, 0.065)  — separation rate
  c_exponent:       (0.25, 1.00)   — power-law exponent of C distribution
  zero_fraction:    (0.00, 0.50)   — fraction of firms with C=0 (insensitive to GDP)
  firing_threshold: (0.00, 0.50)   — α: fire only when (e-ê)/ê > α
  idio_std:         (0.001, 0.06)  — std of AR(1) idiosyncratic firm shocks

Baseline defaults (diagnostic_plot.py): K=2.31, s=0.0043, c_max=0.40, c_exponent=1.2, zero_fraction=0.0625, firing_threshold=0.10

## Fitting strategies
1. OLS (fit.py): LHS scan → find_basins (greedy max-separation) → Nelder-Mead, 6 objectives
2. SMM (fit_smm.py): same structure but loss = weighted squared distance of 11 moments
3. parameter_scan.py: diagnostic 2D grids, results cached as NPZ

## Data notes
- Empirical data hard-capped at Oct 2025 (last real GDP data available)
- COVID shock (Mar-Aug 2020) down-weighted to 0.1 in some runs
- GDP lag configurable (gdp_lag_months, Antoine recommends 18mo lag)
