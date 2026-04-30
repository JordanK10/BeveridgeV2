"""
PHASE 3: Loss function for calibration.

Compares simulated vs empirical u and v trajectories.
"""

import numpy as np


def _pad_or_trim(sim, emp, axis=0):
    """Align two arrays by padding shorter with NaN or trimming longer."""
    n_sim, n_emp = len(sim), len(emp)
    if n_sim == n_emp:
        return sim, emp
    if n_sim > n_emp:
        return sim[:n_emp], emp
    else:
        return sim, emp[:n_sim]


def compute_loss(
    sim_u,
    sim_v,
    emp_u,
    emp_v,
    weights=(1.0, 1.0, 0.5),
):
    """
    Compute weighted loss between simulated and empirical trajectories.

    Parameters
    ----------
    sim_u : np.ndarray
        Simulated unemployment rate
    sim_v : np.ndarray
        Simulated vacancy rate
    emp_u : np.ndarray
        Empirical unemployment rate
    emp_v : np.ndarray
        Empirical vacancy rate
    weights : tuple
        (w_level, w_corr, w_bev) for level MSE, correlation, Beveridge

    Returns
    -------
    loss : float
        Total weighted loss
    components : dict
        Individual loss terms for diagnostics
    """
    w_level, w_corr, w_bev = weights

    # Align lengths
    sim_u, emp_u = _pad_or_trim(sim_u, emp_u)
    sim_v, emp_v = _pad_or_trim(sim_v, emp_v)

    # Remove NaNs
    mask = ~(np.isnan(sim_u) | np.isnan(sim_v) | np.isnan(emp_u) | np.isnan(emp_v))
    sim_u, sim_v = sim_u[mask], sim_v[mask]
    emp_u, emp_v = emp_u[mask], emp_v[mask]

    if len(sim_u) < 10:
        return np.inf, {"n_valid": 0, "reason": "too few valid points"}

    # 1. Level MSE
    mse_u = np.mean((sim_u - emp_u) ** 2)
    mse_v = np.mean((sim_v - emp_v) ** 2)
    loss_level = mse_u + mse_v

    # 2. Correlation matching: target negative u-v correlation (Beveridge curve)
    corr_sim = np.corrcoef(sim_u, sim_v)[0, 1]
    corr_emp = np.corrcoef(emp_u, emp_v)[0, 1]
    if np.isnan(corr_sim) or np.isnan(corr_emp):
        loss_corr = 0.0
    else:
        loss_corr = (corr_sim - corr_emp) ** 2

    # 3. Beveridge curve shape: compare u vs v scatterplot spread
    # Use std of deviations from the diagonal fit
    fit_sim = np.polyfit(emp_u, sim_v, 1)  # Fit to empirical u
    pred_v_sim = np.polyval(fit_sim, emp_u)
    residual_sim = np.std(sim_v - pred_v_sim)

    fit_emp = np.polyfit(emp_u, emp_v, 1)
    pred_v_emp = np.polyval(fit_emp, emp_u)
    residual_emp = np.std(emp_v - pred_v_emp)

    loss_bev = (residual_sim - residual_emp) ** 2 if residual_emp > 1e-6 else 0.0

    # Total
    total_loss = w_level * loss_level + w_corr * loss_corr + w_bev * loss_bev

    components = {
        "mse_u": mse_u,
        "mse_v": mse_v,
        "loss_level": loss_level,
        "loss_corr": loss_corr,
        "loss_bev": loss_bev,
        "corr_sim": corr_sim,
        "corr_emp": corr_emp,
        "n_valid": len(sim_u),
    }

    return total_loss, components


def compute_loss_callable(
    empirical_data,
    time_array,
    gdp_signal,
    K=2.31,
    firing_threshold=0.10,
    dt=1.0,
    weights=(1.0, 1.0, 0.5),
):
    """
    Create a loss function for optimization.

    Parameters
    ----------
    empirical_data : pd.DataFrame
        Must have columns: u_obs, v_obs
    time_array : np.ndarray
    gdp_signal : np.ndarray
    K : float
        Fixed matching constant
    dt : float
        Time step
    weights : tuple

    Returns
    -------
    loss_fn : callable
        loss_fn(params_dict) -> float
    """
    from .simulate import simulate_market

    emp_u = empirical_data["u_obs"].values
    emp_v = empirical_data["v_obs"].values

    def loss_fn(params):
        """
        params: dict with keys {s, c_max, c_exponent, zero_fraction,
                                firing_threshold, hiring_efficiency, n_firms}
        """
        try:
            s             = params.get("s", 0.0043)
            c_max         = params.get("c_max", 0.15)
            c_exponent    = params.get("c_exponent", 1.2)
            zero_fraction = params.get("zero_fraction", 0.0625)
            n_firms       = params.get("n_firms", 250)

            sim_u, sim_v = simulate_market(
                time_array,
                gdp_signal,
                n_firms=n_firms,
                c_exponent=c_exponent,
                c_max=c_max,
                zero_fraction=zero_fraction,
                K=K,
                s=s,
                dt=dt,
                firing_threshold=firing_threshold,
            )

            loss, _ = compute_loss(sim_u, sim_v, emp_u, emp_v, weights=weights)
            return loss
        except Exception as e:
            print(f"Loss computation failed: {e}")
            return 1e6

    return loss_fn
