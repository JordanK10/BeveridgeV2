"""
PHASE 4: Calibration via optimization.

Finds optimal parameters (s, c_max, c_exponent, zero_fraction) to minimize loss.
firing_threshold and hiring_efficiency are fixed externally.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution


def calibrate(
    empirical_data,
    time_array,
    gdp_signal,
    K=2.31,
    s=0.0043,
    firing_threshold=0.10,
    dt=1.0,
    method="L-BFGS-B",
    n_firms=250,
    verbose=True,
):
    """
    Calibrate market model to empirical u, v trajectories.

    Parameters
    ----------
    empirical_data : pd.DataFrame
        With columns u_obs, v_obs
    time_array : np.ndarray
        Time points
    gdp_signal : np.ndarray
        G(t) signal
    K : float
        Fixed matching constant (default 2.31 → 6-month unemployment half-life)
    s : float
        Initial separation rate (used as x0; also calibrated within bounds)
    firing_threshold : float
        Fixed fractional overshoot α before firing triggers
    dt : float
        Time step
    method : str
        "L-BFGS-B" for local optimization, "differential_evolution" for global
    n_firms : int
        Number of firms
    verbose : bool

    Returns
    -------
    result : dict
        {params_optimal, loss_optimal, loss_components, sim_u, sim_v}
    """
    from .loss import compute_loss_callable, compute_loss
    from .simulate import simulate_market

    loss_fn = compute_loss_callable(
        empirical_data,
        time_array,
        gdp_signal,
        K=K,
        firing_threshold=firing_threshold,
        dt=dt,
    )

    # Calibrated parameters and their bounds
    bounds = {
        "s":            (0.002, 0.02),
        "c_max":        (0.05,  0.40),
        "c_exponent":   (1.0,   6.0),
        "zero_fraction":(0.0,   0.30),
    }

    param_keys = ["s", "c_max", "c_exponent", "zero_fraction"]
    x0 = np.array([s, 0.15, 1.2, 0.0625])
    bounds_list = [bounds[k] for k in param_keys]

    def loss_wrapper(x):
        params = {k: x[i] for i, k in enumerate(param_keys)}
        params["n_firms"] = n_firms
        return loss_fn(params)

    if method == "L-BFGS-B":
        if verbose:
            print("Starting L-BFGS-B optimization...")
            print(f"Free parameters: {param_keys}")
            print(f"Fixed: K={K}, firing_threshold={firing_threshold}")

        result = minimize(
            loss_wrapper,
            x0,
            method="L-BFGS-B",
            bounds=bounds_list,
            options={"maxiter": 200},
        )

        x_opt = result.x if result.success else x0
        if not result.success and verbose:
            print(f"Warning: {result.message}")

    elif method == "differential_evolution":
        if verbose:
            print("Starting differential_evolution optimization...")
            print(f"Free parameters: {param_keys}")
            print(f"Fixed: K={K}, firing_threshold={firing_threshold}")

        result = differential_evolution(
            loss_wrapper,
            bounds_list,
            maxiter=200,
            seed=42,
            workers=1,
            disp=verbose,
        )
        x_opt = result.x

    else:
        raise ValueError(f"Unknown method: {method}")

    params_opt = {k: x_opt[i] for i, k in enumerate(param_keys)}
    params_opt["n_firms"] = n_firms
    loss_opt = loss_wrapper(x_opt)

    if verbose:
        print("\nOptimal parameters:")
        for k in param_keys:
            print(f"  {k}: {params_opt[k]:.4f}")
        print(f"  K={K} (fixed), firing_threshold={firing_threshold} (fixed)")
        print(f"Final loss: {loss_opt:.6f}")

    # Compute best-fit trajectories
    sim_u, sim_v = simulate_market(
        time_array,
        gdp_signal,
        K=K,
        dt=dt,
        s=params_opt["s"],
        c_max=params_opt["c_max"],
        c_exponent=params_opt["c_exponent"],
        zero_fraction=params_opt["zero_fraction"],
        firing_threshold=firing_threshold,
        n_firms=n_firms,
    )

    emp_u = empirical_data["u_obs"].values.astype(float)
    emp_v = empirical_data["v_obs"].values.astype(float)
    _, loss_components = compute_loss(sim_u, sim_v, emp_u, emp_v)

    return {
        "params_optimal": params_opt,
        "loss_optimal": loss_opt,
        "loss_components": loss_components,
        "sim_u": sim_u,
        "sim_v": sim_v,
    }
