"""Cross-correlation between unemployment and vacancy rate fluctuations (lag experiment)."""

import os

import matplotlib.pyplot as plt
import numpy as np

from time_grid import DT


def crosscorr_pearson(x, y, max_lag, min_segment=10):
    """
    Pearson correlation between x[t] and y[t + ell] for each integer lag ell.

    For ``corr(u_t, v_{t+ell})``: ell > 0 pairs u with *later* v (unemployment
    leads vacancies if |rho| peaks at ell > 0); ell < 0 pairs u with *earlier* v.

    Returns:
        lags: int array from -max_lag to max_lag
        rho: Pearson r per lag (nan if degenerate)
        n_pairs: sample size per lag
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    T = len(x)
    if len(y) != T:
        raise ValueError("x and y must have the same length")

    max_lag = int(max_lag)
    max_lag = min(max_lag, T - min_segment)
    max_lag = max(max_lag, 0)

    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    rho = np.full(len(lags), np.nan, dtype=float)
    n_pairs = np.zeros(len(lags), dtype=int)

    for i, ell in enumerate(lags):
        if ell >= 0:
            n = T - ell
            if n < min_segment:
                continue
            xs = x[:n]
            ys = y[ell : ell + n]
        else:
            k = -ell
            n = T - k
            if n < min_segment:
                continue
            xs = x[k : k + n]
            ys = y[:n]

        n_pairs[i] = n
        if np.std(xs) < 1e-15 or np.std(ys) < 1e-15:
            continue
        c = np.corrcoef(xs, ys)
        rho[i] = c[0, 1]

    return lags, rho, n_pairs


def plot_uv_cross_correlation(
    lags,
    rho,
    dt,
    output_path,
    title_suffix="",
    transform_label="",
    peak_label=True,
    xlim_lag_steps=None,
    ccf_kind="rates_demeaned",
):
    """
    Plot lag (time units) vs Pearson correlation; mark argmax |rho|.

    ``ccf_kind``:
        ``rates_demeaned`` — figure text for corr(u_t, v_{t+ell}) on demeaned rates.
        ``delta_levels`` — corr(Delta U_t, Delta V_{t+ell}) on aggregate counts.
    """
    lag_time = lags.astype(float) * dt

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(lag_time, rho, color="C0", linewidth=1.2)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)

    finite = np.isfinite(rho)
    if peak_label and np.any(finite):
        idx = int(np.nanargmax(np.abs(rho[finite])))
        # map back to index in full array
        fi = np.flatnonzero(finite)
        i_peak = fi[idx]
        ax.axvline(lag_time[i_peak], color="C3", linestyle="--", linewidth=1.0, alpha=0.85)
        ax.scatter(
            [lag_time[i_peak]],
            [rho[i_peak]],
            color="C3",
            s=40,
            zorder=5,
            label=f"max |ρ| at lag={lags[i_peak]} ({lag_time[i_peak]:.2f} time)",
        )
        ax.legend(loc="upper right", fontsize=8)

    if ccf_kind == "delta_levels":
        ax.set_xlabel(
            r"Lag $\ell$ (time units): $\rho(\ell)=\mathrm{corr}(\Delta U_t,\Delta V_{t+\ell})$; "
            r"$\Delta U_t=U_{t+1}-U_t$, $\Delta V_{t+\ell}=V_{t+\ell+1}-V_{t+\ell}$; "
            r"$\ell>0$ → later $\Delta V$; $\ell<0$ → earlier $\Delta V$"
        )
        ax.set_ylabel(r"Pearson $\rho$")
        ttl = r"$\Delta U$–$\Delta V$ cross-correlation: $\rho(\ell)=\mathrm{corr}(\Delta U_t, \Delta V_{t+\ell})$"
    else:
        ax.set_xlabel(
            r"Lag $\ell$ (time units): $\rho(\ell)=\mathrm{corr}(u_t,v_{t+\ell})$; "
            r"$\ell>0$ → later $v$; $\ell<0$ → earlier $v$"
        )
        ax.set_ylabel(r"Pearson $\rho$")
        ttl = "U–V cross-correlation: " + r"$\rho(\ell)=\mathrm{corr}(u_t, v_{t+\ell})$"
    if title_suffix:
        ttl += f" — {title_suffix}"
    ax.set_title(ttl, fontsize=11)
    if transform_label:
        ax.text(
            0.02,
            0.02,
            transform_label,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
        )
    ax.grid(True, alpha=0.3)

    if xlim_lag_steps is not None:
        lim = float(xlim_lag_steps) * dt
        ax.set_xlim(-lim, lim)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _rates_from_levels(aggregate_unemployment, aggregate_vacancies, population, burn_in):
    """Match diagnostics.plot_fluctuation_analysis rate definitions after burn-in."""
    u = np.asarray(aggregate_unemployment, dtype=float)[burn_in:]
    vlev = np.asarray(aggregate_vacancies, dtype=float)[burn_in:]
    e = population - u
    ur = u / population
    denom = vlev + e
    vr = np.divide(vlev, denom, out=np.zeros_like(vlev, dtype=float), where=denom > 0)
    return ur, vr


def plot_uv_lag_experiment_from_series(
    aggregate_unemployment,
    aggregate_vacancies,
    population,
    burn_in,
    max_lag_steps=150,
    transform="demean_rates",
    output_dir=None,
    economy_name="single_firm",
    dt=None,
):
    """
    Build u,v rates, apply transform, compute CCF, write ``uv_crosscorr_{economy_name}.pdf``.

    transform:
        ``demean_rates`` — subtract mean of u and v on the window.
        ``diff_rates`` — first differences of u and v.
    """
    if dt is None:
        dt = DT
    if output_dir is None:
        from plotting.paths import OUTPUT_DIR

        output_dir = OUTPUT_DIR

    ur, vr = _rates_from_levels(aggregate_unemployment, aggregate_vacancies, population, burn_in)
    T = len(ur)

    if transform == "demean_rates":
        u = ur - np.mean(ur)
        v = vr - np.mean(vr)
        transform_label = r"Series: $u=U/L$, $v=V/(V+E)$; demeaned"
    elif transform == "diff_rates":
        u = np.diff(ur)
        v = np.diff(vr)
        transform_label = r"Series: $\Delta u_t$, $\Delta v_t$ (rates)"
    else:
        raise ValueError(f"Unknown transform {transform!r}; use demean_rates or diff_rates")

    min_seg = 10
    L = min(int(max_lag_steps), max(0, len(u) - min_seg))

    lags, rho, n_pairs = crosscorr_pearson(u, v, L)

    out_path = os.path.join(output_dir, f"uv_crosscorr_{economy_name}.pdf")
    plot_uv_cross_correlation(
        lags,
        rho,
        dt,
        out_path,
        title_suffix=economy_name.replace("_", " "),
        transform_label=transform_label,
        peak_label=True,
        xlim_lag_steps=int(max_lag_steps),
    )
    return {
        "lags": lags,
        "rho": rho,
        "n_pairs": n_pairs,
        "output_path": out_path,
        "max_lag_used": L,
        "transform": transform,
    }


def plot_uv_delta_levels_crosscorr_from_series(
    aggregate_unemployment,
    aggregate_vacancies,
    burn_in,
    max_lag_steps=150,
    output_dir=None,
    economy_name="single_firm",
    dt=None,
):
    """
    Cross-correlation of **first differences** of aggregate levels:
    ``corr(ΔU_t, ΔV_{t+ell})`` with ``ΔU_t = U_{t+1}-U_t``, ``ΔV_s = V_{s+1}-V_s``.

    Uses post–burn-in slices of ``U`` and ``V`` (counts), then ``np.diff``.
    Writes ``uv_crosscorr_deltas_{economy_name}.pdf``.
    """
    if dt is None:
        dt = DT
    if output_dir is None:
        from plotting.paths import OUTPUT_DIR

        output_dir = OUTPUT_DIR

    U = np.asarray(aggregate_unemployment, dtype=float)[burn_in:]
    V = np.asarray(aggregate_vacancies, dtype=float)[burn_in:]
    dU = np.diff(U)
    dV = np.diff(V)
    transform_label = r"Series: $\Delta U_t = U_{t+1}-U_t$, $\Delta V_t = V_{t+1}-V_t$ (aggregate counts)"

    min_seg = 10
    L = min(int(max_lag_steps), max(0, len(dU) - min_seg))

    lags, rho, n_pairs = crosscorr_pearson(dU, dV, L)

    out_path = os.path.join(output_dir, f"uv_crosscorr_deltas_{economy_name}.pdf")
    plot_uv_cross_correlation(
        lags,
        rho,
        dt,
        out_path,
        title_suffix=economy_name.replace("_", " "),
        transform_label=transform_label,
        peak_label=True,
        xlim_lag_steps=int(max_lag_steps),
        ccf_kind="delta_levels",
    )
    return {
        "lags": lags,
        "rho": rho,
        "n_pairs": n_pairs,
        "output_path": out_path,
        "max_lag_used": L,
        "kind": "delta_levels",
    }
