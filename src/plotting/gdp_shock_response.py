"""Figures for GDP step-shock relaxation (unemployment / vacancy rates) and diagnostics."""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_gdp_shock_response_figure(
    g_values,
    tau_axis,
    U_by_g,
    V_by_g,
    ed_gap_by_g,
    population,
    output_path=None,
    title=r"GDP shock response: $u=U/L$, $v/(v+e)$, and $(D-E)/L$",
    plot_window_steps=15,
    tau_centered_shock=None,
    de_lag_window_by_g=None,
    v_lag_window_by_g=None,
    u_lag_window_by_g=None,
):
    """
    Two rows × three columns:

    **Row 1** (shared ``tau``): unemployment rate ``u=U/L``, vacancy rate ``v/(v+e)``
    (:func:`plotting.rates.aggregate_vacancy_rate`), and gap ``(E-D)/L``.

    **Row 2:** **Impulse-style responses** vs ``tau`` (same time axis as row 1): ``Delta u``
    and ``Delta v`` from ``tau=0`` in the first panel. **Middle panel:** if
    ``tau_centered_shock``, ``de_lag_window_by_g``, and ``v_lag_window_by_g`` are provided,
    Pearson **r across post-shock levels** ``g`` between (i) ``(D-E)/L`` at the shock
    (``tau=0``) and (ii) vacancy rate at each lag ``tau in [-L, L]`` (UV-style lead/lag);
    plus a second curve with anchor ``Delta(D-E)/L`` from ``tau=-1`` to ``tau=0`` vs
    ``v(tau)``. Otherwise (legacy) ``Delta(D-E)/L`` and ``Delta v`` vs post-shock ``tau``.
    **Third panel:** Beveridge-style trace of vacancy rate vs unemployment rate over the
    post-shock window.

    ``plot_window_steps``: if an integer, only the first ``N`` post-shock samples (same
    length as the prefix of ``tau_axis``) are plotted; if ``None``, the full series is used.

    ``U_by_g`` / ``V_by_g`` must already be rate series (not level counts).
    ``ed_gap_by_g`` holds ``(D-E)/L`` (demand minus employment, scaled by labor force).

    Legend for post-shock ``g`` sits outside the Beveridge (bottom-right) panel only, with a
    large font. ``population`` is labor force ``L``.
    """

    L = float(population)
    if L <= 0:
        raise ValueError("population (labor force L) must be positive")

    fig, axs = plt.subplots(2, 3, figsize=(17.5, 9.0), constrained_layout=False)
    ax_u, ax_v, ax_ed = axs[0, 0], axs[0, 1], axs[0, 2]
    ax_cuv, ax_cvg, ax_bev = axs[1, 0], axs[1, 1], axs[1, 2]

    for ax in (ax_v, ax_ed):
        ax.sharex(ax_u)

    use_lag_window = (
        tau_centered_shock is not None
        and de_lag_window_by_g is not None
        and v_lag_window_by_g is not None
        and u_lag_window_by_g is not None
        and len(de_lag_window_by_g) == len(g_values)
        and len(v_lag_window_by_g) == len(g_values)
        and len(u_lag_window_by_g) == len(g_values)
    )
    if not use_lag_window:
        ax_cuv.sharex(ax_u)
        ax_cvg.sharex(ax_u)

    fig.suptitle(title, fontsize=14, y=0.97)

    cmap = plt.get_cmap("coolwarm")
    g_arr = np.asarray(g_values, dtype=float)
    gmin, gmax = float(np.min(g_arr)), float(np.max(g_arr))
    span = gmax - gmin if gmax > gmin else 1.0
    norm = plt.Normalize(vmin=gmin - 0.05 * span, vmax=gmax + 0.05 * span)

    tau_axis = np.asarray(tau_axis, dtype=float)
    T_full = int(tau_axis.size)
    if T_full < 1:
        raise ValueError("tau_axis must be non-empty")
    if plot_window_steps is None:
        n_plot = T_full
    else:
        n_plot = min(int(plot_window_steps), T_full)
        if n_plot < 1:
            raise ValueError("plot_window_steps must be >= 1 when set")
    tau_plot = tau_axis[:n_plot]

    for j, g in enumerate(g_values):
        color = cmap(norm(g))
        u_full = np.asarray(U_by_g[j], dtype=float).ravel()
        v_full = np.asarray(V_by_g[j], dtype=float).ravel()
        edr_full = np.asarray(ed_gap_by_g[j], dtype=float).ravel()
        if u_full.size < n_plot or v_full.size < n_plot or edr_full.size < n_plot:
            raise ValueError(
                f"Series length for g index {j} must be >= plot window ({n_plot}); "
                f"got len(u)={u_full.size}, len(v)={v_full.size}, len(gap)={edr_full.size}"
            )
        u = u_full[:n_plot]
        v = v_full[:n_plot]
        edr = edr_full[:n_plot]
        ax_u.plot(tau_plot, u, color=color, linestyle="-", linewidth=1.2, alpha=0.9)
        ax_v.plot(tau_plot, v, color=color, linestyle="-", linewidth=1.2, alpha=0.9)
        ax_ed.plot(tau_plot, edr, color=color, linewidth=1.2, alpha=0.9)
        ax_bev.plot(u, v, color=color, linewidth=1.2, alpha=0.9)

    ax_u.set_ylabel(r"Unemployment rate $u=U/L$", color="C0")
    ax_u.tick_params(axis="y", labelcolor="C0")
    ax_u.set_title(r"Unemployment rate $u$")
    ax_u.grid(True, alpha=0.3)

    ax_v.set_ylabel(r"Vacancy rate $v/(v+e)$", color="C1")
    ax_v.tick_params(axis="y", labelcolor="C1")
    ax_v.set_title(r"Vacancy rate $v/(v+e)$")
    ax_v.grid(True, alpha=0.3)

    ax_ed.set_title(r"Demand gap $(\hat{d}-e)/L = (D-E)/L$")
    ax_ed.set_ylabel(r"$(D - E) / L$")
    ax_ed.axhline(0.0, color="0.45", linestyle=":", linewidth=0.9)
    ax_ed.grid(True, alpha=0.3)

    line_legend = [
        Line2D([0], [0], color="0.25", linestyle="-",  linewidth=1.4, label=r"$u=U/L$"),
        Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.4, label=r"$v/(v+e)$"),
        Line2D([0], [0], color="0.25", linestyle=":",  linewidth=1.4, label=r"$(D-E)/L$"),
    ]

    def _plot_shock_panel(ax, indices, title):
        if use_lag_window:
            tau_cs = np.asarray(tau_centered_shock, dtype=float)
            for j in indices:
                color = cmap(norm(float(g_values[j])))
                u_c  = np.asarray(u_lag_window_by_g[j],  dtype=float)
                v_c  = np.asarray(v_lag_window_by_g[j],  dtype=float)
                de_c = np.asarray(de_lag_window_by_g[j], dtype=float)
                ax.plot(tau_cs, u_c,  color=color, linestyle="-",  linewidth=1.2, alpha=0.9)
                ax.plot(tau_cs, v_c,  color=color, linestyle="--", linewidth=1.2, alpha=0.9)
                ax.plot(tau_cs, de_c, color=color, linestyle=":",  linewidth=1.2, alpha=0.9)
            ax.set_xlim(tau_cs[0], tau_cs[-1])
        ax.set_title(title)
        ax.set_ylabel("Rate")
        ax.set_xlabel(r"$\tau$ (time; $0$ = shock)")
        ax.axvline(0.0, color="0.45", linestyle=":", linewidth=0.8)
        ax.axhline(0.0, color="0.45", linestyle=":", linewidth=0.8)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=line_legend, loc="upper right", fontsize=8, framealpha=0.92)

    neg_idx = [j for j, g in enumerate(g_values) if float(g) < 0]
    pos_idx = [j for j, g in enumerate(g_values) if float(g) > 0]

    _plot_shock_panel(ax_cuv, neg_idx, r"Negative shocks: $u$, $v/(v+e)$, $(D-E)/L$")
    _plot_shock_panel(ax_cvg, pos_idx, r"Positive shocks: $u$, $v/(v+e)$, $(D-E)/L$")

    ax_bev.set_title(
        r"Beveridge trace: $v/(v+e)$ vs $u$ ($\tau$ along path"
        + (rf", first {n_plot} steps)" if n_plot < T_full else r")")
    )
    ax_bev.set_xlabel(r"Unemployment rate $u$")
    ax_bev.set_ylabel(r"Vacancy rate $v/(v+e)$")
    ax_bev.grid(True, alpha=0.3)

    h_leg = [
        Line2D([0], [0], color=cmap(norm(g)), linestyle="-", linewidth=2.5, label=f"{g:.2f}")
        for g in g_values
    ]
    leg_bev = ax_bev.legend(
        handles=h_leg,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        bbox_transform=ax_bev.transAxes,
        borderaxespad=0.0,
        fontsize=12,
        title_fontsize=13,
        ncol=1,
        frameon=True,
        title=r"Post-shock $g$",
    )

    ax_v.set_xlabel(r"Time since shock ($\tau$)", fontsize=11)

    fig.subplots_adjust(
        left=0.06, right=0.80, top=0.92, bottom=0.07, wspace=0.14, hspace=0.18
    )

    extra = (leg_bev,)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(
            output_path,
            dpi=150,
            bbox_extra_artists=extra,
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()

    return fig


def default_shock_g_grid():
    """Post-shock constant G levels: 0.4 … 0.1 and -0.1 … -1.0."""
    pos = [0.4, 0.3, 0.2, 0.1]
    neg = [-.1, -.3, -.5,-1]
    return np.sort(pos + neg)
