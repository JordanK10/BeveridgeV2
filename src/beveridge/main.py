"""CLI entry: orchestrates default experiment batch."""

import os

import numpy as np

from . import config
from .c_distribution_sweep import format_c_sweep_subtitle, run_c_distribution_sweep
from .experiments import (
    run_gdp_shock_response_experiment,
    run_multi_firm_simulation,
    run_single_timeseries,
    run_uv_crosscorr_experiment,
)
from .signals import GDP
from plotting.c_sweep_figs import plot_c_sweep_grid, plot_c_sweep_grid_multi
from plotting.diagnostics import plot_matching_time_vs_K, plot_matching_time_vs_urate

# (GDP column name, subdirectory suffix under output/) for multi-firm runs
MULTI_FIRM_SIGNALS = (
    ("gdp_sine", "sine"),
    ("gdp_ar2", "gdp_ar2"),
)

def main():
    single_ts_output_dir = os.path.join(config.OUTPUT_DIR, "single_timeseries_plots")

    run_single_timeseries(output_dir=single_ts_output_dir)

    for signal_col, dir_slug in MULTI_FIRM_SIGNALS:
        if signal_col not in GDP.columns:
            print(f"Skipping multi_firm_plots_{dir_slug}: column {signal_col!r} not in GDP.")
            continue
        sig = GDP[signal_col]
        if not np.isfinite(sig).all() or float(np.nanvar(np.asarray(sig, dtype=float))) < 1e-20:
            print(
                f"Skipping multi_firm_plots_{dir_slug}: signal {signal_col!r} is empty, "
                "non-finite, or effectively constant (e.g. missing ar2_signal.pkl)."
            )
            continue

        multi_firm_output_dir = os.path.join(config.OUTPUT_DIR, f"multi_firm_plots_{dir_slug}")
        economy_tag = f"{config.MULTI_FIRM_COUNT}_firms_power_law_C_{dir_slug}"

        run_multi_firm_simulation(
            num_firms=config.MULTI_FIRM_COUNT,
            c_distribution_method="power_law",
            c_params={
                "c_max": 0.08,
                "exponent": 2.0,
                "zero_fraction": 0.25,
                "seed": 42,
                "power_law_flip": True,
            },
            economy_name=economy_tag,
            output_dir=multi_firm_output_dir,
            signal_name=signal_col,
        )

        plot_matching_time_vs_K(output_dir=multi_firm_output_dir)
        plot_matching_time_vs_urate(output_dir=multi_firm_output_dir)


def run_c_sweep_cli(power_law_flip=True):
    """
    Exponent sweep over default power-law C.

    When both ``gdp_ar2`` and ``gdp_sine`` are available, observables are overlaid
    in one grid PDF (``sweep_exponent_combined.pdf``): AR(2) solid, sine dashed.
    If only one signal is available, writes ``sweep_exponent_<signal>.pdf`` as before.

    **Default** uses ``C = c_max * (1 - U^{1/\\alpha})`` (``power_law_flip=True``):
    for ``\\alpha > 1``, positive ``C`` is denser **near 0** and thinner **near**
    ``c_max``.  Pass ``power_law_flip=False`` or run ``c_sweep --no-flip`` for the
    legacy map ``C = c_max * U^{1/\\alpha}`` (pile-up near ``c_max``).
    """
    out_root = os.path.join(config.OUTPUT_DIR, "c_distribution_sweep")
    exponent_grid = np.linspace(1.2, 6.0, 10)
    c_params_base = {
        "c_max": 0.08,
        "exponent": 2.0,
        "zero_fraction": 0.25,
        "seed": 42,
        "power_law_flip": bool(power_law_flip),
    }

    legacy_suffix = "" if c_params_base.get("power_law_flip") else "_legacy_cdf_u"
    sweep_rows_by_signal = {}

    for signal_col, _dir_slug in MULTI_FIRM_SIGNALS:
        if signal_col not in GDP.columns:
            print(f"Skipping c_sweep for {signal_col!r}: column not in GDP.")
            continue
        sig = GDP[signal_col]
        if not np.isfinite(sig).all() or float(np.nanvar(np.asarray(sig, dtype=float))) < 1e-20:
            print(
                f"Skipping c_sweep for {signal_col!r}: empty, non-finite, or effectively constant."
            )
            continue
        print(
            f"C-distribution sweep (exponent, "
            f"{'C=c_max(1-U^(1/α))' if c_params_base.get('power_law_flip') else 'C=c_max·U^(1/α)'}): "
            f"{signal_col} …"
        )
        rows = run_c_distribution_sweep(
            "exponent",
            exponent_grid,
            c_params_base=c_params_base,
            signal_name=signal_col,
            output_dir=out_root,
            write_sweep_grid_pdf=False,
        )
        sweep_rows_by_signal[signal_col] = rows

    num_firms = config.MULTI_FIRM_COUNT
    if not sweep_rows_by_signal:
        return

    if len(sweep_rows_by_signal) == 1:
        only_col = next(iter(sweep_rows_by_signal))
        title_extra = format_c_sweep_subtitle(
            num_firms,
            "exponent",
            c_params_base,
            "power_law",
            f"signal={only_col}",
        )
        grid_path = os.path.join(
            out_root, f"sweep_exponent_{only_col}{legacy_suffix}.pdf"
        )
        plot_c_sweep_grid(
            sweep_rows_by_signal[only_col],
            "exponent",
            title_extra,
            grid_path,
        )
        print(f"Saved C-distribution sweep figure to {grid_path}")
    else:
        series_specs = []
        if "gdp_ar2" in sweep_rows_by_signal:
            series_specs.append(
                {
                    "label": "AR(2)",
                    "rows": sweep_rows_by_signal["gdp_ar2"],
                    "linestyle": "-",
                }
            )
        if "gdp_sine" in sweep_rows_by_signal:
            series_specs.append(
                {
                    "label": "sine",
                    "rows": sweep_rows_by_signal["gdp_sine"],
                    "linestyle": "--",
                }
            )
        title_extra = format_c_sweep_subtitle(
            num_firms,
            "exponent",
            c_params_base,
            "power_law",
            "signals gdp_ar2 (solid) + gdp_sine (dashed)",
        )
        grid_path = os.path.join(
            out_root, f"sweep_exponent_combined{legacy_suffix}.pdf"
        )
        plot_c_sweep_grid_multi(
            series_specs, "exponent", title_extra, grid_path
        )
        print(f"Saved combined C-distribution sweep figure to {grid_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "gdp_shock":
        run_gdp_shock_response_experiment()
    elif len(sys.argv) > 1 and sys.argv[1] == "uv_ccf":
        run_uv_crosscorr_experiment()
    elif len(sys.argv) > 1 and sys.argv[1] == "c_sweep":
        legacy = "--no-flip" in sys.argv or "--legacy-c" in sys.argv
        run_c_sweep_cli(power_law_flip=not legacy)
    else:
        main()
