"""CLI entry: orchestrates default experiment batch."""

import os

import numpy as np

from . import config
from .experiments import (
    run_gdp_shock_response_experiment,
    run_multi_firm_simulation,
    run_single_timeseries,
    run_uv_crosscorr_experiment,
)
from .signals import GDP
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
        economy_tag = f"50_firms_power_law_C_{dir_slug}"

        run_multi_firm_simulation(
            num_firms=50,
            c_distribution_method="power_law",
            c_params={"c_max": 0.08, "exponent": 2.0, "zero_fraction": 0.25, "seed": 42},
            economy_name=economy_tag,
            output_dir=multi_firm_output_dir,
            signal_name=signal_col,
        )

        plot_matching_time_vs_K(output_dir=multi_firm_output_dir)
        plot_matching_time_vs_urate(output_dir=multi_firm_output_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "gdp_shock":
        run_gdp_shock_response_experiment()
    elif len(sys.argv) > 1 and sys.argv[1] == "uv_ccf":
        run_uv_crosscorr_experiment()
    else:
        main()
