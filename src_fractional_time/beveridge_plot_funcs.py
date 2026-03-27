"""
Backward-compatible re-exports. Prefer:

- plotting.rates — compute_rates, vacancy_and_unemployment_rates
- plotting.beveridge_figs — Beveridge curve, aggregates, firm employment/vacancy panels
- plotting.diagnostics — fluctuation analysis, growth heatmaps, matching-time figures
"""

from plotting.paths import OUTPUT_DIR
from plotting.rates import compute_rates, vacancy_and_unemployment_rates
from plotting.beveridge_figs import (
    plot_aggregate_overview,
    plot_beveridge_comparison,
    plot_beveridge_trajectory,
    plot_efficiency_time_series,
    plot_employment_growth_rate,
    plot_multi_employment,
    plot_response_to_demand,
    plot_vacancy_time_series,
)
from plotting.diagnostics import (
    plot_efficiency_heatmap,
    plot_fluctuation_analysis,
    plot_growth_rate_analysis,
    plot_growth_rate_heatmaps,
    plot_growth_rates_time_series,
    plot_matching_time_vs_K,
    plot_matching_time_vs_urate,
)

__all__ = [
    "OUTPUT_DIR",
    "compute_rates",
    "vacancy_and_unemployment_rates",
    "plot_aggregate_overview",
    "plot_beveridge_comparison",
    "plot_beveridge_trajectory",
    "plot_efficiency_heatmap",
    "plot_efficiency_time_series",
    "plot_employment_growth_rate",
    "plot_fluctuation_analysis",
    "plot_growth_rate_analysis",
    "plot_growth_rate_heatmaps",
    "plot_growth_rates_time_series",
    "plot_matching_time_vs_K",
    "plot_matching_time_vs_urate",
    "plot_multi_employment",
    "plot_response_to_demand",
    "plot_vacancy_time_series",
]
