# Beveridge Curve Simulation

A micro-founded agent-based model that simulates the Beveridge Curve (unemployment-vacancy relationship) by modeling how firms dynamically adjust their workforce in response to economic demand shocks.

## Project Structure

```
BeveridgeV2/
├── src/
│   ├── beveridge/                 # Simulation core (config, firm, economy, signals, experiments, main)
│   ├── plotting/                # Figures, Beveridge trajectory, diagnostics, aggregate rates
│   ├── generators/              # CLIs that write pickles under data/
│   ├── legacy/                  # Optional test signals (not used by default main)
│   ├── time_grid.py             # DT, STEPS, TIME, interpolate_series
│   ├── discrete.txt             # RevTeX theory write-up (aligned with simulation)
│   ├── beveridge_population_base.py  # Backward-compatible shim + __main__
│   └── beveridge_plot_funcs.py  # Backward-compatible re-exports of plotting.*
│
├── data/                   # Input data files (.pkl)
│   ├── dummy_gdp.pkl      # Baseline GDP signal
│   ├── dummy_demand.pkl   # Test signals (sine, custom, shock)
│   └── ar2_signal.pkl     # AR(2) stochastic signal
│
├── output/                 # Generated plots and results
│
└── docs/                   # Short pointers; see src/discrete.txt for theory
```

## Signal pipeline

The default simulation builds its in-memory `GDP` frame from two pickle files under `data/`:

| Pickle | Generator | Role |
|--------|-----------|------|
| `dummy_gdp.pkl` | `python src/generators/dummy_gdp_gen.py` | Baseline / sine-style columns interpolated onto the model time grid. |
| `ar2_signal.pkl` | `python src/generators/ar2generator.py` | AR(2) growth series → `gdp_ar2`. Default `GDP_SIGNAL_NAME` is `gdp_ar2`. |

`src/legacy/special_signals.py` produces idealized test signals on its own grid. It is **not** imported by the default entry point.

## Quick Start

**Canonical run** (from project root; `src` must be on `PYTHONPATH`):

```bash
PYTHONPATH=src python -m beveridge.main
```

Equivalent: `cd src && python -m beveridge.main` (current directory is on `sys.path`).

**Shim entry** (prepends `src` automatically):

```bash
python src/beveridge_population_base.py
```

1. **Generate data signals** (if needed):

   ```bash
   python src/generators/dummy_gdp_gen.py
   python src/generators/ar2generator.py
   python src/legacy/special_signals.py   # optional
   ```

2. Run as above; plots go to `output/`.

## Key modules

- **`beveridge` package**: `Firm`, market loop, experiments, `main()`
- **`plotting` package**: `compute_rates`, Beveridge and diagnostic figures
- **`generators/`**: AR(2) and dummy GDP pickle writers
- **`legacy/special_signals.py`**: Optional idealized signals

## Model Overview

The model simulates firms that:

- Calculate target employment based on GDP/demand signals
- Post vacancies when understaffed
- Adjust employment through matching and separation processes
- Generate Beveridge curves showing unemployment-vacancy relationships

See `docs/obsidiannote.txt` and `src/discrete.txt` for theory.

## Notes

- Run generators and `PYTHONPATH=src` workflows from the **project root** so `data/` and `output/` resolve correctly.
- Data files are read from `data/`; outputs go to `output/`.
