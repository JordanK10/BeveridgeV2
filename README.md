# Beveridge Curve Simulation

A micro-founded agent-based model that simulates the Beveridge Curve (unemployment-vacancy relationship) by modeling how firms dynamically adjust their workforce in response to economic demand shocks.

## Project Structure

```
BeveridgeV2/
├── src/                    # Python source scripts
│   ├── beveridge.py       # Main simulation engine
│   ├── ar2generator.py    # Generates AR(2) stochastic GDP signals
│   ├── special_signals.py # Generates test signals (sine, step, custom)
│   └── dummy_gdp_gen.py   # Generates simple baseline GDP signal
│
├── data/                   # Input data files (.pkl)
│   ├── dummy_gdp.pkl      # Baseline GDP signal
│   ├── dummy_demand.pkl   # Test signals (sine, custom, shock)
│   └── ar2_signal.pkl     # AR(2) stochastic signal
│
├── output/                 # Generated plots and results
│   ├── *.png              # Main output plots
│   ├── demand_shock/      # Shock experiment results
│   ├── gdp_curve/         # AR(2) experiment results
│   └── interactive_output/ # Interactive experiment results
│
└── docs/                   # Documentation
    ├── obsidiannote.txt   # Theoretical model derivation
    └── note.txt           # Additional notes
```

## Quick Start

1. **Generate data signals** (if needed):
   ```bash
   python src/dummy_gdp_gen.py      # Creates baseline GDP signal
   python src/special_signals.py    # Creates test signals
   python src/ar2generator.py       # Creates AR(2) stochastic signal
   ```

2. **Run the main simulation**:
   ```bash
   python src/beveridge.py
   ```

   This will:
   - Load GDP signals from `data/`
   - Run the Beveridge Curve simulation
   - Save plots to `output/`

## Key Scripts

- **`beveridge.py`**: Main simulation engine with `Firm` class and market dynamics
- **`special_signals.py`**: Generates idealized test signals (sine waves, step functions, custom sawtooth)
- **`ar2generator.py`**: Generates realistic stochastic GDP with engineered recessions
- **`dummy_gdp_gen.py`**: Simple constant/baseline GDP signal generator

## Model Overview

The model simulates firms that:
- Calculate target employment based on GDP/demand signals
- Post vacancies when understaffed
- Adjust employment through matching and separation processes
- Generate Beveridge Curves showing unemployment-vacancy relationships

See `docs/obsidiannote.txt` for the theoretical foundation.

## Notes

- All scripts are designed to be run from the project root directory
- Output files are automatically saved to `output/` with appropriate subdirectories
- Data files are read from `data/` directory
- The model supports parameter sweeps to explore different configurations
