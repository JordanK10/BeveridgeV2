"""External data fetching: BLS (CPS + CES) and FRED (GDP)."""
from .bls import fetch_bls_series
from .fred import fetch_fred_series
from .loader import load_macro_data
