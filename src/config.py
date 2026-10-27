"""
config.py - every data series used in the project, in one place.

Each entry: name -> (source, code, group, description)

Sources (all free, no account or API key):
  "yahoo" : Yahoo Finance, via the yfinance package
  "fred"  : FRED (Federal Reserve Bank of St. Louis), public CSV link
  "cboe"  : CBOE public index history files

Implied volatility by moneyness (the "smile") is NOT free. It comes from
CEDIF (Bloomberg or OptionMetrics) and is loaded separately, see
data/cedif/README.md.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CEDIF_DIR = ROOT / "data" / "cedif"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"

START_DATE = "1990-01-01"
END_DATE = "2026-10-23"   # fixed end date -> reproducible results

SERIES = {
    # --- Underlyings ------------------------------------------------------------
    "SPX":    ("yahoo", "^GSPC",    "underlying", "S&P 500 price index"),
    "SPX_TR": ("yahoo", "^SP500TR", "underlying", "S&P 500 total return index (to estimate dividends)"),
    "SMI":    ("yahoo", "^SSMI",    "underlying", "Swiss Market Index (price index)"),

    # --- Implied volatility, S&P 500 (index points = vol in %) -----------------
    "VIX":    ("cboe", "VIX",    "vol", "30-day implied vol of the S&P 500"),
    "VIX3M":  ("cboe", "VIX3M",  "vol", "3-month implied vol of the S&P 500 (from Dec 2007)"),
    "SKEW":   ("cboe", "SKEW",   "vol", "CBOE SKEW index: price of crash protection"),

    # --- Interest rates (% per year) --------------------------------------------
    # USD_1Y and CHF_3M come from FRED, which may be unreachable from some
    # networks; they are optional (clean_data.py continues without them).
    "USD_3M": ("yahoo", "^IRX", "rate", "US 13-week Treasury bill yield, in %"),
    "USD_1Y": ("fred", "DGS1",  "rate", "US 1-year Treasury yield (discounting a 1-year product)"),
    # Daily Swiss rates are not on FRED; this OECD series is MONTHLY and is
    # carried forward within each month. Replace with SARON from CEDIF/SNB if possible.
    "CHF_3M": ("fred", "IR3TIB01CHM156N", "rate", "Swiss 3-month interbank rate (monthly)"),
}
