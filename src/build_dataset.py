"""
build_dataset.py
----------------
Turns the raw OptionMetrics IvyDB US files copied from CEDIF into small,
clean, SPX-only Parquet files that the rest of the project uses.

Run ONCE:   python build_dataset.py

Input  : data/raw/   (IVYSECUR, IVYSECPR, IVYZEROC, IVYIDXDV, IVYVSURF files,
                      .txt or .zip, any sub-folder: History/ and Update/ both fine)
Output : data/clean/ spx_prices.parquet, zero_curve.parquet,
                     spx_div_yield.parquet, spx_vol_surface.parquet

Column orders come from the IvyDB US Reference Manual v7.0 ("File Formats").
The script checks the number of columns of every file and STOPS if it does not
match the manual, so nothing is silently mislabelled.

Requirements: pandas (pyarrow optional: Parquet output, otherwise CSV)
"""

import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Settings - change RAW_DIR if your raw files are on an external disk
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
TICKER = "SPX"
CHUNK = 2_000_000          # rows read at a time (keeps memory low)
MISSING = -99.99           # OptionMetrics code for "not computed"

# Column names, in file order, from the Reference Manual
COLUMNS = {
    "IVYSECUR": ["secid", "cusip", "ticker", "sic", "index_flag",
                 "exchange_flags", "class", "issue_type", "industry_group"],
    "IVYSECPR": ["secid", "date", "bid_low", "ask_high", "close", "volume",
                 "total_return", "cfadj", "open", "shrout", "cfret"],
    "IVYZEROC": ["date", "days", "rate"],
    # Expiration was added in v6.0 -> older files have 3 columns, newer 4
    "IVYIDXDV": ["secid", "date", "expiration", "rate"],
    "IVYVSURF": ["secid", "date", "days", "delta", "cp_flag",
                 "impl_volatility", "impl_strike", "impl_premium", "dispersion"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_files(prefix):
    """All raw files whose name starts with prefix (case-insensitive), sorted."""
    files = [p for p in RAW_DIR.rglob("*")
             if p.is_file() and p.name.upper().startswith(prefix)]
    if not files:
        raise FileNotFoundError(f"No {prefix} file found in {RAW_DIR}")
    return sorted(files)


def open_text(path):
    """Return a file handle on the .txt, opening it inside the .zip if needed."""
    if path.suffix.lower() == ".zip":
        z = zipfile.ZipFile(path)
        inner = [n for n in z.namelist() if not n.endswith("/")][0]
        return z.open(inner)
    return open(path, "rb")


def read_filtered(prefix, keep=None, secid=None):
    """Read every file of one table in chunks; keep only `secid` rows if given."""
    parts = []
    for path in find_files(prefix):
        with open_text(path) as fh:
            # IVYIDXDV: old files have 3 columns, newer ones 4 (Expiration added
            # in v6). A stacked file can mix both, so read 4 columns and realign.
            extra = {"names": range(4)} if prefix == "IVYIDXDV" else {}
            for chunk in pd.read_csv(fh, sep="\t", header=None, dtype=str,
                                     chunksize=CHUNK, keep_default_na=False,
                                     **extra):
                chunk = chunk.apply(lambda s: s.str.strip())
                cols = COLUMNS[prefix]
                if prefix == "IVYIDXDV":
                    old = chunk[3] == ""                     # 3-column rows
                    chunk.loc[old, 3] = chunk.loc[old, 2]    # rate was in col 3
                    chunk.loc[old, 2] = ""                   # no expiration
                if chunk.shape[1] != len(cols):
                    raise ValueError(
                        f"{path.name}: {chunk.shape[1]} columns, manual says "
                        f"{len(cols)}. Check the file before going further.")
                chunk.columns = cols
                if secid is not None:
                    chunk = chunk[chunk["secid"] == str(secid)]
                if len(chunk):
                    parts.append(chunk)
        print(f"  read {path.name}")
    df = pd.concat(parts, ignore_index=True)
    return df[keep] if keep else df


def to_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save(df, name):
    """Save as Parquet if pyarrow is installed, otherwise as CSV."""
    try:
        df.to_parquet(CLEAN_DIR / f"{name}.parquet", index=False)
        print(f"  saved {name}.parquet")
    except ImportError:
        df.to_csv(CLEAN_DIR / f"{name}.csv", index=False)
        print(f"  saved {name}.csv  (install pyarrow for Parquet)")


def to_date(df, cols):
    for c in cols:
        df[c] = pd.to_datetime(df[c], format="%Y%m%d", errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Find the SECID of SPX (the index, issue type "A")
    print("1/5 Security table")
    sec = read_filtered("IVYSECUR")
    spx = sec[(sec["ticker"] == TICKER) & (sec["index_flag"] == "1")]
    if len(spx) != 1:
        print(spx)
        raise ValueError(f"Expected exactly one index with ticker {TICKER}")
    secid = int(spx["secid"].iloc[0])
    print(f"  {TICKER} -> SECID {secid}")

    # 2. Prices
    print("2/5 Security_Price")
    px = read_filtered("IVYSECPR", secid=secid)
    px = to_date(px, ["date"])
    px = to_numeric(px, ["bid_low", "ask_high", "close", "volume",
                         "total_return", "cfadj", "open", "cfret"])
    px["close"] = px["close"].abs()   # negative = bid/ask midpoint (manual)
    px = px.drop_duplicates("date", keep="last").sort_values("date")
    save(px, "spx_prices")

    # 3. Zero curve (same for all securities, no filter)
    print("3/5 Zero_Curve")
    zc = read_filtered("IVYZEROC")
    zc = to_date(zc, ["date"])
    zc = to_numeric(zc, ["days", "rate"])
    zc = zc.drop_duplicates(["date", "days"], keep="last")
    zc = zc.sort_values(["date", "days"])
    save(zc, "zero_curve")

    # 4. Index dividend yield
    print("4/5 Index_Dividend")
    dv = read_filtered("IVYIDXDV", secid=secid)
    dv = to_date(dv, ["date", "expiration"])
    dv = to_numeric(dv, ["rate"])
    dv = dv.drop_duplicates(["date", "expiration"], keep="last").sort_values("date")
    save(dv, "spx_div_yield")

    # 5. Volatility surface
    print("5/5 Volatility_Surface (largest, be patient)")
    vs = read_filtered("IVYVSURF", secid=secid)
    vs = to_date(vs, ["date"])
    vs = to_numeric(vs, ["days", "delta", "impl_volatility", "impl_strike",
                         "impl_premium", "dispersion"])
    vs = vs[vs["impl_volatility"] != MISSING]          # drop "not computed"
    vs = vs.drop_duplicates(["date", "days", "delta", "cp_flag"], keep="last")
    vs = vs.sort_values(["date", "days", "cp_flag", "delta"])
    save(vs, "spx_vol_surface")

    # Summary so you can check everything looks right
    print("\nDone. Summary:")
    for name, df in [("prices", px), ("zero curve", zc),
                     ("dividend yield", dv), ("vol surface", vs)]:
        print(f"  {name:15s} {len(df):>10,} rows  "
              f"{df['date'].min().date()} -> {df['date'].max().date()}")


if __name__ == "__main__":
    main()
