"""
download_data.py - step 1: download every series in config.py, untouched.

Each series is saved to data/raw/<NAME>.csv with two columns: date, value.
Raw files are never edited by hand; all cleaning is in clean_data.py.

Run from the repository root:
    python src/download_data.py
"""

import io
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

from config import END_DATE, RAW_DIR, SERIES, START_DATE

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research project)"}


def from_yahoo(ticker: str) -> pd.DataFrame:
    """Daily closing level from Yahoo Finance."""
    import yfinance as yf  # imported here so FRED/CBOE still work without it

    raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                      auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"Yahoo returned no data for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance: one column per ticker
        close = close.iloc[:, 0]
    return pd.DataFrame({"date": close.index.tz_localize(None), "value": close.values})


def from_fred(code: str) -> pd.DataFrame:
    """Series from FRED's public CSV link. Holidays are written as '.' -> NaN."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), na_values=".")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def from_cboe(code: str) -> pd.DataFrame:
    """Index history file published by CBOE.

    The file has a DATE column and one or more value columns (for VIX: open,
    high, low, close). We keep the LAST column, which is the closing level.
    """
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{code}_History.csv"
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df = df[[df.columns[0], df.columns[-1]]]
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df


FETCHERS = {"yahoo": from_yahoo, "fred": from_fred, "cboe": from_cboe}


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log = [f"Download run: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"]
    failed = []
    for name, (source, code, _group, desc) in SERIES.items():
        try:
            df = FETCHERS[source](code)
            df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
            df = df.dropna(subset=["value"]).sort_values("date")
            df.to_csv(RAW_DIR / f"{name}.csv", index=False)
            line = (f"OK    {name:<7} {source}:{code:<16} {len(df):>6} rows  "
                    f"{df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}  ({desc})")
        except Exception as err:  # one broken source must not stop the others
            failed.append(name)
            line = f"FAIL  {name:<7} {source}:{code:<16} {type(err).__name__}: {err}"
        print(line)
        log.append(line)
    (RAW_DIR / "download_log.txt").write_text("\n".join(log) + "\n")
    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}. Re-run, or check the source.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
