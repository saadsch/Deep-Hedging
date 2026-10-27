"""
clean_data.py - step 2: build one clean daily panel per underlying.

Why one panel per underlying (not one shared calendar)?
  A hedge is rebalanced on the days the underlying trades. The S&P 500 panel
  therefore follows New York trading days, the SMI panel Zurich trading days.
  Volatility indices and rates are attached to those days.

Steps:
  1. Load the raw CSVs.
  2. Calendar = days on which the underlying has a price.
  3. Attach vol and rates; carry the last value forward over short gaps
     (holidays); monthly Swiss rates are carried within the month.
  4. Add the quantities the hedging study needs:
       ret_log    daily log return of the price index, in %
       rv_21d     realised volatility over 21 days, annualised, in %
       div_yield  dividend yield over the past year, in % (S&P 500 only):
                  total-return growth minus price growth
       vrp        VIX minus rv_21d: the volatility risk premium, in vol points
       div_yield is smoothed with a trailing 21-day median (div_yield_raw kept)
  5. Save the panels and a data report. Extreme moves are flagged, never removed.

Run from the repository root, after download_data.py:
    python src/clean_data.py
"""

import numpy as np
import pandas as pd

from config import CEDIF_DIR, PROCESSED_DIR, RAW_DIR, SERIES

TRADING_DAYS = 252
FILL_LIMIT_DAILY = 5     # business days a daily series may be carried forward
FILL_LIMIT_MONTHLY = 25  # for the monthly Swiss rate: carried within its month
OUTLIER_MADS = 10


def load(name: str) -> pd.Series:
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run download_data.py first")
    s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["value"].astype(float)
    return s[~s.index.duplicated(keep="last")].sort_index().rename(name)


def attach(calendar: pd.DatetimeIndex, s: pd.Series, limit: int) -> tuple[pd.Series, float]:
    """Put a series on the underlying's calendar and fill short gaps.

    Returns the aligned series and the share (in %) of values that were filled,
    measured inside the series' own history.
    """
    union = s.index.union(calendar)
    filled = s.reindex(union).ffill(limit=limit).reindex(calendar)
    first, last = s.index.min(), s.index.max()
    if limit > FILL_LIMIT_DAILY:      # monthly series: its last value covers its whole month
        last = last + pd.offsets.MonthEnd(0)
    inside = (calendar >= first) & (calendar <= last)
    original = s.reindex(calendar)
    filled[~inside] = np.nan           # never extend a series beyond its own dates
    n_filled = (original.isna() & filled.notna())[inside].sum()
    share = 100 * n_filled / max(inside.sum(), 1)
    return filled, round(float(share), 2)


def build_panel(price: str, extras: dict[str, int], tr: str | None = None):
    """One panel on the calendar of the underlying `price`.

    extras: {series name: fill limit}
    tr:     name of the total-return index, if any (for the dividend yield)
    """
    p = load(price)
    p = p[p > 0]
    cal = p.index
    panel = pd.DataFrame({"price": p})
    fill_info = {}

    for name, limit in extras.items():
        if not (RAW_DIR / f"{name}.csv").exists():
            print(f"WARNING: {name} not downloaded; column left empty")
            panel[name], fill_info[name] = np.nan, None
            continue
        panel[name], fill_info[name] = attach(cal, load(name), limit)

    panel["ret_log"] = 100 * np.log(panel["price"]).diff()
    panel["rv_21d"] = panel["ret_log"].rolling(21).std() * np.sqrt(TRADING_DAYS)

    if tr is not None and (RAW_DIR / f"{tr}.csv").exists():
        t, fill_info[tr] = attach(cal, load(tr), FILL_LIMIT_DAILY)
        panel["tr_index"] = t
        # Dividend yield = what the total-return index earned beyond the price,
        # summed over the past year (log terms, so it is already annual).
        extra = np.log(panel["tr_index"]).diff() - np.log(panel["price"]).diff()
        raw_yield = 100 * extra.rolling(TRADING_DAYS).sum()
        # Dividend yields move slowly, so a one-day jump is a data error, not news
        # (e.g. 5.5 % on 23 Jan 1991 in Yahoo's total-return series). Smooth with a
        # TRAILING 21-day median: it uses past days only, so no look-ahead bias.
        panel["div_yield_raw"] = raw_yield
        panel["div_yield"] = raw_yield.rolling(21).median()

    if "VIX" in panel and panel["VIX"].notna().any():
        panel["vrp"] = panel["VIX"] - panel["rv_21d"]

    panel.index.name = "date"
    return panel, fill_info


def flag_outliers(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    """Daily moves far from normal, using the median and MAD (robust to crashes)."""
    rows = []
    for col in ["ret_log"] + [c for c in ["VIX", "SKEW"] if c in panel and panel[c].notna().any()]:
        x = panel[col].diff() if col != "ret_log" else panel[col]
        x = x.dropna()
        mad = 1.4826 * (x - x.median()).abs().median()
        if mad == 0:
            continue
        z = (x - x.median()) / mad
        for d, v in x[z.abs() > OUTLIER_MADS].items():
            rows.append({"panel": label, "column": col, "date": d.date(),
                         "move": round(v, 2), "robust_z": round(z[d], 1)})
    return pd.DataFrame(rows, columns=["panel", "column", "date", "move", "robust_z"])


def describe(panel: pd.DataFrame, fill_info: dict, label: str) -> pd.DataFrame:
    rows = []
    for col in panel.columns:
        s = panel[col].dropna()
        if s.empty:
            rows.append({"panel": label, "column": col, "start": "missing", "end": "",
                         "n": 0, "filled_%": "", "mean": "", "std": "", "min": "", "max": ""})
            continue
        rows.append({"panel": label, "column": col,
                     "start": s.index.min().date(), "end": s.index.max().date(),
                     "n": len(s), "filled_%": fill_info.get(col, 0.0),
                     "mean": round(s.mean(), 3), "std": round(s.std(), 3),
                     "min": round(s.min(), 2), "max": round(s.max(), 2)})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_\n"
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    us, us_fill = build_panel(
        "SPX",
        {"VIX": FILL_LIMIT_DAILY, "VIX3M": FILL_LIMIT_DAILY, "SKEW": FILL_LIMIT_DAILY,
         "USD_3M": FILL_LIMIT_DAILY, "USD_1Y": FILL_LIMIT_DAILY},
        tr="SPX_TR",
    )
    ch, ch_fill = build_panel("SMI", {"CHF_3M": FILL_LIMIT_MONTHLY})

    us.to_csv(PROCESSED_DIR / "panel_spx.csv")
    ch.to_csv(PROCESSED_DIR / "panel_smi.csv")

    table = pd.concat([describe(us, us_fill, "SPX"), describe(ch, ch_fill, "SMI")])
    outliers = pd.concat([flag_outliers(us, "SPX"), flag_outliers(ch, "SMI")])
    outliers.to_csv(PROCESSED_DIR / "outliers_to_check.csv", index=False)

    cedif_files = sorted(p.name for p in CEDIF_DIR.glob("*.csv")) if CEDIF_DIR.exists() else []
    report = [
        "# Data report", "",
        "Two daily panels, each on the trading days of its underlying.", "",
        "## Coverage and summary", "", to_markdown(table),
        "## Extreme daily moves (kept; check each against a real event)", "",
        to_markdown(outliers),
        "## CEDIF implied-volatility files found", "",
        ("\n".join(f"- {f}" for f in cedif_files) or "_none yet (see data/cedif/README.md)_") + "\n",
        "## Cleaning rules", "",
        f"- Daily vol and rate series carried forward at most {FILL_LIMIT_DAILY} trading days.",
        f"- Monthly Swiss rate carried forward at most {FILL_LIMIT_MONTHLY} trading days.",
        "- No series is extended before its first or after its last observation.",
        "- Units: returns, volatilities, rates and dividend yield in %; VIX, VIX3M are vol in %.",
    ]
    (PROCESSED_DIR / "data_report.md").write_text("\n".join(report) + "\n")

    print(to_markdown(table[["panel", "column", "start", "end", "n", "filled_%"]]))
    print(f"{len(outliers)} extreme moves listed in data/processed/outliers_to_check.csv")


if __name__ == "__main__":
    main()
