# CEDIF implied-volatility data

Put the files exported at CEDIF here. They are **not** downloaded by the scripts
(licensed data). Before committing them to GitHub, ask CEDIF whether the licence
allows it; if not, commit only this README and describe the data in the paper.

## What to export

- **Underlyings:** S&P 500 (`SPX Index`) and SMI (`SMI Index`).
- **Implied volatility by moneyness:** 80, 90, 95, 100, 105, 110 % of spot.
  On Bloomberg, find the exact field names with `FLDS` ("implied volatility moneyness").
- **Maturities:** 1, 3, 6 and 12 months (12 months matters most: the product lasts one year).
- **Period:** daily, 2000 to today.

## Expected file format (one CSV per underlying)

`spx_implied_vol.csv`, `smi_implied_vol.csv`:

| date | tenor_months | moneyness_pct | implied_vol_pct |
|---|---|---|---|
| 2020-03-16 | 12 | 80 | 41.2 |

(A "long" table like this is easiest to work with in pandas. If the export is
"wide", one column per tenor and moneyness, reshape it with `pd.melt`.)

## Write down, for the paper

Source (Bloomberg or OptionMetrics), exact field names, date of download.
