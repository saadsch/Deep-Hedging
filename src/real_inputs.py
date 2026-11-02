"""
real_inputs.py
--------------
Step 2 on REAL data: for every trading day from Dec 1999 to Aug 2025, treat
the day as a possible BRC issue date and compute

  r          1-year risk-free rate      (zero curve, linear interpolation at 365 days,
                                         as the IvyDB manual describes)
  q          1-year SPX dividend yield  (index dividend table, interpolated at 365 days)
  sigma_atm  1-year implied vol at the strike   (K = 100 % of spot)
  sigma_70   1-year implied vol at the barrier  (70 % of spot)
                -> both read off the 365-day put smile of the vol surface,
                   interpolated in strike

and the par coupon of the BRC (strike 100 %, barrier 70 %, daily monitoring)
priced with Black-Scholes using sigma_atm, and, as a skew sensitivity,
with sigma_70.

Outputs
  data/clean/brc_inputs.parquet (or .csv)   one row per issue date
  figures/brc_par_coupon.png                SPX, implied vols, par coupon

Run from the project root:   python src/real_inputs.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from brc_pricing import par_coupon

CLEAN = Path("data/clean")
FIG = Path("figures")
TENOR = 365                   # days, 1-year BRC
BARRIER = 0.70                # barrier, % of initial spot
STRIKE = 1.00                 # strike, % of initial spot


# ---------------------------------------------------------------------------
# Loading (Parquet if available, otherwise the CSV fallback of build_dataset.py)
# ---------------------------------------------------------------------------
def load(name, dates=("date",)):
    pq, csv = CLEAN / f"{name}.parquet", CLEAN / f"{name}.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        df = pd.read_csv(csv)
    for c in dates:
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def as_decimal(series, label):
    """OptionMetrics stores rates in PERCENT (e.g. 5.43 = 5.43 %).
    Detect the unit from the data instead of assuming it, and say what we did."""
    med = series.median()
    if med > 0.5:
        print(f"  {label}: median {med:.3f} -> read as PERCENT, divided by 100")
        return series / 100.0
    print(f"  {label}: median {med:.4f} -> already a decimal")
    return series


# ---------------------------------------------------------------------------
# 1-year inputs, one value per date
# ---------------------------------------------------------------------------
def rate_1y(zc):
    zc = zc.sort_values(["date", "days"])
    return zc.groupby("date").apply(
        lambda g: np.interp(TENOR, g["days"], g["rate"]), include_groups=False
    ).rename("r")


def div_1y(dv):
    dv = dv.copy()
    has_exp = dv["expiration"].notna()
    dv["days"] = np.where(has_exp, (dv["expiration"] - dv["date"]).dt.days, TENOR)
    dv = dv.sort_values(["date", "days"])
    return dv.groupby("date").apply(
        lambda g: np.interp(TENOR, g["days"], g["rate"]), include_groups=False
    ).rename("q")


def vols_1y(vs, spot):
    """Implied vol at strike = 100 % and 70 % of spot, from the 365-day put smile.
    Puts span low strikes (delta -10) to high strikes (delta -90), which covers
    both the at-the-money strike and, most of the time, the barrier."""
    sm = vs[(vs["days"] == TENOR) & (vs["cp_flag"] == "P")]
    sm = sm.merge(spot.rename("spot"), left_on="date", right_index=True)
    sm["m"] = sm["impl_strike"] / sm["spot"]          # moneyness K / S
    sm = sm.sort_values(["date", "m"])

    def one_day(g):
        m, v = g["m"].to_numpy(), g["impl_volatility"].to_numpy()
        return pd.Series({
            "sigma_atm": np.interp(STRIKE, m, v),
            "sigma_70": np.interp(BARRIER, m, v),     # flat beyond the lowest strike
            "min_moneyness": m.min(),
        })
    return sm.groupby("date").apply(one_day, include_groups=False)


# ---------------------------------------------------------------------------
# Figure: three stacked panels sharing the time axis (no dual axis)
# ---------------------------------------------------------------------------
def make_figure(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    c1, c2 = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2,
                         "axes.labelcolor": ink2, "xtick.color": ink2,
                         "ytick.color": ink2, "text.color": ink})

    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True,
                             facecolor=surface, constrained_layout=True)
    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    a = axes[0]
    a.plot(df.index, df["spot"], color=c1, lw=1.2)
    a.set_title("S&P 500 index level", loc="left", fontsize=10)

    a = axes[1]
    a.plot(df.index, 100 * df["sigma_70"], color=c2, lw=1.0, label="strike 70 % (barrier)")
    a.plot(df.index, 100 * df["sigma_atm"], color=c1, lw=1.0, label="strike 100 % (at the money)")
    a.set_title("1-year implied volatility, % (OptionMetrics, 365-day put smile)",
                loc="left", fontsize=10)
    a.legend(frameon=False, loc="upper right", fontsize=8)

    a = axes[2]
    a.plot(df.index, df["coupon_70_pct"], color=c2, lw=1.0, label="priced with barrier vol")
    a.plot(df.index, df["coupon_atm_pct"], color=c1, lw=1.0, label="priced with at-the-money vol")
    a.set_title("BRC par coupon, % p.a. (1y, strike 100 %, barrier 70 %, daily)",
                loc="left", fontsize=10)
    a.legend(frameon=False, loc="upper right", fontsize=8)

    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    print("Loading clean data...")
    px = load("spx_prices").set_index("date")["close"].rename("spot")
    zc = load("zero_curve")
    dv = load("spx_div_yield", dates=("date", "expiration"))
    vs = load("spx_vol_surface")

    print("Units check:")
    zc["rate"] = as_decimal(zc["rate"], "zero curve rate")
    dv["rate"] = as_decimal(dv["rate"], "dividend yield")

    print("Computing 1-year r, q, sigma for every date (a few minutes)...")
    df = pd.concat([px, rate_1y(zc), div_1y(dv), vols_1y(vs, px)], axis=1)
    n_all = len(df)
    df = df.dropna(subset=["spot", "r", "q", "sigma_atm"])
    print(f"  {len(df):,} of {n_all:,} dates have all inputs")

    print("Pricing the BRC on every date...")
    T = 1.0
    for col, out in [("sigma_atm", "coupon_atm"), ("sigma_70", "coupon_70")]:
        df[out] = [par_coupon(100.0, STRIKE, BARRIER, T, r, q, s)
                   for r, q, s in zip(df["r"], df["q"], df[col])]
        df[out + "_pct"] = df[out] / 10.0               # per 1,000 notional -> %

    extrap = (df["min_moneyness"] > BARRIER).mean()
    print(f"  barrier strike below the lowest quoted strike on {extrap:.1%} of dates "
          f"(sigma_70 held flat there)")

    try:
        df.to_parquet(CLEAN / "brc_inputs.parquet")
        print("  saved data/clean/brc_inputs.parquet")
    except ImportError:
        df.to_csv(CLEAN / "brc_inputs.csv")
        print("  saved data/clean/brc_inputs.csv")

    FIG.mkdir(exist_ok=True)
    make_figure(df, FIG / "brc_par_coupon.png")
    print("  saved figures/brc_par_coupon.png")

    print("\nSummary (1-year inputs and par coupon, % per year):")
    show = pd.DataFrame({
        "r": 100 * df["r"], "q": 100 * df["q"],
        "vol ATM": 100 * df["sigma_atm"], "vol 70%": 100 * df["sigma_70"],
        "coupon (ATM vol)": df["coupon_atm_pct"],
        "coupon (70% vol)": df["coupon_70_pct"],
    })
    print(show.describe(percentiles=[0.05, 0.5, 0.95]).round(2).to_string())

    print("\nSelected dates:")
    for d in ["2000-03-24", "2007-06-01", "2008-11-20", "2017-06-01",
              "2020-03-16", "2022-10-12", "2025-08-29"]:
        i = df.index.searchsorted(pd.Timestamp(d))     # first date on or after d
        if i < len(df):
            row = df.iloc[i]
            print(f"  {df.index[i].date()}  SPX {row['spot']:8.1f}  r {100*row['r']:5.2f}%  "
                  f"q {100*row['q']:4.2f}%  vol ATM {100*row['sigma_atm']:5.1f}%  "
                  f"coupon {row['coupon_atm_pct']:5.2f}%")


if __name__ == "__main__":
    main()
