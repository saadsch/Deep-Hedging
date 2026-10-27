"""
make_figures.py - step 3: three figures to look at the data (and show at the meeting).

  fig1_spx_and_vix.png       S&P 500 and VIX, with crisis windows shaded
  fig2_implied_vs_realised.png  VIX vs 21-day realised vol: the volatility risk premium
  fig3_skew.png              CBOE SKEW: how expensive crash protection is

Run from the repository root, after clean_data.py:
    python src/make_figures.py
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import pandas as pd

from config import FIG_DIR, PROCESSED_DIR

BLUE, ORANGE, INK, GRID, SHADE = "#2a78d6", "#eb6834", "#3b3b38", "#e4e3dd", "#d9d8d2"
CRISES = [("2008-09-01", "2009-03-31"), ("2020-02-15", "2020-04-30"), ("2022-01-01", "2022-10-31")]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
})


def style(ax):
    for start, end in CRISES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=SHADE, alpha=0.6, lw=0)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    us = pd.read_csv(PROCESSED_DIR / "panel_spx.csv", index_col="date", parse_dates=True)

    # 1. Two stacked panels (one y-axis each) rather than a dual-axis chart.
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    a1.plot(us.index, us["price"], color=BLUE, lw=1.1)
    a1.set_yscale("log")
    a1.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    a1.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a1.set_title("S&P 500 (log scale)", loc="left")
    a2.plot(us.index, us["VIX"], color=ORANGE, lw=1.0)
    a2.set_title("VIX: 30-day implied volatility, %", loc="left")
    for ax in (a1, a2):
        style(ax)
    fig.suptitle("Underlying and implied volatility (grey: 2008, 2020, 2022)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_spx_and_vix.png", dpi=150)
    plt.close(fig)

    # 2. Implied vs realised volatility, both in % per year.
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(us.index, us["VIX"], color=ORANGE, lw=1.0, label="Implied (VIX)")
    ax.plot(us.index, us["rv_21d"], color=BLUE, lw=1.0, label="Realised, 21 days")
    style(ax)
    ax.set_ylabel("volatility, % per year")
    ax.set_title("Implied volatility is usually above realised: the gap is what option buyers pay",
                 loc="left", fontsize=10.5)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_implied_vs_realised.png", dpi=150)
    plt.close(fig)

    # 3. SKEW index.
    if us["SKEW"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(us.index, us["SKEW"], color=BLUE, lw=1.0)
        style(ax)
        ax.set_title("CBOE SKEW index: higher = crash protection more expensive",
                     loc="left", fontsize=10.5)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_skew.png", dpi=150)
        plt.close(fig)

    print(f"Figures saved in {FIG_DIR}")


if __name__ == "__main__":
    main()
