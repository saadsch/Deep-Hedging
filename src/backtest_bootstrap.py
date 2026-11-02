"""
backtest_bootstrap.py
---------------------
Improvement of the deep hedge: train it on REALISTIC paths instead of
Black-Scholes paths, and test it honestly out of sample.

Training paths = block bootstrap of real history. We cut the real S&P 500
history into blocks of 21 consecutive trading days (about one month) and glue
randomly chosen blocks together into 1-year paths. Each block keeps its real
daily returns AND the real r, q and implied vol of those days, so the training
paths contain real crashes, volatility spikes and rebounds, while the order of
the months is reshuffled (so the network cannot memorise the real history).

Out-of-sample design (2 folds, no overlap):
  fold A: train on history up to the split date -> test on BRCs living entirely after it
  fold B: train on history after the split date -> test on BRCs living entirely before it
Every real BRC is therefore tested by a network that never saw its period.

Three strategies on the same out-of-sample BRCs:
  BS delta          Black-Scholes delta
  deep (GBM)        network trained on Black-Scholes paths (as in backtest_real.py)
  deep (bootstrap)  network trained on bootstrapped real paths

Run from the project root:   python src/backtest_bootstrap.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import backtest_real as bt

SPLIT = "2012-07-01"          # boundary between the two folds
BLOCK = 21                    # days per bootstrap block (one month)
COSTS_BP = [0.0, 5.0, 10.0]
SEED = 0


def bootstrap_paths(df, n, seed):
    """1-year paths glued from random 21-day blocks of real history in `df`."""
    rng = np.random.default_rng(seed)
    logret = np.diff(np.log(df["spot"].to_numpy()))
    r = df["r"].to_numpy()[:-1]
    q = df["q"].to_numpy()[:-1]
    s = df[bt.VOL_COL].to_numpy()[:-1]
    n_blocks = int(np.ceil(bt.N / BLOCK))
    starts = rng.integers(0, len(logret) - BLOCK, (n, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(n, -1)[:, :bt.N]
    S = 100.0 * np.exp(np.concatenate([np.zeros((n, 1)),
                                       np.cumsum(logret[idx], axis=1)], axis=1))
    return S, r[idx], q[idx], s[idx]


def brcs_in(df_all, lo, hi):
    """Real BRCs whose WHOLE life lies within [lo, hi)."""
    start, end, S, r, q, s = bt.real_brcs(df_all)
    keep = (start >= pd.Timestamp(lo)) & (end < pd.Timestamp(hi))
    return start[keep], S[keep], r[keep], q[keep], s[keep]


def evaluate(net, S, r, q, s, cost):
    S0, knocked, delta, n_puts, premium, payoff = bt.bs_inputs(S, r, q, s)
    if net is None:
        pos = n_puts[:, None] * delta
    else:
        pos = bt.positions(net, bt.features(S, S0, knocked, delta), n_puts[:, None])
    pnl, _, _ = bt.pnl_and_grad(pos, S, r, q, premium, payoff, cost)
    return pnl


def main():
    df = bt.load_inputs()
    split = pd.Timestamp(SPLIT)
    folds = [  # (name, training history, test window)
        ("A", df[df.index < split], (split, df.index[-1] + pd.Timedelta(days=1))),
        ("B", df[df.index >= split], (df.index[0], split)),
    ]

    rows, per_brc = [], []
    for name, hist, (lo, hi) in folds:
        start, S, r, q, s = brcs_in(df, lo, hi)
        print(f"\nFold {name}: train on {hist.index[0].date()} -> {hist.index[-1].date()}, "
              f"test on {len(S)} BRCs issued {start[0].date()} -> {start[-1].date()}")
        sim_gbm = bt.simulated_training_set(hist, bt.N_TRAIN, SEED)
        sim_boot = bootstrap_paths(hist, bt.N_TRAIN, SEED)
        for bp in COSTS_BP:
            c = bp / 10_000
            print(f"  cost {bp:.0f} bp: training on GBM paths, then on bootstrapped paths...")
            net_gbm = bt.train(sim_gbm, c, SEED)
            net_boot = bt.train(sim_boot, c, SEED)
            for strat, net in [("BS delta", None), ("deep (GBM)", net_gbm),
                               ("deep (bootstrap)", net_boot)]:
                pnl = evaluate(net, S, r, q, s, c)
                per_brc.append(pd.DataFrame({"issue": start, "fold": name, "cost_bp": bp,
                                             "strategy": strat, "pnl": pnl}))

    res = pd.concat(per_brc, ignore_index=True)
    table = (res.groupby(["cost_bp", "strategy"])["pnl"]
                .apply(lambda p: pd.Series(bt.stats(p.to_numpy())))
                .unstack())
    order = ["BS delta", "deep (GBM)", "deep (bootstrap)"]
    table = table.reindex(order, level=1)
    print(f"\nOUT-OF-SAMPLE hedging P&L per BRC on 1,000 notional "
          f"({res.groupby('cost_bp').size().iloc[0] // 3} real BRCs, vol: {bt.VOL_COL}):")
    print(table.round(2).to_string())

    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/backtest_bootstrap.csv", index=False)
    table.round(4).to_csv("results/backtest_bootstrap_summary.csv")
    print("\nsaved results/backtest_bootstrap.csv and results/backtest_bootstrap_summary.csv")


if __name__ == "__main__":
    main()
