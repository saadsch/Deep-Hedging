"""
robustness_seeds.py
-------------------
Is the deep-hedge result real or luck? Repeat the out-of-sample experiment of
backtest_bootstrap.py with several random seeds. The seed changes the simulated
/ bootstrapped training paths, the network initialisation and the mini-batches.
The real test BRCs and the BS delta hedge do not depend on the seed.

For each seed we record mean, VaR 5 %, CVaR 5 % and worst BRC of the
out-of-sample P&L (5 bp costs), then summarise across seeds:
average, min, max, and in how many seeds each network beats BS delta.

Run from the project root:   python src/robustness_seeds.py
(about 1.5-2 minutes per seed on 1 vCPU with INCLUDE_GBM = False)
"""

from pathlib import Path

import numpy as np
import pandas as pd

import backtest_real as bt
from backtest_bootstrap import SPLIT, bootstrap_paths, brcs_in, evaluate

SEEDS = [0, 1, 2]
INCLUDE_GBM = False          # True also retrains the GBM network (doubles the time)
COST_BP = 5.0


def main():
    df = bt.load_inputs()
    split = pd.Timestamp(SPLIT)
    folds = [("A", df[df.index < split], (split, df.index[-1] + pd.Timedelta(days=1))),
             ("B", df[df.index >= split], (df.index[0], split))]
    c = COST_BP / 10_000
    test = {name: brcs_in(df, lo, hi) for name, _, (lo, hi) in folds}

    # BS delta: same for every seed
    pnl_bs = np.concatenate([evaluate(None, *test[n][1:], c) for n, _, _ in folds])
    rows = [{"seed": "-", "strategy": "BS delta", **bt.stats(pnl_bs)}]

    for seed in SEEDS:
        pnl = {"deep (bootstrap)": []}
        if INCLUDE_GBM:
            pnl["deep (GBM)"] = []
        for name, hist, _ in folds:
            net_boot = bt.train(bootstrap_paths(hist, bt.N_TRAIN, seed), c, seed)
            pnl["deep (bootstrap)"].append(evaluate(net_boot, *test[name][1:], c))
            if INCLUDE_GBM:
                net_gbm = bt.train(bt.simulated_training_set(hist, bt.N_TRAIN, seed), c, seed)
                pnl["deep (GBM)"].append(evaluate(net_gbm, *test[name][1:], c))
        for strat, parts in pnl.items():
            rows.append({"seed": seed, "strategy": strat, **bt.stats(np.concatenate(parts))})
        print(f"seed {seed} done")

    res = pd.DataFrame(rows)
    print(f"\nOUT-OF-SAMPLE P&L per BRC, {COST_BP:.0f} bp costs, {len(pnl_bs)} real BRCs, "
          f"vol: {bt.VOL_COL}\n")
    print(res.round(2).to_string(index=False))

    bs = res[res["strategy"] == "BS delta"].iloc[0]
    print("\nAcross seeds (average [min, max]) and number of seeds beating BS delta:")
    for strat in ["deep (GBM)", "deep (bootstrap)"]:
        r = res[res["strategy"] == strat]
        if r.empty:
            continue
        print(f"\n  {strat}")
        for m in ["mean", "VaR5", "CVaR5", "worst"]:
            v = r[m]
            wins = int((v > bs[m]).sum())               # higher P&L = better
            print(f"    {m:6s} {v.mean():8.2f}  [{v.min():8.2f}, {v.max():8.2f}]   "
                  f"BS {bs[m]:8.2f}   better in {wins}/{len(v)} seeds")

    Path("results").mkdir(exist_ok=True)
    res.round(4).to_csv("results/robustness_seeds.csv", index=False)
    print("\nsaved results/robustness_seeds.csv")


if __name__ == "__main__":
    main()
