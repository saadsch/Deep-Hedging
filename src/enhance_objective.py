"""
enhance_objective.py
--------------------
Enhancement of the deep hedge, tested out of sample on the real BRCs.

Two changes compared with backtest_bootstrap.py:

1. Mean-CVaR objective. The network minimises
        CVaR_5%(loss) - LAMBDA * mean(P&L)
   LAMBDA = 0 is the previous network (pure tail objective). Larger LAMBDA makes
   it also care about the average P&L of the 95 % of non-tail BRCs.
   Sweeping LAMBDA gives a trade-off curve (average vs tail) instead of one point.

2. De-meaned bootstrap. Real history contains the equity risk premium (the
   S&P 500 drifts up ~7 % a year). A network rewarded on mean P&L could learn to
   bet on that drift instead of hedging. We remove the historical average
   daily return of the training period and replace it with the risk-neutral
   drift (r - q). Crashes, volatility clusters and rebounds inside each
   21-day block are kept unchanged.

Same out-of-sample design as before: train on one half of history, test on BRCs
living entirely in the other half, and vice versa. 5 bp costs.

Run from the project root:   python src/enhance_objective.py
(about 40 s per training on 1 vCPU: 2 folds x len(LAMBDAS) trainings)
"""

from pathlib import Path

import numpy as np
import pandas as pd

import backtest_real as bt
from backtest_bootstrap import SPLIT, BLOCK, brcs_in, evaluate
from deep_hedge import MLP

LAMBDAS = [0.0, 1.0, 3.0, 10.0]
DEMEAN = True
COST_BP = 5.0
SEED = 2


# ---------------------------------------------------------------------------
# Training paths: block bootstrap, optionally with the drift removed
# ---------------------------------------------------------------------------
def bootstrap_paths(df, n, seed, demean=DEMEAN):
    rng = np.random.default_rng(seed)
    logret = np.diff(np.log(df["spot"].to_numpy()))
    r = df["r"].to_numpy()[:-1]
    q = df["q"].to_numpy()[:-1]
    s = df[bt.VOL_COL].to_numpy()[:-1]
    if demean:
        # keep every day's shock, replace the average drift by the
        # risk-neutral one: (r - q) dt - 0.5 * variance
        shock = logret - logret.mean()
        logret = shock + (r - q) * bt.DT - 0.5 * logret.var()
    n_blocks = int(np.ceil(bt.N / BLOCK))
    starts = rng.integers(0, len(logret) - BLOCK, (n, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(n, -1)[:, :bt.N]
    S = 100.0 * np.exp(np.concatenate([np.zeros((n, 1)),
                                       np.cumsum(logret[idx], axis=1)], axis=1))
    return S, r[idx], q[idx], s[idx]


# ---------------------------------------------------------------------------
# Training with the mean-CVaR objective
# ---------------------------------------------------------------------------
def train(sim, cost, seed, lam):
    """Minimise  w + E[max(loss - w, 0)] / alpha  +  lam * E[loss]
    (Rockafellar-Uryasev CVaR plus lam times the expected loss)."""
    S, r, q, s = sim
    S0, knocked, delta, n_puts, premium, payoff = bt.bs_inputs(S, r, q, s)
    X = bt.features(S, S0, knocked, delta, s)
    net, w, wm, wv = MLP(X.shape[-1], bt.HIDDEN, seed), 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    B = bt.BATCH
    for it in range(1, bt.N_ITER + 1):
        b = rng.choice(len(S), B, replace=False)
        theta = net.forward(X[b])
        pos = n_puts[b, None] * theta
        pnl, g, _ = bt.pnl_and_grad(pos, S[b], r[b], q[b], premium[b], payoff[b], cost)
        tail = (-pnl > w).astype(float)
        d_loss = tail / (bt.ALPHA * B) + lam / B        # dObjective / dloss_j
        d_theta = -d_loss[:, None] * g * n_puts[b, None]
        net.adam_step(net.backward(d_theta), bt.LR)
        gw = 1 - tail.mean() / bt.ALPHA
        wm, wv = 0.9 * wm + 0.1 * gw, 0.999 * wv + 0.001 * gw**2
        w -= 0.5 * (wm / (1 - 0.9**it)) / (np.sqrt(wv / (1 - 0.999**it)) + 1e-8)
    return net


# ---------------------------------------------------------------------------
def main():
    df = bt.load_inputs()
    split = pd.Timestamp(SPLIT)
    folds = [("A", df[df.index < split], (split, df.index[-1] + pd.Timedelta(days=1))),
             ("B", df[df.index >= split], (df.index[0], split))]
    c = COST_BP / 10_000
    test = {name: brcs_in(df, lo, hi) for name, _, (lo, hi) in folds}

    pnl_bs = np.concatenate([evaluate(None, *test[n][1:], c) for n, _, _ in folds])
    rows = [{"strategy": "BS delta", "lambda": np.nan, **bt.stats(pnl_bs),
             "total": pnl_bs.sum()}]
    sims = {name: bootstrap_paths(hist, bt.N_TRAIN, SEED) for name, hist, _ in folds}

    for lam in LAMBDAS:
        parts = []
        for name, _, _ in folds:
            net = train(sims[name], c, SEED, lam)
            parts.append(evaluate(net, *test[name][1:], c))
        p = np.concatenate(parts)
        rows.append({"strategy": "deep", "lambda": lam, **bt.stats(p), "total": p.sum()})
        print(f"lambda {lam:5.1f} done")

    res = pd.DataFrame(rows)
    print(f"\nOUT-OF-SAMPLE P&L per BRC, {COST_BP:.0f} bp costs, {len(pnl_bs)} real BRCs, "
          f"de-meaned bootstrap: {DEMEAN}, seed {SEED}")
    print("(total = sum of P&L over all BRCs; higher is better everywhere)\n")
    print(res.round(2).to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    res.round(4).to_csv("results/enhance_objective.csv", index=False)
    print("\nsaved results/enhance_objective.csv")
    Path("figures").mkdir(exist_ok=True)
    make_figure(res, "figures/enhance_objective.png")
    print("saved figures/enhance_objective.png")


def make_figure(res, path):
    """Trade-off curve: average P&L (x) vs CVaR 5 % (y). Up and right = better."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    c1, c2 = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2})
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=surface, constrained_layout=True)
    ax.set_facecolor(surface)
    ax.grid(color=grid, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    deep = res[res["strategy"] == "deep"]
    ax.plot(deep["mean"], deep["CVaR5"], color=c2, lw=2, marker="o", ms=8,
            label="deep hedge (mean-CVaR)")
    for _, row in deep.iterrows():
        ax.annotate(f"λ = {row['lambda']:g}", (row["mean"], row["CVaR5"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color=ink2)
    bs = res[res["strategy"] == "BS delta"].iloc[0]
    ax.plot(bs["mean"], bs["CVaR5"], color=c1, marker="s", ms=9, ls="none",
            label="Black-Scholes delta")
    ax.set_xlabel("average P&L per BRC (per 1,000)", color=ink2)
    ax.set_ylabel("CVaR 5 % (per 1,000)", color=ink2)
    ax.set_title("Out-of-sample trade-off, 5 bp costs (up and right is better)",
                 loc="left", fontsize=10, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
