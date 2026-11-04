"""
tune_validated.py
-----------------
Improve the deep hedge WITHOUT overfitting the test BRCs.

Two stages, run in this order:

  python src/tune_validated.py validate   (~25 min)  choose settings on validation data
  python src/tune_validated.py test       (~10 min)  evaluate the chosen settings ONCE

VALIDATE
  Each fold's training history is split again in time:
    fold A: fit on 1999-12 -> 2007-06, validate on BRCs living in 2007-07 -> 2012-06 (2008 crisis)
    fold B: fit on 2012-07 -> 2019-06, validate on BRCs living in 2019-07 -> 2025-08 (Covid, 2022)
  The test BRCs (the other half of history) are never touched in this stage.

  Configurations, fixed BEFORE seeing any result (each is ONE change from the baseline):
    base    block 21 days, correction +/-0.3, 300 iterations     (current best model, lambda = 1)
    block63 bootstrap blocks of 63 days (keeps multi-month crashes and rebounds intact)
    adj06   correction range +/-0.6 share per put
    long    900 training iterations instead of 300
    ens     average position of the 3 base networks (seed ensemble), no extra training

  Two validation measures (both folds pooled):
    real   the real validation BRCs (~110 BRCs: realistic but noisy)
    boot   5,000 bootstrap paths drawn from the validation period (less noisy, less realistic)

  SELECTION RULE (written before running; applied mechanically by the code):
    A change is kept only if, compared with base,
      (1) real-BRC CVaR is better in EVERY seed by more than the base seed spread, and
      (2) bootstrap CVaR is better in EVERY seed by more than the base seed spread, and
      (3) the average P&L is not worse by more than max(base seed spread of the mean, 1.0).
    For the ensemble: its CVaR must beat the average base seed by more than the spread,
    on both measures, with condition (3) as well.
    Kept changes are combined into the final configuration. If nothing passes, the
    final configuration is base. The choice is saved to results/validation_choice.json.

  Diagnostics (base networks, real validation BRCs):
    - how often the network sits at its +/-0.3 limit (is the limit binding?)
    - how much of the average-P&L gap to BS delta is extra transaction costs

TEST
  Reads results/validation_choice.json, trains the final configuration on each FULL
  training half (3 seeds), evaluates once on the 285 test BRCs, and compares with
  BS delta and with base. It refuses to run twice (results/final_test.csv exists),
  so the test set is used exactly once.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_real as bt
import deep_hedge
from backtest_bootstrap import brcs_in
from enhance_objective import train

LAMBDA = 1.0
COST_BP = 5.0
SEEDS = [0, 1, 2]
N_BOOT_VAL = 5_000
SPLIT = pd.Timestamp("2012-07-01")
VAL_A = pd.Timestamp("2007-07-01")       # fold A: validation window starts here
VAL_B = pd.Timestamp("2019-07-01")       # fold B: validation window starts here

CONFIGS = {
    "base":    dict(block=21, max_adj=0.3, n_iter=300),
    "block63": dict(block=63, max_adj=0.3, n_iter=300),
    "adj06":   dict(block=21, max_adj=0.6, n_iter=300),
    "long":    dict(block=21, max_adj=0.3, n_iter=900),
}
CHANGES = {"block63": "block", "adj06": "max_adj", "long": "n_iter"}

RESULTS = Path("results")
CHOICE = RESULTS / "validation_choice.json"
FINAL = RESULTS / "final_test.csv"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def bootstrap_paths(df, n, seed, block):
    """De-meaned block bootstrap of real history (same as enhance_objective.py,
    with the block length as a parameter)."""
    rng = np.random.default_rng(seed)
    logret = np.diff(np.log(df["spot"].to_numpy()))
    r = df["r"].to_numpy()[:-1]
    q = df["q"].to_numpy()[:-1]
    s = df[bt.VOL_COL].to_numpy()[:-1]
    logret = logret - logret.mean() + (r - q) * bt.DT - 0.5 * logret.var()
    n_blocks = int(np.ceil(bt.N / block))
    starts = rng.integers(0, len(logret) - block, (n, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :bt.N]
    S = 100.0 * np.exp(np.concatenate([np.zeros((n, 1)),
                                       np.cumsum(logret[idx], axis=1)], axis=1))
    return S, r[idx], q[idx], s[idx]


def fit(hist, cfg, seed):
    deep_hedge.MAX_ADJ = cfg["max_adj"]
    bt.N_ITER = cfg["n_iter"]
    sim = bootstrap_paths(hist, bt.N_TRAIN, seed, cfg["block"])
    return train(sim, COST_BP / 10_000, seed, LAMBDA)


def run(nets, max_adj, paths):
    """P&L, costs and fraction of time at the correction limit.
    nets: list of networks (several = ensemble, positions averaged); [] = BS delta."""
    S, r, q, s = paths
    S0, knocked, delta, n_puts, premium, payoff = bt.bs_inputs(S, r, q, s)
    at_limit = np.nan
    if nets:
        deep_hedge.MAX_ADJ = max_adj
        X = bt.features(S, S0, knocked, delta, s)
        thetas, lim = [], []
        for net in nets:
            thetas.append(net.forward(X))
            lim.append(np.abs(net.cache[3]) > 0.95 * max_adj)
        pos = n_puts[:, None] * np.mean(thetas, axis=0)
        at_limit = float(np.mean(lim))
    else:
        pos = n_puts[:, None] * delta
    pnl, _, costs = bt.pnl_and_grad(pos, S, r, q, premium, payoff, COST_BP / 10_000)
    return pnl, costs, at_limit


def summary(pnl):
    st = bt.stats(pnl)
    return {"mean": st["mean"], "std": st["std"], "VaR5": st["VaR5"],
            "CVaR5": st["CVaR5"], "worst": st["worst"]}


def folds_for_validation(df):
    """(name, fitting history, real validation BRCs, bootstrap validation paths)."""
    out = []
    for name, fit_lo, fit_hi, val_lo, val_hi in [
        ("A", df.index[0], VAL_A, VAL_A, SPLIT),
        ("B", SPLIT, VAL_B, VAL_B, df.index[-1] + pd.Timedelta(days=1)),
    ]:
        hist = df[(df.index >= fit_lo) & (df.index < fit_hi)]
        start, S, r, q, s = brcs_in(df, val_lo, val_hi)
        val_hist = df[(df.index >= val_lo) & (df.index < val_hi)]
        boot = bootstrap_paths(val_hist, N_BOOT_VAL, 999, 21)   # fixed seed: same for all configs
        out.append((name, hist, (S, r, q, s), boot, start))
    return out


# ---------------------------------------------------------------------------
# Stage 1: validation
# ---------------------------------------------------------------------------
def validate():
    df = bt.load_inputs()
    folds = folds_for_validation(df)
    for name, hist, real, _, start in folds:
        print(f"fold {name}: fit on {hist.index[0].date()} -> {hist.index[-1].date()}, "
              f"validate on {len(real[0])} BRCs issued {start[0].date()} -> {start[-1].date()}")

    rows, base_nets, t0 = [], {}, time.time()

    def record(config, seed, nets_by_fold, max_adj):
        res = {}
        for kind in ("real", "boot"):
            pnl = np.concatenate([run(nets_by_fold[f[0]], max_adj, f[2] if kind == "real" else f[3])[0]
                                  for f in folds])
            res[kind] = summary(pnl)
        rows.append({"config": config, "seed": seed,
                     **{f"real_{k}": v for k, v in res["real"].items()},
                     **{f"boot_{k}": v for k, v in res["boot"].items()}})

    record("BS delta", "-", {f[0]: [] for f in folds}, 0.3)

    for config, cfg in CONFIGS.items():
        for seed in SEEDS:
            nets = {name: [fit(hist, cfg, seed)] for name, hist, _, _, _ in folds}
            if config == "base":
                base_nets[seed] = nets
            record(config, seed, nets, cfg["max_adj"])
            print(f"  {config:8s} seed {seed} done  ({(time.time() - t0) / 60:.1f} min)")

    ens = {f[0]: [base_nets[s][f[0]][0] for s in SEEDS] for f in folds}
    record("ens", "all", ens, CONFIGS["base"]["max_adj"])

    tab = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    tab.round(4).to_csv(RESULTS / "validation_table.csv", index=False)
    cols = ["config", "seed", "real_mean", "real_std", "real_CVaR5", "real_worst",
            "boot_mean", "boot_CVaR5"]
    print("\nVALIDATION results (P&L per BRC on 1,000; higher is better except std)\n")
    print(tab[cols].round(2).to_string(index=False))

    # ---- selection rule ----------------------------------------------------
    base = tab[tab["config"] == "base"].set_index("seed")
    spread = {k: base[k].max() - base[k].min() for k in ("real_CVaR5", "boot_CVaR5", "real_mean")}
    tol_mean = max(spread["real_mean"], 1.0)
    print(f"\nbase seed spread: real CVaR {spread['real_CVaR5']:.2f}, "
          f"boot CVaR {spread['boot_CVaR5']:.2f}, mean {spread['real_mean']:.2f}")

    final, decisions = dict(CONFIGS["base"]), {}
    for config, param in CHANGES.items():
        c = tab[tab["config"] == config].set_index("seed")
        d_real = c["real_CVaR5"] - base["real_CVaR5"]
        d_boot = c["boot_CVaR5"] - base["boot_CVaR5"]
        d_mean = (c["real_mean"] - base["real_mean"]).mean()
        keep = bool((d_real > spread["real_CVaR5"]).all()
                    and (d_boot > spread["boot_CVaR5"]).all() and d_mean > -tol_mean)
        decisions[config] = {"keep": keep, "dCVaR_real": d_real.round(2).tolist(),
                             "dCVaR_boot": d_boot.round(2).tolist(), "dmean": round(d_mean, 2)}
        if keep:
            final[param] = CONFIGS[config][param]
    e = tab[tab["config"] == "ens"].iloc[0]
    d_real = e["real_CVaR5"] - base["real_CVaR5"].mean()
    d_boot = e["boot_CVaR5"] - base["boot_CVaR5"].mean()
    d_mean = e["real_mean"] - base["real_mean"].mean()
    keep_ens = bool(d_real > spread["real_CVaR5"] and d_boot > spread["boot_CVaR5"]
                    and d_mean > -tol_mean)
    decisions["ens"] = {"keep": keep_ens, "dCVaR_real": round(d_real, 2),
                        "dCVaR_boot": round(d_boot, 2), "dmean": round(d_mean, 2)}
    final["ensemble"] = keep_ens

    print("\nSELECTION (change kept only if it passes the pre-set rule):")
    for k, v in decisions.items():
        print(f"  {k:8s} {'KEEP' if v['keep'] else 'drop':4s}  dCVaR real {v['dCVaR_real']}  "
              f"dCVaR boot {v['dCVaR_boot']}  dmean {v['dmean']}")
    print(f"\nFINAL CONFIGURATION: {final}")
    if sum(v["keep"] for v in decisions.values()) > 1:
        print("  note: kept changes were validated one at a time, not in combination")

    # ---- diagnostics on base networks ---------------------------------------
    print("\nDIAGNOSTICS (base networks, real validation BRCs, both folds):")
    for seed in SEEDS:
        lim, cost_nn, pnl_nn = [], [], []
        for f in folds:
            p, c, l = run(base_nets[seed][f[0]], CONFIGS["base"]["max_adj"], f[2])
            pnl_nn.append(p); cost_nn.append(c); lim.append(l)
        print(f"  seed {seed}: at the +/-0.3 limit {np.mean(lim):.1%} of the time")
    p_bs = np.concatenate([run([], 0.3, f[2])[0] for f in folds])
    c_bs = np.concatenate([run([], 0.3, f[2])[1] for f in folds])
    p_nn = np.concatenate(pnl_nn); c_nn = np.concatenate(cost_nn)
    gap, cost_gap = p_nn.mean() - p_bs.mean(), c_nn.mean() - c_bs.mean()
    print(f"  average P&L gap deep - BS: {gap:.2f} per BRC, of which extra "
          f"transaction costs: {-cost_gap:.2f} (avg costs: BS {c_bs.mean():.2f}, "
          f"deep {c_nn.mean():.2f})")

    CHOICE.write_text(json.dumps({"final": final, "decisions": decisions,
                                  "lambda": LAMBDA, "cost_bp": COST_BP}, indent=2))
    print(f"\nsaved {CHOICE} and results/validation_table.csv")
    print("next: python src/tune_validated.py test   (runs ONCE)")


# ---------------------------------------------------------------------------
# Stage 2: one-shot test
# ---------------------------------------------------------------------------
def test():
    if not CHOICE.exists():
        sys.exit("Run the validation stage first: python src/tune_validated.py validate")
    if FINAL.exists():
        sys.exit(f"{FINAL} already exists: the test set has been used. Not running again.")
    final = json.loads(CHOICE.read_text())["final"]
    print(f"final configuration (chosen on validation): {final}")

    df = bt.load_inputs()
    folds = [("A", df[df.index < SPLIT], brcs_in(df, SPLIT, df.index[-1] + pd.Timedelta(days=1))),
             ("B", df[df.index >= SPLIT], brcs_in(df, df.index[0], SPLIT))]
    cfg_final = {k: final[k] for k in ("block", "max_adj", "n_iter")}
    variants = {"base": CONFIGS["base"]}
    if cfg_final != CONFIGS["base"] or final["ensemble"]:
        variants["final"] = cfg_final

    rows, t0 = [], time.time()
    pnl_bs = np.concatenate([run([], 0.3, f[2][1:])[0] for f in folds])
    rows.append({"strategy": "BS delta", "seed": "-", **summary(pnl_bs)})
    for vname, cfg in variants.items():
        nets_all = {f[0]: [] for f in folds}
        for seed in SEEDS:
            pnl = []
            for name, hist, test_brcs in folds:
                net = fit(hist, cfg, seed)
                nets_all[name].append(net)
                pnl.append(run([net], cfg["max_adj"], test_brcs[1:])[0])
            rows.append({"strategy": vname, "seed": seed, **summary(np.concatenate(pnl))})
            print(f"  {vname} seed {seed} done  ({(time.time() - t0) / 60:.1f} min)")
        if vname == "final" and final["ensemble"]:
            pnl = np.concatenate([run(nets_all[f[0]], cfg["max_adj"], f[2][1:])[0] for f in folds])
            rows.append({"strategy": "final (ensemble)", "seed": "all", **summary(pnl)})

    tab = pd.DataFrame(rows)
    print(f"\nTEST (used once): P&L per BRC on 1,000, {len(pnl_bs)} real BRCs, 5 bp\n")
    print(tab.round(2).to_string(index=False))
    RESULTS.mkdir(exist_ok=True)
    tab.round(4).to_csv(FINAL, index=False)
    print(f"\nsaved {FINAL}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "validate":
        validate()
    elif stage == "test":
        test()
    else:
        sys.exit("usage: python src/tune_validated.py validate | test")
