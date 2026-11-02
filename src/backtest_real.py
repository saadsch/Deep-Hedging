"""
backtest_real.py
----------------
The core experiment: hedge real BRCs on real S&P 500 paths.

  - One BRC is issued on the first trading day of every month (Dec 1999 onward),
    1-year maturity (252 trading days), strike 100 %, barrier 70 % observed daily.
  - The bank receives the Black-Scholes price of the down-and-in put (DIP),
    computed with the 1-year at-the-money implied vol of the issue date.
  - It hedges every day by trading the S&P 500, with that day's r, q and
    implied vol, paying transaction costs. Cash earns r, shares earn q.
  - Two hedging strategies are compared on the SAME real paths:
        BS delta    Black-Scholes delta of the DIP
        deep hedge  BS delta + a correction learned by a neural network that was
                    trained ONLY on simulated paths (parameters drawn from the
                    real history of r, q, sigma); it never sees the real paths.

Result per BRC: final hedging P&L (positive = the bank gained) on 1,000 notional.

Inputs : data/clean/brc_inputs.parquet (or .csv) from real_inputs.py
Outputs: results/backtest_real.csv, figures/backtest_real.png
Run from the project root:   python src/backtest_real.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from deep_hedge import MLP

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
STRIKE, BARRIER = 1.00, 0.70
N = 252                        # trading days to maturity
DT = 1 / 252
NOTIONAL = 1000.0
VOL_COL = "sigma_atm"          # vol used to price and hedge ("sigma_70" = skew test)
COSTS_BP = [0.0, 5.0, 10.0]
ALPHA = 0.05                   # CVaR level

N_TRAIN, BATCH, N_ITER, LR = 10_000, 500, 300, 1e-3
HIDDEN = 16                    # correction range +/-0.3 share/put, set in deep_hedge.py
SEED = 0

CLEAN, RESULTS, FIG = Path("data/clean"), Path("results"), Path("figures")


# ---------------------------------------------------------------------------
# Vectorised Black-Scholes pieces (all parameters may be arrays)
# ---------------------------------------------------------------------------
def dip_price(S, K, H, tau, r, q, s):
    """Down-and-in put, continuous monitoring, H < K < ... valid for S > H."""
    sT = s * np.sqrt(tau)
    lam = (r - q + 0.5 * s**2) / s**2
    x1 = np.log(S / H) / sT + lam * sT
    y = np.log(H**2 / (S * K)) / sT + lam * sT
    y1 = np.log(H / S) / sT + lam * sT
    return (-S * np.exp(-q * tau) * norm.cdf(-x1)
            + K * np.exp(-r * tau) * norm.cdf(-x1 + sT)
            + S * np.exp(-q * tau) * (H / S) ** (2 * lam) * (norm.cdf(y) - norm.cdf(y1))
            - K * np.exp(-r * tau) * (H / S) ** (2 * lam - 2)
              * (norm.cdf(y - sT) - norm.cdf(y1 - sT)))


def bgk(H, s):
    """Broadie-Glasserman-Kou shift: daily-monitored barrier ~ continuous one."""
    return H * np.exp(-0.5826 * s * np.sqrt(DT))


def dip_delta(S, K, H, tau, r, q, s, knocked, eps=1e-4):
    """Delta per put. Knocked in -> vanilla put delta; else finite difference."""
    sT = s * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * s**2) * tau) / sT
    delta = -np.exp(-q * tau) * norm.cdf(-d1)
    Hm = bgk(H, s)
    a = ~knocked
    if a.any():
        Sa = np.maximum(S[a], Hm[a] * (1 + 2 * eps))
        args = (K[a], Hm[a], tau, r[a], q[a], s[a])
        delta[a] = (dip_price(Sa * (1 + eps), *args)
                    - dip_price(Sa * (1 - eps), *args)) / (2 * eps * Sa)
    return delta


# ---------------------------------------------------------------------------
# P&L of a hedge and its gradient (per-path, per-day rates allowed)
# ---------------------------------------------------------------------------
def pnl_and_grad(pos, S, r, q, premium, payoff, cost):
    """pos (n,N) shares held over each day; S (n,N+1); r,q (n,N) daily inputs.
    All cash flows are valued at maturity."""
    n = len(S)
    # value at T of 1 unit of cash at t_i, i = 0..N
    G = np.exp(np.concatenate([np.cumsum((r * DT)[:, ::-1], axis=1)[:, ::-1],
                               np.zeros((n, 1))], axis=1))
    gain = (S[:, 1:] * np.exp(q * DT) - S[:, :-1] * np.exp(r * DT)) * G[:, 1:]
    trades = np.diff(np.concatenate([np.zeros((n, 1)), pos, np.zeros((n, 1))], 1), axis=1)
    px = S * G                                                  # trade at t_k, valued at T
    costs = cost * np.abs(trades) * px
    pnl = premium * G[:, 0] + (pos * gain).sum(1) - costs.sum(1) - payoff
    sg = np.sign(trades)
    grad = gain - cost * (sg[:, :-1] * px[:, :-1] - sg[:, 1:] * px[:, 1:])
    return pnl, grad, costs.sum(1)


def features(S, S0, knocked, bs_delta):
    n = len(S)
    tau = np.broadcast_to(1 - np.arange(N) / N, (n, N))
    return np.stack([np.log(S[:, :-1] / S0[:, None]), tau,
                     knocked[:, :-1].astype(float), bs_delta], axis=-1)


def positions(net, X, n_puts):
    """Deep-hedge position = n_puts * (BS delta + learned correction)."""
    return n_puts * net.forward(X)


# ---------------------------------------------------------------------------
# Real BRCs
# ---------------------------------------------------------------------------
def load_inputs():
    pq, csv = CLEAN / "brc_inputs.parquet", CLEAN / "brc_inputs.csv"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(csv, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[df["spot"].notna()]
    return df[["spot", "r", "q", VOL_COL]].ffill().dropna()


def real_brcs(df):
    """One BRC per month: real price path + the daily inputs seen by the hedger."""
    dates = df.index
    first_days = df.groupby([dates.year, dates.month]).head(1).index
    rows = []
    for d in first_days:
        i = dates.get_loc(d)
        if i + N < len(dates):
            rows.append(i)
    idx = np.array(rows)[:, None] + np.arange(N + 1)[None, :]
    S = df["spot"].to_numpy()[idx]
    r = df["r"].to_numpy()[idx][:, :-1]
    q = df["q"].to_numpy()[idx][:, :-1]
    s = df[VOL_COL].to_numpy()[idx][:, :-1]
    return dates[idx[:, 0]], dates[idx[:, -1]], S, r, q, s


def bs_inputs(S, r, q, s):
    """Knock-in flags, BS deltas (per put), premium and payoff on given paths."""
    n = len(S)
    S0 = S[:, 0]
    K, H = STRIKE * S0, BARRIER * S0
    knocked = np.maximum.accumulate(S <= H[:, None], axis=1)
    delta = np.zeros((n, N))
    for i in range(N):
        delta[:, i] = dip_delta(S[:, i], K, H, (N - i) * DT,
                                r[:, i], q[:, i], s[:, i], knocked[:, i].copy())
    n_puts = NOTIONAL / K
    premium = n_puts * dip_price(S0, K, bgk(H, s[:, 0]), 1.0, r[:, 0], q[:, 0], s[:, 0])
    payoff = n_puts * np.where(knocked[:, -1], np.maximum(K - S[:, -1], 0), 0)
    return S0, knocked, delta, n_puts, premium, payoff


# ---------------------------------------------------------------------------
# Training on simulated paths with parameters drawn from history
# ---------------------------------------------------------------------------
def simulated_training_set(df, n, seed):
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(df), n)
    r0 = df["r"].to_numpy()[pick]
    q0 = df["q"].to_numpy()[pick]
    s0 = df[VOL_COL].to_numpy()[pick]
    z = rng.standard_normal((n, N))
    logS = np.cumsum((r0 - q0 - 0.5 * s0**2)[:, None] * DT
                     + s0[:, None] * np.sqrt(DT) * z, axis=1)
    S = 100.0 * np.exp(np.concatenate([np.zeros((n, 1)), logS], axis=1))
    rep = lambda x: np.repeat(x[:, None], N, axis=1)
    return S, rep(r0), rep(q0), rep(s0)


def train(sim, cost, seed):
    S, r, q, s = sim
    S0, knocked, delta, n_puts, premium, payoff = bs_inputs(S, r, q, s)
    X = features(S, S0, knocked, delta)
    net, w, wm, wv = MLP(X.shape[-1], HIDDEN, seed), 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    for it in range(1, N_ITER + 1):
        b = rng.choice(len(S), BATCH, replace=False)
        theta = net.forward(X[b])
        pos = n_puts[b, None] * theta
        pnl, g, _ = pnl_and_grad(pos, S[b], r[b], q[b], premium[b], payoff[b], cost)
        loss = -pnl
        tail = (loss > w).astype(float)
        d_theta = -(tail / (ALPHA * BATCH))[:, None] * g * n_puts[b, None]
        net.adam_step(net.backward(d_theta), LR)
        gw = 1 - tail.mean() / ALPHA
        wm, wv = 0.9 * wm + 0.1 * gw, 0.999 * wv + 0.001 * gw**2
        w -= 0.5 * (wm / (1 - 0.9**it)) / (np.sqrt(wv / (1 - 0.999**it)) + 1e-8)
    return net


def stats(p):
    v = np.percentile(p, 100 * ALPHA)
    return {"mean": p.mean(), "std": p.std(), "VaR5": v, "CVaR5": p[p <= v].mean(),
            "worst": p.min()}


# ---------------------------------------------------------------------------
def main():
    df = load_inputs()
    start, end, S, r, q, s = real_brcs(df)
    S0, knocked, delta, n_puts, premium, payoff = bs_inputs(S, r, q, s)
    X = features(S, S0, knocked, delta)
    print(f"{len(S)} real BRCs, issued {start[0].date()} -> {start[-1].date()} "
          f"(pricing/hedging vol: {VOL_COL})")
    print(f"knocked in: {knocked[:, -1].mean():.1%} of BRCs, "
          f"average premium {premium.mean():.2f} per 1,000\n")

    sim = simulated_training_set(df, N_TRAIN, SEED)
    out = pd.DataFrame({"issue": start, "maturity": end, "S0": S0,
                        "premium": premium, "knocked_in": knocked[:, -1],
                        "payoff": payoff})
    table = []
    for bp in COSTS_BP:
        c = bp / 10_000
        print(f"cost {bp:.0f} bp: training the network on simulated paths...")
        net = train(sim, c, SEED)
        pnl_bs, _, cost_bs = pnl_and_grad(n_puts[:, None] * delta, S, r, q, premium, payoff, c)
        pnl_nn, _, cost_nn = pnl_and_grad(positions(net, X, n_puts[:, None]), S, r, q,
                                          premium, payoff, c)
        out[f"pnl_bs_{bp:.0f}bp"], out[f"pnl_deep_{bp:.0f}bp"] = pnl_bs, pnl_nn
        for name, p, cst in [("BS delta", pnl_bs, cost_bs), ("deep hedge", pnl_nn, cost_nn)]:
            table.append({"cost_bp": bp, "strategy": name, **stats(p),
                          "avg_costs": cst.mean()})

    res = pd.DataFrame(table).set_index(["cost_bp", "strategy"])
    print("\nHedging P&L per BRC on 1,000 notional (REAL S&P 500 paths):")
    print(res.round(2).to_string())

    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "backtest_real.csv", index=False)
    res.round(4).to_csv(RESULTS / "backtest_real_summary.csv")
    print("\nsaved results/backtest_real.csv and results/backtest_real_summary.csv")

    worst = out.nsmallest(5, "pnl_bs_5bp")[["issue", "knocked_in", "premium",
                                            "payoff", "pnl_bs_5bp", "pnl_deep_5bp"]]
    print("\nFive worst BRCs for the BS delta hedge (5 bp):")
    print(worst.to_string(index=False, float_format="%.2f"))

    FIG.mkdir(exist_ok=True)
    make_figure(out, FIG / "backtest_real.png")
    print("saved figures/backtest_real.png")


def make_figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2, grid, surface = "#52514e", "#e4e3df", "#fcfcfb"
    c1, c2 = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2})
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=surface, constrained_layout=True)
    ax.set_facecolor(surface)
    ax.grid(axis="y", color=grid, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.axhline(0, color=ink2, lw=0.8)
    ki = out["knocked_in"].to_numpy()
    for d in out["issue"][ki]:
        ax.axvspan(d - pd.Timedelta(days=14), d + pd.Timedelta(days=14),
                   color="#ece9e2", lw=0)
    ax.plot(out["issue"], out["pnl_bs_5bp"], color=c1, lw=1.2, marker="o", ms=2.5,
            label="Black-Scholes delta")
    ax.plot(out["issue"], out["pnl_deep_5bp"], color=c2, lw=1.2, marker="o", ms=2.5,
            label="deep hedge")
    ax.set_title("Hedging P&L per BRC by issue date, 5 bp costs (per 1,000; "
                 "grey = barrier hit)", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
