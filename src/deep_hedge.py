"""
deep_hedge.py
-------------
First version of the deep hedging model (Buehler, Gonon, Teichmann & Wood, 2019),
written in pure numpy so it runs anywhere with nothing to install.

Set-up (same as delta_hedge.py):
  - the bank is SHORT the down-and-in put (DIP) embedded in the BRC
  - it trades the underlying once a day and pays transaction costs
  - final P&L = hedge portfolio at T - payoff owed

The network decides, every day, how many shares to hold.
  inputs : log(S/S0), time to maturity, knocked-in flag, Black-Scholes delta
  output : position per embedded put = BS delta + learned correction
           (the correction starts at 0, so training starts from the BS hedge)
It is trained to minimise the CVaR (expected shortfall) at 95 % of the
hedging LOSS, i.e. the average of the 5 % worst outcomes, using the
Rockafellar-Uryasev formulation:
    CVaR = min_w  w + E[ max(loss - w, 0) ] / 0.05

Version 1 limitation: the network does not see its own previous position,
so it cannot learn a full "no-trade band". It can still learn a smoother,
cheaper hedge than Black-Scholes. Adding the previous position is the next step.

Inputs r, q, sigma are PLACEHOLDERS until the OptionMetrics data is ready.
"""

import numpy as np

from brc_pricing import down_and_in_put, daily_barrier
from delta_hedge import dip_delta_vec

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
S0, STRIKE, BARRIER, T = 100.0, 1.0, 0.70, 1.0
R, Q, SIGMA = 0.03, 0.015, 0.20           # PLACEHOLDERS
NOTIONAL = 1000.0
N_STEPS = 252
ALPHA = 0.05                               # CVaR level: average of the 5 % worst
HIDDEN = 16
N_TRAIN, N_TEST = 10_000, 10_000
BATCH = 500
N_ITER = 300
LR = 1e-3
MAX_ADJ = 0.3        # the network can move the position by at most 0.3 share per put

K, H = STRIKE * S0, BARRIER * S0
DT = T / N_STEPS
H_MODEL = daily_barrier(H, SIGMA, DT)
N_PUTS = NOTIONAL / K
PREMIUM = N_PUTS * down_and_in_put(S0, K, H_MODEL, T, R, Q, SIGMA)


# ---------------------------------------------------------------------------
# Simulated market + features
# ---------------------------------------------------------------------------
def simulate(n_paths, seed):
    """Returns prices S (n, N+1), features X (n, N, 4), BS delta (n, N), payoff."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, N_STEPS))
    logret = (R - Q - 0.5 * SIGMA**2) * DT + SIGMA * np.sqrt(DT) * z
    S = S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)),
                                    np.cumsum(logret, axis=1)], axis=1))
    knocked = np.maximum.accumulate(S <= H, axis=1)           # True once barrier hit

    X = np.zeros((n_paths, N_STEPS, 4))
    bs_delta = np.zeros((n_paths, N_STEPS))
    for i in range(N_STEPS):
        tau = T - i * DT
        bs_delta[:, i] = dip_delta_vec(S[:, i], K, H_MODEL, tau, R, Q, SIGMA,
                                       knocked[:, i].copy())
        X[:, i, 0] = np.log(S[:, i] / S0)
        X[:, i, 1] = tau / T
        X[:, i, 2] = knocked[:, i]
        X[:, i, 3] = bs_delta[:, i]
    payoff = N_PUTS * np.where(knocked[:, -1], np.maximum(K - S[:, -1], 0.0), 0.0)
    return S, X, bs_delta, payoff


def pnl_and_grad(theta, S, payoff, cost):
    """Final P&L for positions theta (n, N) in shares PER PUT, and dP&L/dtheta.
    Cash earns r, shares earn the dividend yield q, costs paid on every trade,
    hedge closed at T. All amounts valued at T."""
    pos = N_PUTS * theta
    growth = np.exp(R * (T - DT * np.arange(1, N_STEPS + 1)))          # value at T of 1 at t_{i+1}
    grow_i = np.exp(R * (T - DT * np.arange(N_STEPS)))                 # value at T of 1 at t_i
    gain_unit = (S[:, 1:] * np.exp(Q * DT) - S[:, :-1] * np.exp(R * DT)) * growth
    trades = np.diff(np.concatenate([np.zeros((len(S), 1)), pos,
                                     np.zeros((len(S), 1))], axis=1), axis=1)  # N+1 trades
    trade_px = np.concatenate([S[:, :-1] * grow_i, S[:, -1:]], axis=1)
    costs = cost * np.abs(trades) * trade_px

    pnl = PREMIUM * np.exp(R * T) + (pos * gain_unit).sum(1) - costs.sum(1) - payoff

    sgn = np.sign(trades)
    d_cost = cost * (sgn[:, :-1] * trade_px[:, :-1] - sgn[:, 1:] * trade_px[:, 1:])
    grad = N_PUTS * (gain_unit - d_cost)
    return pnl, grad


# ---------------------------------------------------------------------------
# Small neural network (2 hidden layers, tanh), hand-written gradients + Adam
# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, n_in, n_hidden, seed=0):
        rng = np.random.default_rng(seed)
        self.p = {
            "W1": rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_hidden)), "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_hidden)), "b2": np.zeros(n_hidden),
            "W3": np.zeros((n_hidden, 1)), "b3": np.zeros(1),   # start at BS delta
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X):
        p = self.p
        h1 = np.tanh(X @ p["W1"] + p["b1"])
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])
        z = (h2 @ p["W3"] + p["b3"])[..., 0]
        adj = MAX_ADJ * np.tanh(z)
        # Position = Black-Scholes delta (feature 3) + a learned correction.
        # At initialisation the correction is ~0, so the network starts AT the
        # Black-Scholes hedge and only learns how to improve on it.
        theta = X[..., 3] + adj
        self.cache = (X, h1, h2, adj)
        return theta

    def backward(self, d_theta):
        X, h1, h2, adj = self.cache
        p = self.p
        dz = d_theta * MAX_ADJ * (1 - (adj / MAX_ADJ) ** 2)   # derivative of tanh
        Xf, h1f, h2f, dzf = (a.reshape(-1, a.shape[-1]) for a in (X, h1, h2, dz[..., None]))
        g = {"W3": h2f.T @ dzf, "b3": dzf.sum(0)}
        dh2 = (dzf @ p["W3"].T) * (1 - h2f**2)
        g["W2"], g["b2"] = h1f.T @ dh2, dh2.sum(0)
        dh1 = (dh2 @ p["W2"].T) * (1 - h1f**2)
        g["W1"], g["b1"] = Xf.T @ dh1, dh1.sum(0)
        return g

    def adam_step(self, g, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1**self.t)
            vh = self.v[k] / (1 - b2**self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)


def train(cost_bp, data, seed=0, verbose=True):
    S, X, _, payoff = data
    cost = cost_bp / 10_000
    net, w = MLP(X.shape[-1], HIDDEN, seed), 0.0
    w_m = w_v = 0.0
    rng = np.random.default_rng(seed)
    for it in range(1, N_ITER + 1):
        idx = rng.choice(len(S), BATCH, replace=False)
        theta = net.forward(X[idx])
        pnl, dpnl = pnl_and_grad(theta, S[idx], payoff[idx], cost)
        loss = -pnl
        tail = (loss > w).astype(float)
        cvar = w + (np.maximum(loss - w, 0)).mean() / ALPHA
        # gradients of CVaR objective
        d_loss = tail / (ALPHA * BATCH)                 # dCVaR / dloss_j
        d_theta = -d_loss[:, None] * dpnl              # loss = -pnl
        net.adam_step(net.backward(d_theta), LR)
        g_w = 1 - tail.mean() / ALPHA                   # dCVaR / dw, Adam on w too
        w_m = 0.9 * w_m + 0.1 * g_w
        w_v = 0.999 * w_v + 0.001 * g_w**2
        w -= 0.5 * (w_m / (1 - 0.9**it)) / (np.sqrt(w_v / (1 - 0.999**it)) + 1e-8)
        if verbose and it % 100 == 0:
            print(f"    iter {it:4d}  CVaR(batch) = {cvar:7.2f}")
    return net


def stats(pnl):
    var = np.percentile(pnl, 100 * ALPHA)
    return pnl.mean(), pnl.std(), var, pnl[pnl <= var].mean()


if __name__ == "__main__":
    print("Simulating training and test paths (placeholder inputs)...")
    train_data = simulate(N_TRAIN, seed=10)
    test_data = simulate(N_TEST, seed=20)            # never seen during training
    S_te, X_te, delta_te, pay_te = test_data

    print(f"\nShort DIP on {NOTIONAL:,.0f} notional, premium {PREMIUM:.2f}, "
          f"{N_TEST:,} TEST paths")
    header = f"{'':>14} {'mean P&L':>9} {'std':>7} {'VaR 5%':>8} {'CVaR 5%':>8}"
    for bp in [0.0, 5.0, 10.0]:
        print(f"\n--- transaction costs {bp:.0f} bp ---  training:")
        net = train(bp, train_data)
        pnl_bs, _ = pnl_and_grad(delta_te, S_te, pay_te, bp / 10_000)
        pnl_nn, _ = pnl_and_grad(net.forward(X_te), S_te, pay_te, bp / 10_000)
        print(header)
        for name, p in [("BS delta", pnl_bs), ("deep hedge", pnl_nn)]:
            m, s, v, cv = stats(p)
            print(f"{name:>14} {m:9.2f} {s:7.2f} {v:8.2f} {cv:8.2f}")
