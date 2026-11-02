"""
delta_hedge.py
--------------
Benchmark for the project: the bank sells the BRC, so it is SHORT a
down-and-in put (DIP). It hedges with the Black-Scholes delta, trading the
underlying every day, and pays transaction costs on each trade.

We simulate the underlying (geometric Brownian motion), run the hedge on every
path, and look at the distribution of the final hedging P&L:
    P&L = value of the hedge portfolio at T - payoff owed to the investor
A perfect hedge gives P&L = 0 on every path. Daily rebalancing and transaction
costs create a hedging error; deep hedging will try to reduce it.

The inputs (r, q, sigma) are PLACEHOLDERS until the OptionMetrics data is ready.
"""

import numpy as np
from scipy.stats import norm

from brc_pricing import down_and_in_put, daily_barrier


# ---------------------------------------------------------------------------
# Vectorised prices and deltas (work on arrays of spot prices)
# ---------------------------------------------------------------------------
def bs_put_vec(S, K, tau, r, q, sigma):
    sT = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / sT
    d2 = d1 - sT
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)


def bs_put_delta_vec(S, K, tau, r, q, sigma):
    sT = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / sT
    return -np.exp(-q * tau) * norm.cdf(-d1)


def dip_vec(S, K, H, tau, r, q, sigma):
    """Down-and-in put, closed form (H < K, S > H), on an array of spots."""
    sT = sigma * np.sqrt(tau)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    x1 = np.log(S / H) / sT + lam * sT
    y = np.log(H**2 / (S * K)) / sT + lam * sT
    y1 = np.log(H / S) / sT + lam * sT
    return (-S * np.exp(-q * tau) * norm.cdf(-x1)
            + K * np.exp(-r * tau) * norm.cdf(-x1 + sT)
            + S * np.exp(-q * tau) * (H / S) ** (2 * lam) * (norm.cdf(y) - norm.cdf(y1))
            - K * np.exp(-r * tau) * (H / S) ** (2 * lam - 2)
              * (norm.cdf(y - sT) - norm.cdf(y1 - sT)))


def dip_delta_vec(S, K, H_model, tau, r, q, sigma, knocked, eps=1e-4):
    """Delta of the DIP.
    - already knocked in  -> it is a vanilla put: vanilla put delta
    - not knocked in      -> central finite difference of the closed form
    H_model is the BGK-shifted barrier used for pricing with daily monitoring."""
    delta = bs_put_delta_vec(S, K, tau, r, q, sigma)          # knocked-in case
    alive = ~knocked
    if alive.any():
        s = np.maximum(S[alive], H_model * (1 + 2 * eps))     # stay above barrier
        up = dip_vec(s * (1 + eps), K, H_model, tau, r, q, sigma)
        dn = dip_vec(s * (1 - eps), K, H_model, tau, r, q, sigma)
        delta[alive] = (up - dn) / (2 * eps * s)
    return delta


# ---------------------------------------------------------------------------
# Hedging simulation
# ---------------------------------------------------------------------------
def simulate_delta_hedge(S0=100.0, strike=1.0, barrier=0.70, T=1.0,
                         r=0.03, q=0.015, sigma=0.20, notional=1000.0,
                         cost_bp=5.0, n_steps=252, n_paths=20_000, seed=1):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    K, H = strike * S0, barrier * S0
    H_model = daily_barrier(H, sigma, dt)          # pricing barrier (daily monitoring)
    n_puts = notional / K                          # puts embedded in one BRC

    # Premium the bank receives for the DIP at issue
    premium = n_puts * down_and_in_put(S0, K, H_model, T, r, q, sigma)

    S = np.full(n_paths, S0)
    knocked = np.zeros(n_paths, dtype=bool)
    cost = cost_bp / 10_000

    # Replicating portfolio of the DIP: shares + cash
    shares = n_puts * dip_delta_vec(S, K, H_model, T, r, q, sigma, knocked)
    cash = premium - shares * S - cost * np.abs(shares) * S
    total_costs = cost * np.abs(shares) * S

    for i in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        S = S * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        cash = cash * np.exp(r * dt) + shares * S * (np.exp(q * dt) - 1)  # interest + dividends
        knocked |= S <= H                                                 # daily barrier check
        if i < n_steps:
            tau = T - i * dt
            new_shares = n_puts * dip_delta_vec(S, K, H_model, tau, r, q, sigma, knocked)
            trade = new_shares - shares
            c = cost * np.abs(trade) * S
            cash -= trade * S + c
            total_costs += c
            shares = new_shares

    # Close the hedge at maturity (selling the shares also costs)
    c = cost * np.abs(shares) * S
    portfolio = cash + shares * S - c
    total_costs += c
    payoff = n_puts * np.where(knocked, np.maximum(K - S, 0.0), 0.0)
    pnl = portfolio - payoff
    return pnl, premium, knocked.mean(), total_costs.mean()


def summarise(pnl):
    var95 = np.percentile(pnl, 5)
    cvar95 = pnl[pnl <= var95].mean()
    return pnl.mean(), pnl.std(), var95, cvar95


if __name__ == "__main__":
    print("Short down-and-in put on 1,000 notional, Black-Scholes delta hedge, daily")
    print("Inputs: r=3%, q=1.5%, sigma=20% (PLACEHOLDERS), 20,000 simulated paths\n")
    print(f"{'cost':>7} {'premium':>8} {'knock-in':>9} {'avg cost':>9} "
          f"{'mean P&L':>9} {'std P&L':>8} {'VaR 5%':>8} {'CVaR 5%':>8}")
    for bp in [0.0, 5.0, 10.0]:
        pnl, prem, p_ki, avg_cost = simulate_delta_hedge(cost_bp=bp)
        m, s, v, cv = summarise(pnl)
        print(f"{bp:>5.0f}bp {prem:>8.2f} {p_ki:>8.1%} {avg_cost:>9.2f} "
              f"{m:>9.2f} {s:>8.2f} {v:>8.2f} {cv:>8.2f}")
