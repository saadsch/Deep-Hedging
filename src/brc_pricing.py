"""
brc_pricing.py
--------------
Black-Scholes pricing of the Barrier Reverse Convertible (BRC) on SPX.

BRC = zero-coupon bond (notional + coupon)  MINUS  N/(K*S0) down-and-in puts.
The bank is short the down-and-in put; this is the position we hedge later.

Contents
  1. down_and_in_put()      closed form (Reiner-Rubinstein 1991, as in Hull),
                            continuous barrier monitoring
  2. daily_barrier()        Broadie-Glasserman-Kou (1997) correction to
                            approximate DAILY monitoring
  3. brc_value(), par_coupon()
  4. mc_down_and_in_put()   Monte Carlo check with daily monitoring

The inputs at the bottom (r, q, sigma) are PLACEHOLDERS for testing only.
They will be replaced by values from the OptionMetrics data
(zero curve, index dividend yield, volatility surface).
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# 1. Down-and-in put, continuous monitoring, barrier H below strike K
# ---------------------------------------------------------------------------
def down_and_in_put(S, K, H, T, r, q, sigma):
    """Price of a down-and-in put (H < K), continuous barrier monitoring."""
    if H >= K:
        raise ValueError("Formula valid for H < K only")
    if S <= H:                                   # already knocked in -> vanilla put
        return bs_put(S, K, T, r, q, sigma)

    sT = sigma * np.sqrt(T)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    x1 = np.log(S / H) / sT + lam * sT
    y = np.log(H**2 / (S * K)) / sT + lam * sT
    y1 = np.log(H / S) / sT + lam * sT

    return (-S * np.exp(-q * T) * norm.cdf(-x1)
            + K * np.exp(-r * T) * norm.cdf(-x1 + sT)
            + S * np.exp(-q * T) * (H / S) ** (2 * lam) * (norm.cdf(y) - norm.cdf(y1))
            - K * np.exp(-r * T) * (H / S) ** (2 * lam - 2)
              * (norm.cdf(y - sT) - norm.cdf(y1 - sT)))


def bs_put(S, K, T, r, q, sigma):
    """Vanilla European put (Black-Scholes with dividend yield q)."""
    sT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sT
    d2 = d1 - sT
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# 2. Daily monitoring: shift the barrier (Broadie-Glasserman-Kou)
# ---------------------------------------------------------------------------
def daily_barrier(H, sigma, dt=1 / 252):
    """Down barrier observed every dt behaves like a continuous barrier
    shifted down by exp(-0.5826 * sigma * sqrt(dt))."""
    return H * np.exp(-0.5826 * sigma * np.sqrt(dt))


# ---------------------------------------------------------------------------
# 3. BRC value and par coupon
# ---------------------------------------------------------------------------
def brc_value(coupon, S0, strike, barrier, T, r, q, sigma, notional=1000.0,
              monitoring="daily"):
    """Value at issue of a BRC paying `coupon` (in currency) at maturity."""
    K, H = strike * S0, barrier * S0
    if monitoring == "daily":
        H = daily_barrier(H, sigma)
    dip = down_and_in_put(S0, K, H, T, r, q, sigma)
    return (notional + coupon) * np.exp(-r * T) - notional / K * dip


def par_coupon(S0, strike, barrier, T, r, q, sigma, notional=1000.0,
               monitoring="daily"):
    """Coupon (in currency, paid at T) such that the BRC is worth par."""
    K, H = strike * S0, barrier * S0
    if monitoring == "daily":
        H = daily_barrier(H, sigma)
    dip = down_and_in_put(S0, K, H, T, r, q, sigma)
    return notional * (1 - np.exp(-r * T)) * np.exp(r * T) \
        + notional / K * dip * np.exp(r * T)


# ---------------------------------------------------------------------------
# 4. Monte Carlo check (daily monitoring, geometric Brownian motion)
# ---------------------------------------------------------------------------
def mc_down_and_in_put(S0, K, H, T, r, q, sigma, n_steps=252,
                       n_paths=200_000, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_paths = np.cumsum((r - q - 0.5 * sigma**2) * dt
                          + sigma * np.sqrt(dt) * z, axis=1)
    paths = S0 * np.exp(log_paths)
    hit = paths.min(axis=1) <= H
    payoff = np.where(hit, np.maximum(K - paths[:, -1], 0.0), 0.0)
    disc = np.exp(-r * T) * payoff
    return disc.mean(), disc.std() / np.sqrt(n_paths)


# ---------------------------------------------------------------------------
# Test with placeholder inputs
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    S0, strike, barrier, T = 100.0, 1.00, 0.70, 1.0
    r, q, sigma = 0.03, 0.015, 0.2          # PLACEHOLDERS, not from data
    K, H = strike * S0, barrier * S0

    cont = down_and_in_put(S0, K, H, T, r, q, sigma)
    daily = down_and_in_put(S0, K, daily_barrier(H, sigma), T, r, q, sigma)
    mc, se = mc_down_and_in_put(S0, K, H, T, r, q, sigma)

    print("Down-and-in put (strike 100, barrier 70, 1y)")
    print(f"  closed form, continuous barrier : {cont:.4f}")
    print(f"  closed form, daily (BGK shift)  : {daily:.4f}")
    print(f"  Monte Carlo, daily              : {mc:.4f}  (+/- {1.96 * se:.4f})")
    print(f"  vanilla put (upper bound)       : {bs_put(S0, K, T, r, q, sigma):.4f}")

    c = par_coupon(S0, strike, barrier, T, r, q, sigma)
    print(f"\nPar coupon on 1,000: {c:.2f}  ({c / 10:.2f} % p.a.)")
    print(f"Check BRC value at par coupon: "
          f"{brc_value(c, S0, strike, barrier, T, r, q, sigma):.4f}")
