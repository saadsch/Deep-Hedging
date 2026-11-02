"""
pricing.py - Black-Scholes prices and deltas. YOU write the bodies.

This file is a skeleton on purpose: pricing and delta are exactly what an
interviewer asks you to derive or code, so write them yourself. Check your
work with:

    python -m pytest tests/test_pricing.py -v

All rates, yields and volatilities are decimals per year (0.02, not 2 %).
Time T is in years. Notation: S spot, K strike, H barrier, r risk-free rate,
q dividend yield, sigma volatility.

Reference: J. Hull, "Options, Futures, and Other Derivatives", chapter on
exotic options, section "Barrier options" (down-and-in put, case H < K).
Cite the edition you use in the paper.
"""

import numpy as np
from scipy.stats import norm

def _d1_d2(S, K, T, r, q, sigma):
    """The two Black-Scholes terms, shared by the price and the delta."""
    vol = sigma * np.sqrt(T)                                   # total uncertainty, σ√T
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol  # distance to strike, in std devs
    d2 = d1 - vol
    return d1, d2



def bs_put(S, K, T, r, q, sigma):
    """European put = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    receive = K * np.exp(-r * T) * norm.cdf(-d2)   # strike received, today's money, x prob. of exercise
    give = S * np.exp(-q * T) * norm.cdf(-d1)      # index handed over, net of dividends
    return receive - give



def bs_put_delta(S, K, T, r, q, sigma):
    """Change in the put price for a 1-point move in the index: -e^(-qT) N(-d1)."""
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return -np.exp(-q * T) * norm.cdf(-d1)


def down_in_put(S, K, H, T, r, q, sigma):
    """Down-and-in put, barrier H < K, monitored continuously (closed form, Hull).

    Two cases:
      - S <= H: the barrier is already hit, so it is a plain put -> return bs_put(...)
      - S > H : Hull's formula with lambda = (r - q + sigma^2/2) / sigma^2 and the
                terms x1, y, y1. Write it carefully; the tests catch most typos.

    Interview point: the product is monitored DAILY on closing prices, not
    continuously. Broadie, Glasserman & Kou (1997) correct for this by moving
    the barrier: H_daily = H * exp(-0.5826 * sigma * sqrt(1/252)) for a down barrier.
    Decide in the paper whether you use it, and say why.
    """
    raise NotImplementedError("write down_in_put")


def down_in_put_delta(S, K, H, T, r, q, sigma, bump=1e-4):
    """Delta of the down-and-in put.

    Start with a central finite difference:
        (price(S * (1 + bump)) - price(S * (1 - bump))) / (2 * S * bump)
    It is enough for hedging. An analytic delta is a possible extension.
    """
    raise NotImplementedError("write down_in_put_delta")
