"""
Tests for src/pricing.py. Run from the repository root:

    python -m pytest tests/test_pricing.py -v

Reference values were computed independently with Hull's closed form and
checked against a Monte Carlo simulation (100,000 paths; the Monte Carlo
value is slightly lower because it monitors the barrier on a finite grid).

Parameters used: S = 100, K = 100, T = 1, r = 2 %, q = 1.5 %, sigma = 20 %.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pricing import bs_put, bs_put_delta, down_in_put, down_in_put_delta  # noqa: E402

P = dict(S=100.0, K=100.0, T=1.0, r=0.02, q=0.015, sigma=0.20)


def test_bs_put_reference_value():
    assert bs_put(**P) == pytest.approx(7.5842, abs=1e-3)


def test_bs_put_delta_is_between_minus_one_and_zero():
    d = bs_put_delta(**P)
    assert -1.0 < d < 0.0


@pytest.mark.parametrize("H, expected", [(60, 0.5036), (70, 2.4926), (80, 5.6692)])
def test_down_in_put_reference_values(H, expected):
    assert down_in_put(H=H, **P) == pytest.approx(expected, abs=1e-3)


def test_down_in_put_is_cheaper_than_plain_put():
    # It pays the same as a put, but only if the barrier is hit.
    assert down_in_put(H=70, **P) < bs_put(**P)


def test_down_in_put_rises_with_the_barrier():
    # A higher barrier is easier to hit, so the option is worth more.
    values = [down_in_put(H=h, **P) for h in (50, 60, 70, 80, 90)]
    assert values == sorted(values)


def test_barrier_already_hit_gives_plain_put():
    q = dict(P, S=65.0)
    assert down_in_put(H=70, **q) == pytest.approx(bs_put(**q), abs=1e-8)


def test_down_in_put_delta_reference_value():
    assert down_in_put_delta(H=70, **P) == pytest.approx(-0.2626, abs=2e-3)
