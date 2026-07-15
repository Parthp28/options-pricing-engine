"""Binomial tree convergence and American premium tests."""

import numpy as np
import pytest

from optionslib.models import binomial
from optionslib.models.black_scholes import price as bs_price


def test_european_call_converges_to_bs():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    bs = bs_price(S, K, T, r, sigma, q, "call")
    tree = binomial.price(
        S, K, T, r, sigma, q, "call", steps=2000, american=False, richardson=True
    )
    rel = abs(tree - bs) / bs
    assert rel < 0.0001  # 0.01 percent


def test_european_put_converges_to_bs():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.01
    bs = bs_price(S, K, T, r, sigma, q, "put")
    tree = binomial.price(
        S, K, T, r, sigma, q, "put", steps=2000, american=False, richardson=True
    )
    assert abs(tree - bs) / bs < 0.0001


def test_american_put_ge_european():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.25, 0.0
    eu = binomial.price(S, K, T, r, sigma, q, "put", steps=500, american=False)
    am = binomial.price(S, K, T, r, sigma, q, "put", steps=500, american=True)
    assert am >= eu - 1e-12


def test_american_put_strict_premium_itm():
    # ITM put with r > 0: early exercise premium should be strictly positive
    S, K, T, r, sigma, q = 90.0, 100.0, 1.0, 0.08, 0.25, 0.0
    eu = binomial.price(S, K, T, r, sigma, q, "put", steps=400, american=False)
    am = binomial.price(S, K, T, r, sigma, q, "put", steps=400, american=True)
    assert am > eu


def test_american_call_equals_european_no_dividend():
    # Why: never exercise a non dividend American call early when q=0
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    eu = binomial.price(S, K, T, r, sigma, q, "call", steps=300, american=False)
    am = binomial.price(S, K, T, r, sigma, q, "call", steps=300, american=True)
    assert am == pytest.approx(eu, abs=1e-10)


def test_hull_ch13_american_put():
    # Hull Chapter 13 Example 13.1 parameters; DerivaGem / 100 step tree gives 4.278
    px = binomial.price(
        50.0, 50.0, 5.0 / 12.0, 0.10, 0.40, 0.0, "put", steps=100, american=True
    )
    assert px == pytest.approx(4.278, abs=0.02)


def test_richardson_improves_error():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    bs = bs_price(S, K, T, r, sigma, q, "call")
    plain = binomial.price(S, K, T, r, sigma, q, "call", steps=100, american=False, richardson=False)
    rich = binomial.price(S, K, T, r, sigma, q, "call", steps=100, american=False, richardson=True)
    assert abs(rich - bs) < abs(plain - bs)


def test_binomial_vectorized_strikes():
    K = np.array([90.0, 100.0, 110.0])
    px = binomial.price(100.0, K, 0.5, 0.05, 0.2, 0.0, "call", steps=200)
    assert px.shape == (3,)
    assert np.all(np.diff(px) < 0)  # higher strike, lower call


def test_american_put_grid_premium():
    spots = [80.0, 90.0, 100.0, 110.0]
    for S in spots:
        eu = binomial.price(S, 100.0, 0.75, 0.06, 0.3, 0.0, "put", steps=200, american=False)
        am = binomial.price(S, 100.0, 0.75, 0.06, 0.3, 0.0, "put", steps=200, american=True)
        assert am >= eu - 1e-12
