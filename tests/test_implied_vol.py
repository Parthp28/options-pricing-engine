"""Implied volatility round trip and bound tests."""

import numpy as np
import pytest

from optionslib.implied_vol import ImpliedVolError, implied_vol
from optionslib.models.black_scholes import price


def test_iv_round_trip_scalar():
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.01, 0.25
    px = price(S, K, T, r, sigma, q, "call")
    iv = implied_vol(px, S, K, T, r, q, "call")
    assert iv == pytest.approx(sigma, abs=1e-8)


def test_iv_round_trip_put():
    S, K, T, r, q, sigma = 100.0, 110.0, 0.5, 0.03, 0.0, 0.35
    px = price(S, K, T, r, sigma, q, "put")
    iv = implied_vol(px, S, K, T, r, q, "put")
    assert iv == pytest.approx(sigma, abs=1e-8)


def test_iv_round_trip_batch():
    # Keep strikes and tenors in a region with meaningful time value
    n = 200
    rng = np.random.default_rng(42)
    S = np.full(n, 100.0)
    K = rng.uniform(85, 115, n)
    T = rng.uniform(0.25, 2.0, n)
    r = 0.04
    q = 0.01
    sigma = rng.uniform(0.1, 0.6, n)
    kinds = np.where(rng.random(n) > 0.5, "call", "put")
    px = price(S, K, T, r, sigma, q, kinds)
    iv = implied_vol(px, S, K, T, r, q, kinds)
    np.testing.assert_allclose(iv, sigma, atol=1e-8)


def test_iv_deep_itm_call_brent_fallback():
    # Mild deep ITM with enough maturity that vega is small but recoverable
    S, K, T, r, q, sigma = 130.0, 100.0, 0.5, 0.02, 0.0, 0.15
    px = price(S, K, T, r, sigma, q, "call")
    iv = implied_vol(px, S, K, T, r, q, "call")
    assert iv == pytest.approx(sigma, abs=1e-8)


def test_iv_deep_otm_put():
    S, K, T, r, q, sigma = 130.0, 100.0, 0.75, 0.02, 0.0, 0.25
    px = price(S, K, T, r, sigma, q, "put")
    iv = implied_vol(px, S, K, T, r, q, "put")
    assert iv == pytest.approx(sigma, abs=1e-8)


def test_price_below_lower_bound_raises():
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    lo = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    with pytest.raises(ImpliedVolError, match="lower"):
        implied_vol(lo - 1.0, S, K, T, r, q, "call")


def test_price_above_upper_bound_raises():
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    hi = S * np.exp(-q * T)
    with pytest.raises(ImpliedVolError, match="upper"):
        implied_vol(hi + 1.0, S, K, T, r, q, "call")


def test_put_above_upper_bound_raises():
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    hi = K * np.exp(-r * T)
    with pytest.raises(ImpliedVolError, match="upper"):
        implied_vol(hi + 0.5, S, K, T, r, q, "put")


def test_iv_vector_mixed_kinds():
    S = np.array([100.0, 100.0, 100.0])
    K = np.array([90.0, 100.0, 110.0])
    T = 0.5
    r = 0.05
    q = 0.0
    sigma = np.array([0.2, 0.25, 0.3])
    kind = np.array(["call", "put", "call"])
    px = price(S, K, T, r, sigma, q, kind)
    iv = implied_vol(px, S, K, T, r, q, kind)
    np.testing.assert_allclose(iv, sigma, atol=1e-8)
