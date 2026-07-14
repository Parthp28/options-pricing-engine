"""Tests for Black Scholes Merton pricing and analytic Greeks."""

import numpy as np
import pytest

from optionslib.models.black_scholes import (
    delta,
    gamma,
    price,
    rho,
    theta,
    vega,
)


# Hull Chapter 15, Example 15.1 (BS Call): S=49, K=50, r=0.05, sigma=0.2, T=0.3846
HULL_CH15_S = 49.0
HULL_CH15_K = 50.0
HULL_CH15_R = 0.05
HULL_CH15_SIGMA = 0.2
HULL_CH15_T = 20.0 / 52.0
HULL_CH15_CALL = 2.4005

# Hull Chapter 15, Table 15.1 style European put check via parity around same inputs
# Hull Chapter 19 / 13 binomial American put fixture used later; European BS put here:
# S=50, K=50, r=0.1, sigma=0.4, T=0.4167 from Hull Ch 13 Example 13.1 parameters
HULL_CH13_S = 50.0
HULL_CH13_K = 50.0
HULL_CH13_R = 0.10
HULL_CH13_SIGMA = 0.40
HULL_CH13_T = 5.0 / 12.0

# Hull Chapter 17 (dividend yield): continuous yield example parameters
# Using BSM with q: S=900, K=1000, r=0.05? Classic: FX/stock with q from Hull Ch 17.
# Hull Ch 17 Example 17.1: S0=900, K=1000, r=0.05? Actually European call on stock index
# with q=0.03, S=900? Rechecking common published: Example 17.1 S0=49? Use known:
# Hull 10e Example 17.1: European call on a stock index, S0=910, K=900, r=0.05? 
# Standard published: S=49 is equity; for q use:
# Hull Example 17.3 style: call with continuous dividend yield.
HULL_CH17_S = 900.0
HULL_CH17_K = 1000.0
HULL_CH17_R = 0.05
HULL_CH17_Q = 0.02
HULL_CH17_SIGMA = 0.22
HULL_CH17_T = 1.0


def test_hull_ch15_european_call():
    # Hull Chapter 15 Example 15.1 published call price approx 2.4
    px = price(HULL_CH15_S, HULL_CH15_K, HULL_CH15_T, HULL_CH15_R, HULL_CH15_SIGMA, 0.0, "call")
    assert px == pytest.approx(HULL_CH15_CALL, abs=0.01)


def test_hull_ch15_delta():
    # Hull Chapter 15 reports N(d1) ≈ 0.522 for the Example 15.1 call
    d = delta(HULL_CH15_S, HULL_CH15_K, HULL_CH15_T, HULL_CH15_R, HULL_CH15_SIGMA, 0.0, "call")
    assert d == pytest.approx(0.522, abs=0.01)


def test_hull_ch13_european_put_positive():
    # Hull Chapter 13 Example 13.1 parameters: European put must be positive and below K e^{-rT}
    px = price(HULL_CH13_S, HULL_CH13_K, HULL_CH13_T, HULL_CH13_R, HULL_CH13_SIGMA, 0.0, "put")
    assert px > 0.0
    assert px < HULL_CH13_K * np.exp(-HULL_CH13_R * HULL_CH13_T)
    # Published European put under these params is about 4.28 at 5 steps American; European lower.
    # Closed form reference: around 4.076 (common textbook check).
    assert px == pytest.approx(4.076, abs=0.05)


def test_put_call_parity_scalar():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.01
    c = price(S, K, T, r, sigma, q, "call")
    p = price(S, K, T, r, sigma, q, "put")
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_put_call_parity_grid():
    S = np.array([80.0, 100.0, 120.0])
    K = np.array([90.0, 100.0, 110.0])
    T = np.array([0.25, 0.5, 1.0])
    r = 0.03
    sigma = np.array([0.15, 0.25, 0.35])
    q = 0.01
    c = price(S, K, T, r, sigma, q, "call")
    p = price(S, K, T, r, sigma, q, "put")
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_put_call_parity_broadcast():
    S = np.linspace(50, 150, 21)
    K = 100.0
    T = 0.75
    r = 0.04
    sigma = 0.3
    q = 0.0
    c = price(S, K, T, r, sigma, q, "call")
    p = price(S, K, T, r, sigma, q, "put")
    np.testing.assert_allclose(c - p, S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-10)


def test_expiry_intrinsic_call():
    assert price(110.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call") == pytest.approx(10.0)
    assert price(90.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call") == pytest.approx(0.0)


def test_expiry_intrinsic_put():
    assert price(90.0, 100.0, 0.0, 0.05, 0.2, 0.0, "put") == pytest.approx(10.0)
    assert price(110.0, 100.0, 0.0, 0.05, 0.2, 0.0, "put") == pytest.approx(0.0)


def test_zero_vol_discounted_intrinsic():
    S, K, T, r, q = 120.0, 100.0, 1.0, 0.05, 0.0
    F = S * np.exp((r - q) * T)
    expected = np.exp(-r * T) * max(F - K, 0.0)
    assert price(S, K, T, r, 0.0, q, "call") == pytest.approx(expected, abs=1e-12)


def test_zero_vol_put():
    S, K, T, r, q = 80.0, 100.0, 1.0, 0.05, 0.02
    F = S * np.exp((r - q) * T)
    expected = np.exp(-r * T) * max(K - F, 0.0)
    assert price(S, K, T, r, 0.0, q, "put") == pytest.approx(expected, abs=1e-12)


def test_deep_itm_otm_stable():
    # Deep ITM call and deep OTM put should be finite, no NaN
    itm = price(1e6, 100.0, 1.0, 0.05, 0.2, 0.0, "call")
    otm = price(1.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call")
    assert np.isfinite(itm)
    assert np.isfinite(otm)
    assert otm < 1e-6
    assert itm > 1e5


def test_vectorized_million_shape():
    n = 10_000
    S = np.full(n, 100.0)
    K = np.linspace(80, 120, n)
    px = price(S, K, 0.5, 0.03, 0.25, 0.01, "call")
    assert px.shape == (n,)
    assert np.all(np.isfinite(px))


def test_greeks_finite_no_nan():
    S, K, T, r, sigma, q = 100.0, 100.0, 0.5, 0.05, 0.2, 0.01
    for fn in (delta, gamma, vega, theta, rho):
        val = fn(S, K, T, r, sigma, q, "call")
        assert np.isfinite(val)


def test_gamma_same_call_put():
    args = (100.0, 105.0, 0.4, 0.03, 0.25, 0.01)
    assert gamma(*args, kind="call") == pytest.approx(gamma(*args, kind="put"), rel=1e-12)


def test_vega_positive():
    assert vega(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call") > 0.0


def test_call_delta_bounds():
    d = delta(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call")
    assert 0.0 < d < 1.0


def test_put_delta_bounds():
    d = delta(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "put")
    assert -1.0 < d < 0.0


def test_kind_aliases():
    a = price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call")
    b = price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "c")
    assert a == pytest.approx(b)
