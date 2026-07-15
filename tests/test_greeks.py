"""Cross validation of analytic Greeks against central finite differences."""

import numpy as np
import pytest

from optionslib import greeks as fd
from optionslib.models.black_scholes import (
    delta as bs_delta,
    gamma as bs_gamma,
    price,
    rho as bs_rho,
    theta as bs_theta,
    vega as bs_vega,
)


CASES = [
    # S, K, T, r, sigma, q, kind
    (100.0, 100.0, 1.0, 0.05, 0.20, 0.00, "call"),
    (100.0, 100.0, 1.0, 0.05, 0.20, 0.00, "put"),
    (100.0, 90.0, 0.5, 0.03, 0.25, 0.01, "call"),
    (100.0, 110.0, 0.75, 0.04, 0.30, 0.02, "put"),
    (80.0, 100.0, 0.25, 0.02, 0.35, 0.00, "call"),
    (120.0, 100.0, 1.5, 0.06, 0.15, 0.01, "put"),
]


@pytest.mark.parametrize("S,K,T,r,sigma,q,kind", CASES)
def test_delta_matches_fd(S, K, T, r, sigma, q, kind):
    analytic = bs_delta(S, K, T, r, sigma, q, kind)
    numerical = fd.delta(price, S, K, T, r, sigma, q, kind, bump=1e-4)
    assert numerical == pytest.approx(analytic, rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("S,K,T,r,sigma,q,kind", CASES)
def test_gamma_matches_fd(S, K, T, r, sigma, q, kind):
    analytic = bs_gamma(S, K, T, r, sigma, q, kind)
    numerical = fd.gamma(price, S, K, T, r, sigma, q, kind, bump=1e-3)
    assert numerical == pytest.approx(analytic, rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("S,K,T,r,sigma,q,kind", CASES)
def test_vega_matches_fd(S, K, T, r, sigma, q, kind):
    analytic = bs_vega(S, K, T, r, sigma, q, kind)
    numerical = fd.vega(price, S, K, T, r, sigma, q, kind, bump=1e-4)
    assert numerical == pytest.approx(analytic, rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("S,K,T,r,sigma,q,kind", CASES)
def test_theta_matches_fd(S, K, T, r, sigma, q, kind):
    analytic = bs_theta(S, K, T, r, sigma, q, kind)
    numerical = fd.theta(price, S, K, T, r, sigma, q, kind, bump=1e-5)
    assert numerical == pytest.approx(analytic, rel=1e-4, abs=1e-3)


@pytest.mark.parametrize("S,K,T,r,sigma,q,kind", CASES)
def test_rho_matches_fd(S, K, T, r, sigma, q, kind):
    analytic = bs_rho(S, K, T, r, sigma, q, kind)
    numerical = fd.rho(price, S, K, T, r, sigma, q, kind, bump=1e-4)
    assert numerical == pytest.approx(analytic, rel=1e-4, abs=1e-4)


def test_all_greeks_keys():
    out = fd.all_greeks(price, 100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call")
    assert set(out) == {"delta", "gamma", "vega", "theta", "rho"}


def test_fd_vectorized_delta():
    S = np.array([90.0, 100.0, 110.0])
    analytic = bs_delta(S, 100.0, 0.5, 0.05, 0.2, 0.0, "call")
    numerical = fd.delta(price, S, 100.0, 0.5, 0.05, 0.2, 0.0, "call")
    np.testing.assert_allclose(numerical, analytic, rtol=1e-4, atol=1e-4)
