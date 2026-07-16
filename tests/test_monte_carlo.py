"""Monte Carlo confidence interval and variance reduction tests."""

import numpy as np
import pytest

from optionslib.models import monte_carlo as mc
from optionslib.models.black_scholes import price as bs_price


GRID = [
    (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call"),
    (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "put"),
    (100.0, 90.0, 0.5, 0.03, 0.25, 0.01, "call"),
    (100.0, 110.0, 0.75, 0.04, 0.3, 0.0, "put"),
]


@pytest.mark.parametrize("S,K,T,r,sigma,q,kind", GRID)
def test_mc_ci_contains_bs(S, K, T, r, sigma, q, kind):
    analytic = bs_price(S, K, T, r, sigma, q, kind)
    mean, se, lo, hi = mc.price(
        S, K, T, r, sigma, q, kind, n_paths=80_000, seed=123, antithetic=True, control_variate=True
    )
    assert lo <= analytic <= hi


def test_mc_deterministic_seed():
    a = mc.price(100, 100, 1, 0.05, 0.2, 0.0, "call", n_paths=5_000, seed=7)
    b = mc.price(100, 100, 1, 0.05, 0.2, 0.0, "call", n_paths=5_000, seed=7)
    assert a == b


def test_variance_reduction_factor():
    kwargs = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, q=0.0, kind="call", n_paths=50_000)
    _, se_plain, _, _ = mc.price(**kwargs, seed=99, antithetic=False, control_variate=False)
    _, se_vr, _, _ = mc.price(**kwargs, seed=99, antithetic=True, control_variate=True)
    factor = se_plain / se_vr
    assert factor >= 2.0


def test_full_paths_shape_and_price():
    mean, se, lo, hi = mc.price(
        100, 100, 1.0, 0.05, 0.2, 0.0, "call",
        n_paths=20_000, n_steps=50, seed=3, full_paths=True, antithetic=True
    )
    analytic = bs_price(100, 100, 1.0, 0.05, 0.2, 0.0, "call")
    assert lo <= analytic <= hi
    assert se > 0


def test_antithetic_alone_reduces_se():
    kwargs = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, q=0.0, kind="call", n_paths=40_000)
    _, se_plain, _, _ = mc.price(**kwargs, seed=1, antithetic=False)
    _, se_anti, _, _ = mc.price(**kwargs, seed=1, antithetic=True)
    assert se_anti < se_plain


def test_control_variate_alone_reduces_se():
    kwargs = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, q=0.0, kind="call", n_paths=40_000)
    _, se_plain, _, _ = mc.price(**kwargs, seed=2, control_variate=False)
    _, se_cv, _, _ = mc.price(**kwargs, seed=2, control_variate=True)
    assert se_cv < se_plain


def test_simulate_terminal_mean_near_forward():
    rng = np.random.default_rng(0)
    S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.2, 0.01
    ST = mc.simulate_terminal(S, T, r, sigma, q, 200_000, rng)
    assert np.mean(ST) == pytest.approx(S * np.exp((r - q) * T), rel=0.01)
