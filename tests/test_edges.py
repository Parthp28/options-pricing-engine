"""Extra edge case tests aimed at uncovered branches."""

import numpy as np
import pytest

from optionslib.implied_vol import implied_vol
from optionslib.models import binomial, monte_carlo
from optionslib.models.black_scholes import (
    delta,
    greeks,
    price,
)
from optionslib.surface import ArbitrageReport, VolSurface
from optionslib.chains import generate_chain


def test_bs_greeks_bundle_matches_parts():
    out = greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.01, "call")
    assert set(out) == {"price", "delta", "gamma", "vega", "theta", "rho"}
    assert out["price"] == pytest.approx(price(100.0, 100.0, 1.0, 0.05, 0.2, 0.01, "call"))


def test_bs_greeks_expired_and_zero_vol():
    out_exp = greeks(110.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call")
    assert out_exp["price"] == pytest.approx(10.0)
    out_zv = greeks(120.0, 100.0, 1.0, 0.05, 0.0, 0.0, "call")
    F = 120.0 * np.exp(0.05)
    assert out_zv["price"] == pytest.approx(np.exp(-0.05) * (F - 100.0))


def test_delta_expired_itm_otm_atm():
    assert delta(110.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call") == pytest.approx(1.0)
    assert delta(90.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call") == pytest.approx(0.0)
    assert delta(100.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call") == pytest.approx(0.5)
    assert delta(90.0, 100.0, 0.0, 0.05, 0.2, 0.0, "put") == pytest.approx(-1.0)
    assert delta(110.0, 100.0, 0.0, 0.05, 0.2, 0.0, "put") == pytest.approx(0.0)


def test_delta_zero_vol():
    d_call = delta(120.0, 100.0, 1.0, 0.05, 0.0, 0.0, "call")
    assert d_call == pytest.approx(1.0)
    d_put = delta(80.0, 100.0, 1.0, 0.05, 0.0, 0.0, "put")
    assert d_put == pytest.approx(-1.0)


def test_iv_expired_returns_zero():
    px = price(110.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call")
    iv = implied_vol(px, 110.0, 100.0, 0.0, 0.05, 0.0, "call")
    assert iv == pytest.approx(0.0)


def test_iv_near_intrinsic_returns_zero():
    S, K, T, r, q = 150.0, 100.0, 0.01, 0.0, 0.0
    lo = max(S - K * np.exp(-r * T), 0.0)
    # Price essentially at lower bound
    iv = implied_vol(lo + 1e-16, S, K, T, r, q, "call")
    assert iv == pytest.approx(0.0)


def test_binomial_expired():
    assert binomial.price(110, 100, 0.0, 0.05, 0.2, 0.0, "call", steps=50) == pytest.approx(10.0)
    assert binomial.price(90, 100, 0.0, 0.05, 0.2, 0.0, "put", steps=50) == pytest.approx(10.0)


def test_mc_expired():
    mean, se, lo, hi = monte_carlo.price(110, 100, 0.0, 0.05, 0.2, 0.0, "call", n_paths=100, seed=0)
    assert mean == pytest.approx(10.0)
    assert se == 0.0


def test_arbitrage_report_repr():
    r = ArbitrageReport(True, False, ["x"], ["y"])
    assert "butterfly_ok=True" in repr(r)
    assert r.ok is False


def test_surface_fit_with_noise():
    chain = generate_chain(seed=9, noise=0.001)
    surf = VolSurface.fit(chain, use_calls=True)
    assert surf.vol(100.0, chain["expiries"][0]) > 0


def test_surface_put_fit():
    chain = generate_chain(seed=10, noise=0.0)
    surf = VolSurface.fit(chain, use_calls=False)
    assert np.isfinite(surf.vol(100.0, 0.5))


def test_greeks_fd_with_kind_in_bump():
    from optionslib.greeks import delta as fd_delta

    # exercise _bump kind branch via normal call already; also array S
    d = fd_delta(price, np.array([100.0, 101.0]), 100.0, 1.0, 0.05, 0.2, 0.0, "put")
    assert d.shape == (2,)
