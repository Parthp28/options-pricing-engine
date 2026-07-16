"""Synthetic chain and volatility surface tests."""

import numpy as np
import pytest

from optionslib.chains import generate_chain, smile_vol
from optionslib.surface import VolSurface, corrupt_butterfly, corrupt_calendar


def test_chain_deterministic():
    a = generate_chain(seed=42, noise=0.01)
    b = generate_chain(seed=42, noise=0.01)
    np.testing.assert_array_equal(a["call_mid"], b["call_mid"])
    np.testing.assert_array_equal(a["put_mid"], b["put_mid"])


def test_chain_shape():
    expiries = np.array([0.25, 0.5, 1.0])
    strikes = np.linspace(80, 120, 9)
    chain = generate_chain(expiries=expiries, strikes=strikes, seed=0)
    assert chain["call_mid"].shape == (len(expiries) * len(strikes),)
    assert np.all(chain["call_mid"] >= 0)
    assert np.all(chain["put_mid"] >= 0)


def test_smile_vol_positive():
    K = np.linspace(70, 130, 20)
    v = smile_vol(K, 100.0, 0.5)
    assert np.all(v >= 0.01)


def test_surface_fit_recovers_atm():
    chain = generate_chain(seed=1, noise=0.0)
    surf = VolSurface.fit(chain)
    T = chain["expiries"][1]
    atm = surf.vol(chain["S"], T)
    # True smile atm vol
    true = smile_vol(chain["S"], chain["S"], T)
    assert atm == pytest.approx(float(true), rel=0.05)


def test_surface_vol_query_vectorized():
    chain = generate_chain(seed=2)
    surf = VolSurface.fit(chain)
    K = np.array([90.0, 100.0, 110.0])
    T = 0.5
    vols = surf.vol(K, T)
    assert vols.shape == (3,)
    assert np.all(vols > 0)


def test_clean_surface_passes_arbitrage():
    chain = generate_chain(seed=3, noise=0.0, skew=-0.05, smile=0.04)
    surf = VolSurface.fit(chain)
    report = surf.check_arbitrage()
    assert report.butterfly_ok
    assert report.calendar_ok
    assert report.ok


def test_corrupt_butterfly_fails():
    chain = generate_chain(seed=4, noise=0.0)
    surf = VolSurface.fit(chain)
    bad = corrupt_butterfly(surf, expiry_index=0, scale=-0.5)
    report = bad.check_arbitrage()
    assert not report.butterfly_ok


def test_corrupt_calendar_fails():
    chain = generate_chain(seed=5, noise=0.0)
    surf = VolSurface.fit(chain)
    bad = corrupt_calendar(surf)
    report = bad.check_arbitrage()
    assert not report.calendar_ok


def test_total_variance_increases_clean():
    chain = generate_chain(seed=6, noise=0.0)
    surf = VolSurface.fit(chain)
    K = 100.0
    w = [float(surf.total_variance(K, T)) for T in surf.expiries]
    assert np.all(np.diff(w) >= -1e-10)


def test_surface_interpolates_between_expiries():
    chain = generate_chain(seed=7, noise=0.0)
    surf = VolSurface.fit(chain)
    T_mid = 0.5 * (surf.expiries[0] + surf.expiries[1])
    v = surf.vol(100.0, T_mid)
    assert np.isfinite(v) and v > 0
