"""Synthetic option chain generator with a quadratic smile model."""

from __future__ import annotations

import numpy as np

from optionslib.models.black_scholes import price as bs_price


def smile_vol(K, S, T, base_vol=0.2, skew=-0.1, smile=0.05, term=0.02):
    """Quadratic smile in log moneyness with a mild term structure bump."""
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    x = np.log(K / S)
    # Why: atm vol drifts slowly with maturity so longer tenors are not flat clones
    atm = base_vol + term * np.sqrt(np.maximum(T, 1e-12))
    return np.maximum(atm + skew * x + smile * x**2, 0.01)


def generate_chain(
    S=100.0,
    r=0.03,
    q=0.01,
    expiries=None,
    strikes=None,
    base_vol=0.2,
    skew=-0.1,
    smile=0.05,
    term=0.02,
    seed=None,
    noise=0.0,
):
    """Build a synthetic call/put chain from the smile model.

    Returns a dict with arrays: expiry, strike, call_mid, put_mid, call_bid,
    call_ask, put_bid, put_ask, and the true smile vol used for each quote.
    """
    if expiries is None:
        expiries = np.array([0.25, 0.5, 1.0, 1.5])
    if strikes is None:
        strikes = np.linspace(0.7 * S, 1.3 * S, 13)

    expiries = np.asarray(expiries, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    rng = np.random.default_rng(seed)

    exp_grid, strike_grid = np.meshgrid(expiries, strikes, indexing="ij")
    T = exp_grid.ravel()
    K = strike_grid.ravel()
    sig = smile_vol(K, S, T, base_vol=base_vol, skew=skew, smile=smile, term=term)

    call_mid = bs_price(S, K, T, r, sig, q, "call")
    put_mid = bs_price(S, K, T, r, sig, q, "put")

    if noise > 0.0:
        call_noise = rng.normal(0.0, noise, size=call_mid.shape)
        put_noise = rng.normal(0.0, noise, size=put_mid.shape)
        call_mid = np.maximum(call_mid + call_noise, 0.0)
        put_mid = np.maximum(put_mid + put_noise, 0.0)
        spread = np.maximum(0.01, 0.02 * np.maximum(call_mid, put_mid))
    else:
        spread = np.full_like(call_mid, 0.01)

    return {
        "S": float(S),
        "r": float(r),
        "q": float(q),
        "expiry": T,
        "strike": K,
        "vol": sig,
        "call_mid": np.asarray(call_mid, dtype=float),
        "put_mid": np.asarray(put_mid, dtype=float),
        "call_bid": np.asarray(call_mid - 0.5 * spread, dtype=float),
        "call_ask": np.asarray(call_mid + 0.5 * spread, dtype=float),
        "put_bid": np.asarray(put_mid - 0.5 * spread, dtype=float),
        "put_ask": np.asarray(put_mid + 0.5 * spread, dtype=float),
        "expiries": expiries,
        "strikes": strikes,
    }
