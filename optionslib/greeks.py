"""Central finite difference Greeks wrapping any pricer callable."""

from __future__ import annotations

import numpy as np


def _bump(pricer, base_kwargs, key, bump):
    kw_up = dict(base_kwargs)
    kw_dn = dict(base_kwargs)
    kw_up[key] = np.asarray(base_kwargs[key], dtype=float) + bump
    kw_dn[key] = np.asarray(base_kwargs[key], dtype=float) - bump
    return pricer(**kw_up), pricer(**kw_dn)


def delta(pricer, S, K, T, r, sigma, q=0.0, kind="call", bump=1e-4):
    """Central difference delta: dV/dS."""
    base = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind=kind)
    up, dn = _bump(pricer, base, "S", bump * max(float(np.mean(np.asarray(S))), 1.0))
    h = bump * max(float(np.mean(np.asarray(S))), 1.0)
    return (up - dn) / (2.0 * h)


def gamma(pricer, S, K, T, r, sigma, q=0.0, kind="call", bump=1e-4):
    """Central difference gamma: d2V/dS2."""
    S = np.asarray(S, dtype=float)
    h = bump * max(float(np.mean(S)), 1.0)
    mid = pricer(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind=kind)
    up = pricer(S=S + h, K=K, T=T, r=r, sigma=sigma, q=q, kind=kind)
    dn = pricer(S=S - h, K=K, T=T, r=r, sigma=sigma, q=q, kind=kind)
    return (up - 2.0 * mid + dn) / (h * h)


def vega(pricer, S, K, T, r, sigma, q=0.0, kind="call", bump=1e-4):
    """Central difference vega: dV/dsigma."""
    base = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind=kind)
    up, dn = _bump(pricer, base, "sigma", bump)
    return (up - dn) / (2.0 * bump)


def theta(pricer, S, K, T, r, sigma, q=0.0, kind="call", bump=1e-5):
    """Central difference theta: dV/dT with sign flipped to calendar decay.

    Price falls as T shrinks, so theta = (V(T-h) - V(T+h)) / (2h) matches
    the usual negative-carry sign of analytic BS theta.
    """
    T = np.asarray(T, dtype=float)
    # Why: keep T-h >= 0 so the BS edge path is defined
    h = np.minimum(bump, np.maximum(T / 2.0, bump * 0.1))
    up = pricer(S=S, K=K, T=T + h, r=r, sigma=sigma, q=q, kind=kind)
    dn = pricer(S=S, K=K, T=np.maximum(T - h, 0.0), r=r, sigma=sigma, q=q, kind=kind)
    return (dn - up) / (2.0 * h)


def rho(pricer, S, K, T, r, sigma, q=0.0, kind="call", bump=1e-4):
    """Central difference rho: dV/dr."""
    base = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind=kind)
    up, dn = _bump(pricer, base, "r", bump)
    return (up - dn) / (2.0 * bump)


def all_greeks(pricer, S, K, T, r, sigma, q=0.0, kind="call", bump=1e-4):
    """Compute all FD Greeks for a pricer callable with signature like BS price."""
    return {
        "delta": delta(pricer, S, K, T, r, sigma, q, kind, bump),
        "gamma": gamma(pricer, S, K, T, r, sigma, q, kind, bump),
        "vega": vega(pricer, S, K, T, r, sigma, q, kind, bump),
        "theta": theta(pricer, S, K, T, r, sigma, q, kind, bump),
        "rho": rho(pricer, S, K, T, r, sigma, q, kind, bump),
    }
