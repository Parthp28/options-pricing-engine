"""Black Scholes Merton pricing and analytic Greeks for European options."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _as_array(*args):
    # Why: broadcast once so price and Greeks share identical shapes
    return np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])


def _d1_d2(S, K, T, r, sigma, q):
    """Compute d1 and d2 once for reuse by price and Greeks."""
    S, K, T, r, sigma, q = _as_array(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(np.maximum(T, 0.0))
    # Why: clamp sigma*sqrt(T) away from zero only for the d1 path; edges handled separately
    denom = sigma * sqrt_T
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / denom
        d2 = d1 - denom
    return d1, d2, S, K, T, r, sigma, q, sqrt_T


def _intrinsic(S, K, kind):
    kind = np.asarray(kind)
    call = np.maximum(S - K, 0.0)
    put = np.maximum(K - S, 0.0)
    is_call = (kind == "call") | (kind == "c") | (kind == "C")
    return np.where(is_call, call, put)


def _is_call_mask(kind, shape):
    kind = np.asarray(kind)
    if kind.shape != shape:
        kind = np.broadcast_to(kind, shape)
    return (kind == "call") | (kind == "c") | (kind == "C")


def price(S, K, T, r, sigma, q=0.0, kind="call"):
    """European Black Scholes Merton price with continuous dividend yield q.

    Every argument accepts a scalar or array. Broadcasting applies across all inputs.
    At expiry (T=0) returns intrinsic. At sigma=0 returns discounted intrinsic on the forward.
    """
    d1, d2, S, K, T, r, sigma, q, sqrt_T = _d1_d2(S, K, T, r, sigma, q)
    kind_arr = np.asarray(kind)
    is_call = _is_call_mask(kind_arr, S.shape)

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    forward_S = S * disc_q
    forward_K = K * disc_r

    call = forward_S * norm.cdf(d1) - forward_K * norm.cdf(d2)
    put = forward_K * norm.cdf(-d2) - forward_S * norm.cdf(-d1)
    result = np.where(is_call, call, put)

    # Why: T=0 is intrinsic; sigma=0 is discounted intrinsic on the forward (still European)
    expired = T <= 0.0
    zero_vol = (~expired) & (sigma <= 0.0)
    if np.any(expired):
        result = np.where(expired, _intrinsic(S, K, kind_arr if kind_arr.shape == S.shape else np.broadcast_to(kind_arr, S.shape)), result)
    if np.any(zero_vol):
        F = S * np.exp((r - q) * T)
        disc = np.exp(-r * T)
        call_zv = disc * np.maximum(F - K, 0.0)
        put_zv = disc * np.maximum(K - F, 0.0)
        zv = np.where(is_call, call_zv, put_zv)
        result = np.where(zero_vol, zv, result)

    return result if result.shape else float(result)


def delta(S, K, T, r, sigma, q=0.0, kind="call"):
    """Analytic delta. Shares d1 with the rest of the Greek set."""
    d1, d2, S, K, T, r, sigma, q, sqrt_T = _d1_d2(S, K, T, r, sigma, q)
    is_call = _is_call_mask(kind, S.shape)
    disc_q = np.exp(-q * T)
    call_d = disc_q * norm.cdf(d1)
    put_d = disc_q * (norm.cdf(d1) - 1.0)
    result = np.where(is_call, call_d, put_d)

    expired = T <= 0.0
    if np.any(expired):
        intrinsic_delta_call = np.where(S > K, disc_q, np.where(S < K, 0.0, 0.5 * disc_q))
        # At T=0 and sigma=0 edges: use digital like step; ATM call delta convention 0.5*e^{-qT}
        at_expiry = np.where(is_call, np.where(S > K, 1.0, np.where(S < K, 0.0, 0.5)),
                             np.where(S < K, -1.0, np.where(S > K, 0.0, -0.5)))
        result = np.where(expired, at_expiry * np.exp(-q * np.maximum(T, 0.0)), result)

    zero_vol = (T > 0.0) & (sigma <= 0.0)
    if np.any(zero_vol):
        F = S * np.exp((r - q) * T)
        disc_q = np.exp(-q * T)
        call_zv = np.where(F > K, disc_q, 0.0)
        put_zv = np.where(F < K, -disc_q, 0.0)
        # ATM forward: average of digital jump
        call_zv = np.where(F == K, 0.5 * disc_q, call_zv)
        put_zv = np.where(F == K, -0.5 * disc_q, put_zv)
        zv = np.where(is_call, call_zv, put_zv)
        result = np.where(zero_vol, zv, result)

    return result if result.ndim else float(result)


def gamma(S, K, T, r, sigma, q=0.0, kind="call"):
    """Analytic gamma. Identical for calls and puts."""
    d1, d2, S, K, T, r, sigma, q, sqrt_T = _d1_d2(S, K, T, r, sigma, q)
    denom = S * sigma * sqrt_T
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.exp(-q * T) * norm.pdf(d1) / denom
    g = np.where((T <= 0.0) | (sigma <= 0.0) | (S <= 0.0), 0.0, g)
    return g if g.ndim else float(g)


def vega(S, K, T, r, sigma, q=0.0, kind="call"):
    """Analytic vega (per unit vol, not per percent)."""
    d1, d2, S, K, T, r, sigma, q, sqrt_T = _d1_d2(S, K, T, r, sigma, q)
    v = S * np.exp(-q * T) * norm.pdf(d1) * sqrt_T
    v = np.where((T <= 0.0) | (S <= 0.0), 0.0, v)
    return v if v.ndim else float(v)


def theta(S, K, T, r, sigma, q=0.0, kind="call"):
    """Analytic theta in calendar time (per year)."""
    d1, d2, S, K, T, r, sigma, q, sqrt_T = _d1_d2(S, K, T, r, sigma, q)
    is_call = _is_call_mask(kind, S.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        density_term = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * sqrt_T)
    density_term = np.where((T <= 0.0) | (sigma <= 0.0), 0.0, density_term)
    call_t = density_term + q * S * np.exp(-q * T) * norm.cdf(d1) - r * K * np.exp(-r * T) * norm.cdf(d2)
    put_t = density_term - q * S * np.exp(-q * T) * norm.cdf(-d1) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    result = np.where(is_call, call_t, put_t)
    result = np.where(T <= 0.0, 0.0, result)
    return result if result.ndim else float(result)


def rho(S, K, T, r, sigma, q=0.0, kind="call"):
    """Analytic rho (per unit rate, not per percent)."""
    d1, d2, S, K, T, r, sigma, q, sqrt_T = _d1_d2(S, K, T, r, sigma, q)
    is_call = _is_call_mask(kind, S.shape)
    call_r = K * T * np.exp(-r * T) * norm.cdf(d2)
    put_r = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    result = np.where(is_call, call_r, put_r)
    result = np.where(T <= 0.0, 0.0, result)
    return result if result.ndim else float(result)


def greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """Return all analytic Greeks in one pass, sharing d1 and d2."""
    d1, d2, S, K, T, r, sigma, q, sqrt_T = _d1_d2(S, K, T, r, sigma, q)
    is_call = _is_call_mask(kind, S.shape)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    call_price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    put_price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    px = np.where(is_call, call_price, put_price)

    call_d = disc_q * norm.cdf(d1)
    put_d = disc_q * (norm.cdf(d1) - 1.0)
    dlt = np.where(is_call, call_d, put_d)

    with np.errstate(divide="ignore", invalid="ignore"):
        gmm = disc_q * norm.pdf(d1) / (S * sigma * sqrt_T)
    gmm = np.where((T <= 0.0) | (sigma <= 0.0) | (S <= 0.0), 0.0, gmm)

    vg = S * disc_q * norm.pdf(d1) * sqrt_T
    vg = np.where((T <= 0.0) | (S <= 0.0), 0.0, vg)

    with np.errstate(divide="ignore", invalid="ignore"):
        density_term = -S * disc_q * norm.pdf(d1) * sigma / (2.0 * sqrt_T)
    density_term = np.where((T <= 0.0) | (sigma <= 0.0), 0.0, density_term)
    call_t = density_term + q * S * disc_q * norm.cdf(d1) - r * K * disc_r * norm.cdf(d2)
    put_t = density_term - q * S * disc_q * norm.cdf(-d1) + r * K * disc_r * norm.cdf(-d2)
    th = np.where(is_call, call_t, put_t)
    th = np.where(T <= 0.0, 0.0, th)

    call_r = K * T * disc_r * norm.cdf(d2)
    put_r = -K * T * disc_r * norm.cdf(-d2)
    rh = np.where(is_call, call_r, put_r)
    rh = np.where(T <= 0.0, 0.0, rh)

    # Edge: T=0 intrinsic price
    expired = T <= 0.0
    if np.any(expired):
        kind_b = np.broadcast_to(np.asarray(kind), S.shape)
        px = np.where(expired, _intrinsic(S, K, kind_b), px)

    zero_vol = (~expired) & (sigma <= 0.0)
    if np.any(zero_vol):
        F = S * np.exp((r - q) * T)
        call_zv = disc_r * np.maximum(F - K, 0.0)
        put_zv = disc_r * np.maximum(K - F, 0.0)
        px = np.where(zero_vol, np.where(is_call, call_zv, put_zv), px)

    out = {
        "price": px if px.ndim else float(px),
        "delta": dlt if dlt.ndim else float(dlt),
        "gamma": gmm if gmm.ndim else float(gmm),
        "vega": vg if vg.ndim else float(vg),
        "theta": th if th.ndim else float(th),
        "rho": rh if rh.ndim else float(rh),
    }
    return out
