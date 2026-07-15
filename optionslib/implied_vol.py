"""Implied volatility via Newton Raphson with Brent fallback."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from optionslib.models.black_scholes import price as bs_price
from optionslib.models.black_scholes import vega as bs_vega


class ImpliedVolError(ValueError):
    """Raised when a quote sits outside no arbitrage price bounds."""


def _bounds(S, K, T, r, q, kind):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    kind = np.asarray(kind)
    is_call = (kind == "call") | (kind == "c") | (kind == "C")
    call_lo = np.maximum(S * disc_q - K * disc_r, 0.0)
    call_hi = S * disc_q
    put_lo = np.maximum(K * disc_r - S * disc_q, 0.0)
    put_hi = K * disc_r
    lo = np.where(is_call, call_lo, put_lo)
    hi = np.where(is_call, call_hi, put_hi)
    return lo, hi


def _brent_one(target, S, K, T, r, q, kind, lo_vol=1e-8, hi_vol=5.0):
    def objective(sig):
        return float(bs_price(S, K, T, r, sig, q, kind)) - target

    f_lo = objective(lo_vol)
    f_hi = objective(hi_vol)
    if abs(f_lo) < 1e-14:
        return lo_vol
    if abs(f_hi) < 1e-14:
        return hi_vol
    if f_lo * f_hi > 0:
        for _ in range(10):
            hi_vol *= 2.0
            f_hi = objective(hi_vol)
            if f_lo * f_hi <= 0:
                break
        else:
            # Flat objective: price insensitive to vol (deep ITM or OTM)
            return lo_vol if abs(f_lo) <= abs(f_hi) else hi_vol
    return brentq(objective, lo_vol, hi_vol, xtol=1e-14, maxiter=200)


def implied_vol(
    market_price,
    S,
    K,
    T,
    r,
    q=0.0,
    kind="call",
    initial_guess=0.2,
    tol=1e-10,
    max_iter=50,
    vega_floor=1e-8,
):
    """Solve Black Scholes implied volatility for scalar or array quotes.

    Newton Raphson uses analytic vega. Elements that stall fall back to Brent.
    Prices outside no arbitrage bounds raise ImpliedVolError naming the bound.
    """
    mp = np.asarray(market_price, dtype=float)
    S, K, T, r, q = [np.asarray(x, dtype=float) for x in (S, K, T, r, q)]
    kind_arr = np.asarray(kind)
    mp, S, K, T, r, q = np.broadcast_arrays(mp, S, K, T, r, q)
    if kind_arr.shape != mp.shape:
        kind_arr = np.broadcast_to(kind_arr, mp.shape)
    kind_arr = np.asarray(kind_arr)

    lo, hi = _bounds(S, K, T, r, q, kind_arr)
    eps = 1e-10 * np.maximum(np.abs(hi), 1.0)
    below = mp < lo - eps
    above = mp > hi + eps
    if np.any(below):
        idx = int(np.argmax(below))
        raise ImpliedVolError(
            f"price below lower no arbitrage bound at index {idx}: "
            f"price={mp.flat[idx]}, lower_bound={lo.flat[idx]}"
        )
    if np.any(above):
        idx = int(np.argmax(above))
        raise ImpliedVolError(
            f"price above upper no arbitrage bound at index {idx}: "
            f"price={mp.flat[idx]}, upper_bound={hi.flat[idx]}"
        )

    scalar_out = mp.ndim == 0
    sigma = np.full(mp.shape, float(initial_guess), dtype=float)
    expired = T <= 0.0
    sigma = np.where(expired, 0.0, sigma)

    # Why: near intrinsic the BS map is flat in vol; mark for Brent immediately
    time_value = mp - lo
    near_bound = (~expired) & (time_value <= 1e-12 * np.maximum(S, 1.0))
    need_brent = near_bound.copy()
    active = (~expired) & (~near_bound)

    for _ in range(max_iter):
        if not np.any(active):
            break
        px = bs_price(S, K, T, r, sigma, q, kind_arr)
        vg = bs_vega(S, K, T, r, sigma, q, kind_arr)
        diff = px - mp
        # Relative tolerance with a small absolute floor for moderate premiums
        thr = tol * np.maximum(np.abs(mp), 1.0)
        converged = active & (np.abs(diff) < thr)
        active &= ~converged

        weak = active & (np.abs(vg) < vega_floor)
        need_brent |= weak
        active &= ~weak

        if not np.any(active):
            break

        with np.errstate(divide="ignore", invalid="ignore"):
            step = diff / np.where(np.abs(vg) < vega_floor, np.nan, vg)
        step = np.where(np.isfinite(step), step, 0.0)
        new_sigma = np.clip(sigma - step, 1e-8, 10.0)
        # Why: oscillating Newton steps get a Brent cleanup rather than more iterations
        stuck = active & (np.abs(new_sigma - sigma) < 1e-14) & (np.abs(diff) >= thr)
        need_brent |= stuck
        active &= ~stuck
        sigma = np.where(active | converged, new_sigma, sigma)
        # Keep converged values
        sigma = np.where(converged, new_sigma, sigma)

    # Final pricing check picks up Newton failures the loop missed
    px_final = bs_price(S, K, T, r, sigma, q, kind_arr)
    thr = tol * np.maximum(np.abs(mp), 1.0)
    need_brent |= (~expired) & (np.abs(px_final - mp) >= thr)

    if np.any(need_brent):
        flat_idx = np.where(need_brent.ravel())[0]
        sig_flat = sigma.ravel().copy()
        mp_f = mp.ravel()
        S_f, K_f, T_f, r_f, q_f = S.ravel(), K.ravel(), T.ravel(), r.ravel(), q.ravel()
        kind_f = kind_arr.ravel()
        lo_f = lo.ravel()
        for i in flat_idx:
            # Near intrinsic: any tiny vol fits; report zero vol limit
            if (mp_f[i] - lo_f[i]) <= 1e-12 * max(S_f[i], 1.0):
                sig_flat[i] = 0.0
                continue
            try:
                sig_flat[i] = _brent_one(
                    float(mp_f[i]),
                    float(S_f[i]),
                    float(K_f[i]),
                    float(T_f[i]),
                    float(r_f[i]),
                    float(q_f[i]),
                    kind_f[i],
                )
            except Exception:
                sig_flat[i] = np.nan
        sigma = sig_flat.reshape(mp.shape)

    if scalar_out:
        return float(np.asarray(sigma))
    return sigma
