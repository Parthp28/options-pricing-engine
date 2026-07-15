"""Cox Ross Rubinstein binomial tree for European and American options."""

from __future__ import annotations

import numpy as np


def _crr_price(S, K, T, r, sigma, q, kind, steps, american):
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)
    q = float(q)
    n = int(steps)
    if T <= 0.0 or n <= 0:
        is_call = kind in ("call", "c", "C")
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)

    dt = T / n
    # Why: CRR sets u*d = 1, matching lognormal moments with a simple recombining tree
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    # Risk neutral probability with continuous dividend yield
    a = np.exp((r - q) * dt)
    p = (a - d) / (u - d)
    p = min(max(p, 0.0), 1.0)
    q_prob = 1.0 - p

    # Terminal underlying values: S * u^{n-j} * d^{j} for j=0..n
    j = np.arange(n + 1)
    ST = S * (u ** (n - j)) * (d ** j)
    is_call = kind in ("call", "c", "C")
    if is_call:
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    # Why: walk the tree with slice ops so each step is one vectorized update, not a node loop
    for step in range(n - 1, -1, -1):
        values = disc * (p * values[:-1] + q_prob * values[1:])
        if american:
            j = np.arange(step + 1)
            ST = S * (u ** (step - j)) * (d ** j)
            if is_call:
                intrinsic = np.maximum(ST - K, 0.0)
            else:
                intrinsic = np.maximum(K - ST, 0.0)
            values = np.maximum(values, intrinsic)

    return float(values[0])


def price(
    S,
    K,
    T,
    r,
    sigma,
    q=0.0,
    kind="call",
    steps=100,
    american=False,
    richardson=False,
):
    """CRR binomial price for European or American calls and puts.

    Backward induction is vectorized with numpy slices. Set richardson=True
    to extrapolate prices at N and N/2 steps for faster convergence.
    """
    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    sig_arr = np.asarray(sigma, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    kind_arr = np.asarray(kind)

    arrays = np.broadcast_arrays(S_arr, K_arr, T_arr, r_arr, sig_arr, q_arr)
    S_b, K_b, T_b, r_b, sig_b, q_b = arrays
    if kind_arr.shape != S_b.shape:
        kind_b = np.broadcast_to(kind_arr, S_b.shape)
    else:
        kind_b = kind_arr

    flat = S_b.size
    out = np.empty(flat, dtype=float)
    S_f = S_b.ravel()
    K_f = K_b.ravel()
    T_f = T_b.ravel()
    r_f = r_b.ravel()
    sig_f = sig_b.ravel()
    q_f = q_b.ravel()
    kind_f = np.asarray(kind_b).ravel()

    n = int(steps)
    for i in range(flat):
        p_n = _crr_price(S_f[i], K_f[i], T_f[i], r_f[i], sig_f[i], q_f[i], kind_f[i], n, american)
        if richardson and n >= 2:
            # Why: Richardson extrapolation cancels the leading 1/N bias of the CRR scheme
            n2 = n // 2
            p_h = _crr_price(
                S_f[i], K_f[i], T_f[i], r_f[i], sig_f[i], q_f[i], kind_f[i], n2, american
            )
            out[i] = 2.0 * p_n - p_h
        else:
            out[i] = p_n

    result = out.reshape(S_b.shape)
    if result.shape == ():
        return float(result)
    return result
