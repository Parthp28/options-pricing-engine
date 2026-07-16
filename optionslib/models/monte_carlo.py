"""GBM Monte Carlo pricing with antithetic and control variate options."""

from __future__ import annotations

import numpy as np


def _payoff(ST, K, kind):
    is_call = kind in ("call", "c", "C")
    if is_call:
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)


def simulate_terminal(S, T, r, sigma, q, n_paths, rng, antithetic=False):
    """Draw terminal GBM spots. With antithetic, returns 2 * n_paths samples."""
    Z = rng.standard_normal(n_paths)
    if antithetic:
        Z = np.concatenate([Z, -Z])
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    return S * np.exp(drift + vol * Z)


def simulate_paths(S, T, r, sigma, q, n_paths, n_steps, rng, antithetic=False):
    """Full GBM paths, shape (n_total, n_steps + 1) including t=0."""
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    if antithetic:
        Z = np.vstack([Z, -Z])
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    log_increments = drift + vol * Z
    log_paths = np.cumsum(log_increments, axis=1)
    S_paths = S * np.exp(log_paths)
    S0 = np.full((S_paths.shape[0], 1), S, dtype=float)
    return np.hstack([S0, S_paths])


def price(
    S,
    K,
    T,
    r,
    sigma,
    q=0.0,
    kind="call",
    n_paths=100_000,
    n_steps=1,
    seed=None,
    antithetic=False,
    control_variate=False,
    full_paths=False,
):
    """Monte Carlo European price under GBM.

    Returns (price, standard_error, ci_low, ci_high) for a 95 percent interval.
    Control variate uses the discounted terminal spot, mean S * exp(-q T).
    """
    S, K, T, r, sigma, q = (float(S), float(K), float(T), float(r), float(sigma), float(q))
    rng = np.random.default_rng(seed)

    if T <= 0.0:
        px = float(_payoff(np.array([S]), K, kind)[0])
        return px, 0.0, px, px

    if full_paths or n_steps > 1:
        paths = simulate_paths(S, T, r, sigma, q, n_paths, n_steps, rng, antithetic)
        ST = paths[:, -1]
    else:
        ST = simulate_terminal(S, T, r, sigma, q, n_paths, rng, antithetic)

    pay = _payoff(ST, K, kind)
    disc = np.exp(-r * T)
    samples = disc * pay

    if control_variate:
        # Why: E[e^{-rT} S_T] = S e^{-qT} under the risk neutral measure, a free exact mean
        cv = disc * ST
        cv_mean = S * np.exp(-q * T)
        # Optimal beta = Cov(payoff_disc, cv) / Var(cv)
        cov = np.cov(samples, cv, ddof=1)
        var_cv = cov[1, 1]
        beta = 0.0 if var_cv <= 0.0 else cov[0, 1] / var_cv
        samples = samples - beta * (cv - cv_mean)

    n = samples.size
    mean = float(np.mean(samples))
    # With antithetic pairing, SE uses n_paths independent pairs when antithetic is on
    if antithetic and not control_variate:
        # Why: pair Z and -Z so the SE reflects independent antithetic averages
        paired = 0.5 * (samples[: n // 2] + samples[n // 2 :])
        mean = float(np.mean(paired))
        se = float(np.std(paired, ddof=1) / np.sqrt(paired.size))
    elif antithetic and control_variate:
        paired = 0.5 * (samples[: n // 2] + samples[n // 2 :])
        mean = float(np.mean(paired))
        se = float(np.std(paired, ddof=1) / np.sqrt(paired.size))
    else:
        se = float(np.std(samples, ddof=1) / np.sqrt(n))

    z = 1.959963984540054  # Phi^{-1}(0.975)
    return mean, se, mean - z * se, mean + z * se
