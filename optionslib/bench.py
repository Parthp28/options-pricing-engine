"""Performance benchmarks for the pricing engine.

Run with: python -m optionslib.bench
"""

from __future__ import annotations

import time

import numpy as np

from optionslib.implied_vol import implied_vol
from optionslib.models import binomial, monte_carlo
from optionslib.models.black_scholes import price as bs_price


def _fmt(x):
    if isinstance(x, float):
        if x >= 1000:
            return f"{x:,.0f}"
        if x >= 10:
            return f"{x:.2f}"
        return f"{x:.4f}"
    return str(x)


def bench_black_scholes(n=1_000_000):
    rng = np.random.default_rng(0)
    S = np.full(n, 100.0)
    K = rng.uniform(70, 130, n)
    T = rng.uniform(0.05, 2.0, n)
    r = 0.03
    sigma = rng.uniform(0.1, 0.5, n)
    q = 0.01
    # Warmup
    bs_price(S[:10_000], K[:10_000], T[:10_000], r, sigma[:10_000], q, "call")
    t0 = time.perf_counter()
    px = bs_price(S, K, T, r, sigma, q, "call")
    elapsed = time.perf_counter() - t0
    assert px.shape == (n,)
    rate = n / elapsed
    return {"name": "Black Scholes vectorized", "value": rate, "unit": "prices/s", "elapsed": elapsed, "n": n}


def bench_implied_vol(n=100_000):
    rng = np.random.default_rng(1)
    S = np.full(n, 100.0)
    K = rng.uniform(85, 115, n)
    T = rng.uniform(0.25, 2.0, n)
    r = 0.03
    q = 0.01
    sigma = rng.uniform(0.1, 0.5, n)
    px = bs_price(S, K, T, r, sigma, q, "call")
    # Warmup
    implied_vol(px[:1000], S[:1000], K[:1000], T[:1000], r, q, "call")
    t0 = time.perf_counter()
    iv = implied_vol(px, S, K, T, r, q, "call")
    elapsed = time.perf_counter() - t0
    assert iv.shape == (n,)
    return {"name": "Batch IV (100k quotes)", "value": elapsed, "unit": "seconds", "elapsed": elapsed, "n": n}


def bench_binomial_american(steps=1000):
    # Warmup
    binomial.price(100.0, 100.0, 1.0, 0.05, 0.25, 0.0, "put", steps=100, american=True)
    t0 = time.perf_counter()
    px = binomial.price(100.0, 100.0, 1.0, 0.05, 0.25, 0.0, "put", steps=steps, american=True)
    elapsed = time.perf_counter() - t0
    assert px > 0
    return {
        "name": "Binomial American put (1000 steps)",
        "value": elapsed * 1000.0,
        "unit": "ms",
        "elapsed": elapsed,
        "n": steps,
    }


def bench_monte_carlo(n_paths=100_000, n_steps=252):
    # Warmup
    monte_carlo.price(100, 100, 1.0, 0.05, 0.2, 0.0, "call", n_paths=1000, n_steps=10, seed=0)
    t0 = time.perf_counter()
    mean, se, lo, hi = monte_carlo.price(
        100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call",
        n_paths=n_paths, n_steps=n_steps, seed=0, full_paths=True,
    )
    elapsed = time.perf_counter() - t0
    return {
        "name": "Monte Carlo (100k paths x 252 steps)",
        "value": elapsed,
        "unit": "seconds",
        "elapsed": elapsed,
        "n": n_paths,
        "se": se,
    }


def bench_variance_reduction(n_paths=50_000):
    kwargs = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, q=0.0, kind="call", n_paths=n_paths)
    _, se_plain, _, _ = monte_carlo.price(**kwargs, seed=42, antithetic=False, control_variate=False)
    _, se_vr, _, _ = monte_carlo.price(**kwargs, seed=42, antithetic=True, control_variate=True)
    factor = se_plain / se_vr if se_vr > 0 else float("inf")
    return {
        "name": "MC variance reduction factor",
        "value": factor,
        "unit": "x SE reduction",
        "se_plain": se_plain,
        "se_vr": se_vr,
    }


def run_all():
    """Run every benchmark and print a markdown table for the README."""
    results = [
        bench_black_scholes(),
        bench_implied_vol(),
        bench_binomial_american(),
        bench_monte_carlo(),
        bench_variance_reduction(),
    ]

    print("## Benchmark results")
    print()
    print("| Metric | Result | Target |")
    print("| --- | --- | --- |")
    targets = [
        ("Black Scholes vectorized", results[0]["value"], "prices/s", "1,000,000+ /s"),
        ("Batch IV (100k quotes)", results[1]["value"], "s", "< 2 s"),
        ("Binomial American put (1000 steps)", results[2]["value"], "ms", "< 50 ms"),
        ("Monte Carlo (100k x 252)", results[3]["value"], "s", "< 3 s"),
        ("MC SE reduction (antithetic + CV)", results[4]["value"], "x", ">= 2x"),
    ]
    # Use measured display strings
    rows = [
        (targets[0][0], f"{results[0]['value']:,.0f} prices/s", targets[0][3]),
        (targets[1][0], f"{results[1]['value']:.4f} s", targets[1][3]),
        (targets[2][0], f"{results[2]['value']:.2f} ms", targets[2][3]),
        (targets[3][0], f"{results[3]['value']:.4f} s", targets[3][3]),
        (targets[4][0], f"{results[4]['value']:.2f}x", targets[4][3]),
    ]
    for name, result, target in rows:
        print(f"| {name} | {result} | {target} |")

    print()
    print("RAW_RESULTS")
    for r in results:
        print(r)
    return results


def main():
    run_all()


if __name__ == "__main__":
    main()
