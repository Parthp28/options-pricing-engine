# options-pricing-engine

Pricing engine for European and American options in pure Python. Black-Scholes with analytic Greeks, Cox-Ross-Rubinstein binomial trees, Monte Carlo with variance reduction, a Newton-Raphson implied volatility solver, and an arbitrage-checked volatility surface fit from option chains.

Everything runs offline. Every number in the benchmark table below comes from a real measured run of `python -m optionslib.bench` on the build machine.

## Architecture
                    ┌───────────────────┐
                    │   chains.py       │
                    │  synthetic quotes │
                    └─────────┬─────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
  ┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
  │ black_scholes  │  │   binomial     │  │ monte_carlo    │
  │ vectorized     │  │  European +    │  │ antithetic +   │
  │ analytic Greeks│  │   American     │  │ control variate│
  └───────┬────────┘  └────────────────┘  └────────────────┘
          │
  ┌───────▼────────┐          ┌────────────────┐
  │ implied_vol    │◄─────────┤   greeks.py    │
  │ Newton + Brent │          │ finite diff xv │
  └───────┬────────┘          └────────────────┘
          │
  ┌───────▼────────┐
  │  surface.py    │
  │ spline fit +   │
  │ butterfly and  │
  │ calendar check │
  └────────────────┘

  ## Benchmarks

Measured with `python -m optionslib.bench`.

| Metric                              | Result             |
| ----------------------------------- | ------------------ |
| Black-Scholes vectorized            | 8,560,523 prices/s |
| Batch implied vol, 100k quotes      | 0.1165 s           |
| Binomial American put, 1000 steps   | 14.57 ms           |
| Monte Carlo, 100k paths x 252 steps | 0.5524 s           |
| MC standard error reduction         | 2.67x              |

## Design Notes

**Why CRR over trinomial or a general lattice.** CRR has a clean vectorized backward induction using two numpy slices per step, and it converges to Black-Scholes at a predictable rate that gives us a clean unit test. Trinomial trees are more accurate per step but harder to vectorize cleanly, and the extra accuracy is not worth the code complexity here.

**Why a spline fit over SVI or SABR.** A cubic spline on implied variance in log moneyness is stable, monotonic where the data is monotonic, and cheap to evaluate. SVI would give a tighter parametric form but adds a nonlinear fit per expiry and does not automatically satisfy no-arbitrage without extra constraint work. The spline plus explicit butterfly and calendar checks catches the same arbitrage cases with less code.

**Why a terminal stock price control variate for Monte Carlo.** The discounted terminal price has known expectation `S0 * exp(-qT)` and is highly correlated with the payoff for standard European options. Combining it with antithetic sampling gets a 2.67x standard error reduction at no extra path cost, which is the biggest lever available before moving to quasi-random sequences.

**Why Newton then Brent for implied vol.** Newton with analytic vega converges quadratically in the well-behaved region, which is where 95% of quotes sit. The per-element Brent fallback handles deep in and out of the money quotes and short expiries where vega collapses and Newton stalls. Doing this in two phases keeps the vectorized path fast without giving up on the hard corners.

## Validation

- Put-call parity holds within 1e-10 across a broad parameter grid
- Binomial European price within 0.01% of Black-Scholes at 2000 steps
- American put is at least the European put everywhere, strictly greater in the money with positive rates
- Analytic Greeks match central finite differences within 1e-4 relative error
- IV round trip: price at a known sigma, solve back, recover sigma within 1e-8
- Butterfly and calendar checks pass on clean synthetic surfaces and correctly flag deliberately corrupted ones
- Test fixtures include published values from Hull, chapter numbers named in test comments

## Quickstart

```bash
pip install -e .
python examples/price_a_chain.py
python examples/fit_surface.py
```

## Tests

```bash
pytest --cov=optionslib
```

100 tests, 95.78% coverage.