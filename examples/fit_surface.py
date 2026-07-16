"""Fit a vol surface from a synthetic chain and run arbitrage checks."""

from optionslib.chains import generate_chain
from optionslib.surface import VolSurface


def main():
    chain = generate_chain(seed=11, noise=0.0)
    surf = VolSurface.fit(chain)
    report = surf.check_arbitrage()
    print(report)
    print(f"vol(K=100, T=0.75) = {surf.vol(100.0, 0.75):.4f}")
    print(f"vol(K=90,  T=0.75) = {surf.vol(90.0, 0.75):.4f}")
    print(f"vol(K=110, T=0.75) = {surf.vol(110.0, 0.75):.4f}")


if __name__ == "__main__":
    main()
