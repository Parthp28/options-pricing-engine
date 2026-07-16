"""Price a synthetic option chain with Black Scholes and the binomial tree."""

from optionslib.chains import generate_chain
from optionslib.models import binomial
from optionslib.models.black_scholes import price as bs_price


def main():
    chain = generate_chain(S=100.0, seed=7, noise=0.0)
    # Spot check a few ATM like quotes against the tree
    idx = 0
    for i in range(min(5, len(chain["strike"]))):
        K = chain["strike"][i]
        T = chain["expiry"][i]
        sig = chain["vol"][i]
        bs = bs_price(chain["S"], K, T, chain["r"], sig, chain["q"], "call")
        tree = binomial.price(
            chain["S"], K, T, chain["r"], sig, chain["q"], "call", steps=200, american=False
        )
        print(f"K={K:.2f} T={T:.2f} BS={bs:.4f} tree={tree:.4f} mid={chain['call_mid'][i]:.4f}")
        idx += 1


if __name__ == "__main__":
    main()
