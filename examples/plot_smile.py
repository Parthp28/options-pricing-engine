"""Plot a fitted smile at one expiry (matplotlib allowed in examples only)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optionslib.chains import generate_chain
from optionslib.surface import VolSurface


def main():
    chain = generate_chain(seed=21, noise=0.0)
    surf = VolSurface.fit(chain)
    T = float(surf.expiries[1])
    K = np.linspace(surf.strikes.min(), surf.strikes.max(), 80)
    vols = surf.vol(K, T)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(K, vols, color="#1f4e79", lw=2.0)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied vol")
    ax.set_title(f"Fitted smile at T={T:.2f}")
    ax.grid(True, alpha=0.3)
    out = Path(__file__).resolve().parent / "smile.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
