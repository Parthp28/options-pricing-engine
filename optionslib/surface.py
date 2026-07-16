"""Volatility surface fit from option chains with arbitrage checks."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from optionslib.implied_vol import implied_vol
from optionslib.models.black_scholes import price as bs_price


class ArbitrageReport:
    """Pass/fail summary for butterfly and calendar checks."""

    def __init__(self, butterfly_ok, calendar_ok, butterfly_details=None, calendar_details=None):
        self.butterfly_ok = bool(butterfly_ok)
        self.calendar_ok = bool(calendar_ok)
        self.butterfly_details = butterfly_details or []
        self.calendar_details = calendar_details or []

    @property
    def ok(self):
        return self.butterfly_ok and self.calendar_ok

    def __repr__(self):
        return (
            f"ArbitrageReport(butterfly_ok={self.butterfly_ok}, "
            f"calendar_ok={self.calendar_ok})"
        )


class VolSurface:
    """Cubic spline smile in implied variance vs log moneyness, per expiry.

    Time interpolation uses total variance w = sigma^2 * T between expiries.
    """

    def __init__(self, S, r, q, expiries, strikes, implied_vars, smile_splines):
        self.S = float(S)
        self.r = float(r)
        self.q = float(q)
        self.expiries = np.asarray(expiries, dtype=float)
        self.strikes = np.asarray(strikes, dtype=float)
        self.implied_vars = implied_vars  # list of arrays, one per expiry
        self.smile_splines = smile_splines  # CubicSpline per expiry on log-moneyness
        self._log_k = np.log(self.strikes / self.S)

    @classmethod
    def fit(cls, chain, use_calls=True):
        """Fit a surface from a synthetic (or real shaped) chain dict."""
        S, r, q = chain["S"], chain["r"], chain["q"]
        expiries = np.asarray(chain["expiries"], dtype=float)
        strikes = np.asarray(chain["strikes"], dtype=float)
        mid = chain["call_mid"] if use_calls else chain["put_mid"]
        kind = "call" if use_calls else "put"

        # Reshape quotes to (n_exp, n_strike)
        n_e, n_k = len(expiries), len(strikes)
        mids = np.asarray(mid, dtype=float).reshape(n_e, n_k)
        log_k = np.log(strikes / S)

        ivs = []
        var_splines = []
        implied_vars = []
        for i, T in enumerate(expiries):
            px = mids[i]
            iv = implied_vol(px, S, strikes, T, r, q, kind)
            iv = np.asarray(iv, dtype=float)
            # Fallback to chain true vol if IV fails at a sparse point
            if np.any(~np.isfinite(iv)):
                if "vol" in chain:
                    true_v = np.asarray(chain["vol"], dtype=float).reshape(n_e, n_k)[i]
                    iv = np.where(np.isfinite(iv), iv, true_v)
                else:
                    iv = np.nan_to_num(iv, nan=0.2)
            ivs.append(iv)
            w = iv**2  # implied variance (not yet total)
            implied_vars.append(w)
            # Why: spline on variance vs log moneyness stays smoother than raw vol
            var_splines.append(CubicSpline(log_k, w, bc_type="natural", extrapolate=True))

        return cls(S, r, q, expiries, strikes, implied_vars, var_splines)

    def _var_at_expiry_index(self, i, K):
        x = np.log(np.asarray(K, dtype=float) / self.S)
        return np.maximum(self.smile_splines[i](x), 1e-8)

    def total_variance(self, K, T):
        """Interpolate total variance sigma^2 * T in maturity at fixed strike."""
        K = np.asarray(K, dtype=float)
        T = np.asarray(T, dtype=float)
        K_b, T_b = np.broadcast_arrays(K, T)
        out = np.empty(K_b.shape, dtype=float)

        t_min, t_max = self.expiries[0], self.expiries[-1]
        for idx in np.ndindex(K_b.shape):
            k = K_b[idx]
            t = float(T_b[idx])
            t_clip = min(max(t, t_min), t_max)
            # Total variance w(T) = sigma^2(T) * T at each expiry node
            w_nodes = np.array([
                float(self._var_at_expiry_index(i, k)) * float(self.expiries[i])
                for i in range(len(self.expiries))
            ])
            # Why: linear interp in total variance keeps calendar shapes simple and stable
            w = float(np.interp(t_clip, self.expiries, w_nodes))
            out[idx] = w
        return out if out.shape else float(out)

    def vol(self, K, T):
        """Query Black Scholes vol at strike K and maturity T inside the fit region."""
        K = np.asarray(K, dtype=float)
        T = np.asarray(T, dtype=float)
        K_b, T_b = np.broadcast_arrays(K, T)
        w = self.total_variance(K_b, T_b)
        with np.errstate(divide="ignore", invalid="ignore"):
            sig = np.sqrt(np.maximum(w, 0.0) / np.maximum(T_b, 1e-12))
        if sig.shape == ():
            return float(sig)
        return sig

    def check_arbitrage(self, n_strike_grid=40, n_money=21):
        """Butterfly (call convexity in K) and calendar (total var vs T) checks."""
        butterfly_details = []
        calendar_details = []

        # Butterfly: for each expiry, call price should be convex in K
        k_grid = np.linspace(self.strikes.min(), self.strikes.max(), n_strike_grid)
        butterfly_ok = True
        for i, T in enumerate(self.expiries):
            sig = np.sqrt(np.maximum(self._var_at_expiry_index(i, k_grid), 1e-8))
            calls = bs_price(self.S, k_grid, T, self.r, sig, self.q, "call")
            # Discrete second difference
            d2 = calls[:-2] - 2.0 * calls[1:-1] + calls[2:]
            # Allow tiny numerical noise
            if np.any(d2 < -1e-8):
                butterfly_ok = False
                bad = np.where(d2 < -1e-8)[0]
                butterfly_details.append(
                    f"expiry={T}: negative call convexity at {len(bad)} grid points"
                )

        # Calendar: total variance non decreasing in T at fixed moneyness
        calendar_ok = True
        moneyness = np.linspace(0.8, 1.2, n_money)
        strikes = moneyness * self.S
        for k in strikes:
            w = np.array([
                float(self._var_at_expiry_index(i, k)) * float(self.expiries[i])
                for i in range(len(self.expiries))
            ])
            if np.any(np.diff(w) < -1e-10):
                calendar_ok = False
                calendar_details.append(f"strike={k:.4f}: total variance decreased")

        return ArbitrageReport(butterfly_ok, calendar_ok, butterfly_details, calendar_details)


def corrupt_butterfly(surface, expiry_index=0, scale=-0.05):
    """Return a new surface with a deliberately non convex smile bump."""
    # Rebuild splines with a dip in mid variance to break convexity
    new_vars = [v.copy() for v in surface.implied_vars]
    mid = len(new_vars[expiry_index]) // 2
    new_vars[expiry_index][mid] *= 1.0 + scale  # reduce mid variance
    log_k = np.log(surface.strikes / surface.S)
    new_splines = [
        CubicSpline(log_k, w, bc_type="natural", extrapolate=True) for w in new_vars
    ]
    return VolSurface(
        surface.S, surface.r, surface.q, surface.expiries, surface.strikes, new_vars, new_splines
    )


def corrupt_calendar(surface):
    """Return a surface whose longer expiry has lower total variance."""
    new_vars = [v.copy() for v in surface.implied_vars]
    # Shrink the last expiry's variance enough to invert total variance
    T0, T1 = surface.expiries[0], surface.expiries[-1]
    new_vars[-1] = new_vars[0] * (T0 / T1) * 0.5
    log_k = np.log(surface.strikes / surface.S)
    new_splines = [
        CubicSpline(log_k, w, bc_type="natural", extrapolate=True) for w in new_vars
    ]
    return VolSurface(
        surface.S, surface.r, surface.q, surface.expiries, surface.strikes, new_vars, new_splines
    )
