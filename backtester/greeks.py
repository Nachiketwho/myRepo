"""Black-Scholes option pricing and Greeks.

Used to derive option premiums and sensitivities from spot price,
strike, time-to-expiry, risk-free rate, and implied volatility
(sourced from India VIX).
"""

import math
from typing import Literal

OptionType = Literal["CE", "PE"]

# ---------------------------------------------------------------------------
# Standard normal helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Cumulative distribution function of the standard normal."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Probability density function of the standard normal."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# d1/d2 terms
# ---------------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(max(T, 0))


# ---------------------------------------------------------------------------
# Black-Scholes price
# ---------------------------------------------------------------------------

def bs_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: OptionType = "CE",
) -> float:
    """Black-Scholes European option price.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years (e.g. 7/365 for 7 calendar days).
        r: Risk-free rate (e.g. 0.07 for 7%).
        sigma: Implied volatility as decimal (e.g. 0.13 for 13%).
        option_type: 'CE' for call, 'PE' for put.

    Returns:
        Theoretical option premium.
    """
    if T <= 0:
        # At expiry: intrinsic value only
        if option_type == "CE":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    if option_type == "CE":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: OptionType = "CE",
) -> float:
    """Delta: rate of premium change per 1-point spot move."""
    if T <= 0:
        if option_type == "CE":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1 = _d1(S, K, T, r, sigma)
    if option_type == "CE":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Gamma: rate of delta change per 1-point spot move."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: OptionType = "CE",
) -> float:
    """Theta: daily premium decay (negative = premium lost per day).

    Returns value in points per calendar day.
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)

    # First term: time decay
    term1 = -(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)

    if option_type == "CE":
        term2 = -r * K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        term2 = r * K * math.exp(-r * T) * _norm_cdf(-d2)

    # Convert from per-year to per-calendar-day
    return (term1 + term2) / 365.0


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega: premium change per 1% (0.01) change in IV.

    Returns value in points per 1 percentage-point IV change.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T) / 100.0


# ---------------------------------------------------------------------------
# Greeks snapshot
# ---------------------------------------------------------------------------

def greeks_snapshot(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: OptionType = "CE",
) -> dict:
    """Return all Greeks + premium in a single dict."""
    return {
        "premium": round(bs_price(S, K, T, r, sigma, option_type), 2),
        "delta": round(delta(S, K, T, r, sigma, option_type), 4),
        "gamma": round(gamma(S, K, T, r, sigma), 6),
        "theta": round(theta(S, K, T, r, sigma, option_type), 2),
        "vega": round(vega(S, K, T, r, sigma), 2),
        "iv": round(sigma * 100, 2),
    }


# ---------------------------------------------------------------------------
# Implied Volatility (Newton-Raphson)
# ---------------------------------------------------------------------------

def implied_vol(
    market_price: float, S: float, K: float, T: float, r: float,
    option_type: OptionType = "CE",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """Derive implied volatility from market premium using Newton-Raphson.

    Returns IV as decimal (e.g. 0.13) or None if not converged.
    """
    if market_price <= 0 or T <= 0:
        return None

    sigma = 0.20  # initial guess 20%

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        v = vega(S, K, T, r, sigma) * 100  # vega returns per 1%, need per 1.0
        if abs(v) < 1e-12:
            break
        sigma = sigma - (price - market_price) / v
        if sigma <= 0:
            sigma = 0.001
        if abs(price - market_price) < tol:
            return sigma

    return sigma  # best estimate even if not fully converged
