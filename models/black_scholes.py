"""
Black-Scholes Pricing Model
==============================
European option pricing with analytical Greeks.

Provides:
  - Call/put pricing
  - Full analytical Greeks (delta, gamma, vega, theta, rho, vanna, volga)
  - Implied vol solver (Newton-Raphson)
  - Vectorized operations for batch pricing

All formulas use the forward-based (Black) formulation:
  C = D * [F*N(d1) - K*N(d2)]
  P = D * [K*N(-d2) - F*N(-d1)]

where:
  d1 = [ln(F/K) + 0.5*sigma^2*T] / (sigma*sqrt(T))
  d2 = d1 - sigma*sqrt(T)
  D  = exp(-r*T)   (discount factor)
  F  = S*exp(r*T)  (forward price)
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Core pricing
# ---------------------------------------------------------------------------

def price(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    vol: float | np.ndarray,
    tenor: float | np.ndarray,
    rate: float = 0.045,
    is_call: bool = True,
) -> float | np.ndarray:
    """
    Black-Scholes European option price.

    Args:
        spot: Current underlying price.
        strike: Option strike price.
        vol: Annualized implied volatility (e.g., 0.20 for 20%).
        tenor: Time to expiry in years.
        rate: Risk-free rate (continuous compounding).
        is_call: True for call, False for put.

    Returns:
        Option price.
    """
    spot, strike, vol, tenor = _to_arrays(spot, strike, vol, tenor)
    d1, d2 = _d1d2(spot, strike, vol, tenor, rate)
    df = np.exp(-rate * tenor)

    if is_call:
        p = spot * norm.cdf(d1) - strike * df * norm.cdf(d2)
    else:
        p = strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1)

    return _squeeze(p)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def delta(spot, strike, vol, tenor, rate=0.045, is_call=True):
    """Option delta (dV/dS)."""
    spot, strike, vol, tenor = _to_arrays(spot, strike, vol, tenor)
    d1, _ = _d1d2(spot, strike, vol, tenor, rate)
    if is_call:
        return _squeeze(norm.cdf(d1))
    else:
        return _squeeze(norm.cdf(d1) - 1.0)


def gamma(spot, strike, vol, tenor, rate=0.045):
    """Option gamma (d2V/dS2). Same for calls and puts."""
    spot, strike, vol, tenor = _to_arrays(spot, strike, vol, tenor)
    d1, _ = _d1d2(spot, strike, vol, tenor, rate)
    return _squeeze(norm.pdf(d1) / (spot * vol * np.sqrt(tenor)))


def vega(spot, strike, vol, tenor, rate=0.045):
    """Option vega (dV/dsigma) per 1 vol point. Same for calls and puts."""
    spot, strike, vol, tenor = _to_arrays(spot, strike, vol, tenor)
    d1, _ = _d1d2(spot, strike, vol, tenor, rate)
    return _squeeze(spot * norm.pdf(d1) * np.sqrt(tenor))


def theta(spot, strike, vol, tenor, rate=0.045, is_call=True):
    """
    Option theta (dV/dt) per calendar day.
    Negative for long options (time decay).
    """
    spot, strike, vol, tenor = _to_arrays(spot, strike, vol, tenor)
    d1, d2 = _d1d2(spot, strike, vol, tenor, rate)
    df = np.exp(-rate * tenor)

    # First term: time decay of optionality
    term1 = -(spot * norm.pdf(d1) * vol) / (2 * np.sqrt(tenor))

    if is_call:
        term2 = -rate * strike * df * norm.cdf(d2)
        t = (term1 + term2) / 365.0
    else:
        term2 = rate * strike * df * norm.cdf(-d2)
        t = (term1 + term2) / 365.0

    return _squeeze(t)


def rho(spot, strike, vol, tenor, rate=0.045, is_call=True):
    """Option rho (dV/dr) per 1% rate move."""
    spot, strike, vol, tenor = _to_arrays(spot, strike, vol, tenor)
    _, d2 = _d1d2(spot, strike, vol, tenor, rate)
    df = np.exp(-rate * tenor)

    if is_call:
        return _squeeze(strike * tenor * df * norm.cdf(d2) / 100)
    else:
        return _squeeze(-strike * tenor * df * norm.cdf(-d2) / 100)


def vanna(spot, strike, vol, tenor, rate=0.045):
    """Vanna (d2V/dS dsigma) - sensitivity of delta to vol."""
    spot, strike, vol, tenor = _to_arrays(spot, strike, vol, tenor)
    d1, d2 = _d1d2(spot, strike, vol, tenor, rate)
    return _squeeze(-norm.pdf(d1) * d2 / vol)


def volga(spot, strike, vol, tenor, rate=0.045):
    """Volga / vomma (d2V/dsigma2) - sensitivity of vega to vol."""
    spot, strike, vol, tenor = _to_arrays(spot, strike, vol, tenor)
    d1, d2 = _d1d2(spot, strike, vol, tenor, rate)
    v = vega(spot, strike, vol, tenor, rate)
    return _squeeze(v * d1 * d2 / vol)


def all_greeks(spot, strike, vol, tenor, rate=0.045, is_call=True) -> dict:
    """Compute all Greeks in a single pass (efficient)."""
    spot, strike, vol, tenor = _to_arrays(spot, strike, vol, tenor)
    d1, d2 = _d1d2(spot, strike, vol, tenor, rate)
    df = np.exp(-rate * tenor)
    sqrt_t = np.sqrt(tenor)
    pdf_d1 = norm.pdf(d1)

    g = {}

    # Delta
    if is_call:
        g["delta"] = _squeeze(norm.cdf(d1))
    else:
        g["delta"] = _squeeze(norm.cdf(d1) - 1.0)

    # Gamma
    g["gamma"] = _squeeze(pdf_d1 / (spot * vol * sqrt_t))

    # Vega (per 1 vol point)
    g["vega"] = _squeeze(spot * pdf_d1 * sqrt_t)

    # Theta (per calendar day)
    term1 = -(spot * pdf_d1 * vol) / (2 * sqrt_t)
    if is_call:
        g["theta"] = _squeeze((term1 - rate * strike * df * norm.cdf(d2)) / 365.0)
    else:
        g["theta"] = _squeeze((term1 + rate * strike * df * norm.cdf(-d2)) / 365.0)

    # Price
    if is_call:
        g["price"] = _squeeze(spot * norm.cdf(d1) - strike * df * norm.cdf(d2))
    else:
        g["price"] = _squeeze(strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1))

    return g


# ---------------------------------------------------------------------------
# Implied vol solver
# ---------------------------------------------------------------------------

def implied_vol(
    market_price: float,
    spot: float,
    strike: float,
    tenor: float,
    rate: float = 0.045,
    is_call: bool = True,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Solve for implied vol using Newton-Raphson with vega.

    Args:
        market_price: Observed option price.
        spot, strike, tenor, rate: Option parameters.
        is_call: True for call.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        Implied volatility (annualized).
    """
    # Initial guess from Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * np.pi / tenor) * market_price / spot

    # Bounds
    sigma = max(min(sigma, 5.0), 0.001)

    for _ in range(max_iter):
        p = price(spot, strike, sigma, tenor, rate, is_call)
        v = vega(spot, strike, sigma, tenor, rate)

        if abs(v) < 1e-14:
            break

        diff = p - market_price
        if abs(diff) < tol:
            break

        sigma -= diff / v
        sigma = max(min(sigma, 5.0), 0.001)

    return float(sigma)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _d1d2(spot, strike, vol, tenor, rate):
    """Compute d1 and d2."""
    sqrt_t = np.sqrt(np.maximum(tenor, 1e-10))
    vol = np.maximum(vol, 1e-10)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol ** 2) * tenor) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def _to_arrays(*args):
    """Convert inputs to numpy arrays."""
    return tuple(np.atleast_1d(np.asarray(a, dtype=float)) for a in args)


def _squeeze(x):
    """Return scalar if single element, array otherwise."""
    x = np.asarray(x)
    if x.ndim == 0:
        return float(x)
    if x.size == 1:
        return float(x.flat[0])
    return x
