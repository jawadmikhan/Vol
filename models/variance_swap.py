"""
Variance Swap Replication
===========================
Replicates variance swaps from a strip of OTM options, weighted by 1/K^2.

A variance swap pays: Notional * (RealizedVar - StrikeVar)

The fair strike (K_var^2) is replicated by the log-contract:
  K_var^2 = (2/T) * integral[ (1/K^2) * max(C(K)-max(F-K,0), P(K)-max(K-F,0)) dK ]

In practice, discretized as a strip of OTM options:
  K_var^2 = (2/T) * sum[ (dK_i / K_i^2) * O_i ]

where O_i is the OTM option price at strike K_i:
  - Put for K_i < F (forward)
  - Call for K_i >= F

This module:
  1. Computes the fair variance strike from an option strip
  2. Builds the replicating portfolio (weighted OTM options)
  3. Computes Greeks of the replicating portfolio
  4. Tracks mark-to-market PnL of the variance swap

Reference:
  Demeterfi, Derman, Kamal, Zou (1999). "More Than You Ever Wanted
  to Know About Volatility Swaps." Goldman Sachs Quantitative Strategies.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import black_scholes as bs

logger = logging.getLogger(__name__)


@dataclass
class VarSwapStrip:
    """A discretized variance swap replicating portfolio."""
    strikes: np.ndarray             # Option strikes in the strip
    weights: np.ndarray             # 1/K^2 * dK weights
    is_call: np.ndarray             # True for calls (K >= F), False for puts
    option_prices: np.ndarray       # Current option mid-prices
    forward: float                  # Forward price
    tenor: float                    # Time to expiry (years)
    fair_var: float = 0.0           # Fair variance (annualized, in vol^2 terms)
    fair_vol: float = 0.0           # Fair vol strike (sqrt of fair_var)

    # Portfolio Greeks (aggregate of the strip)
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0


def build_var_swap_strip(
    forward: float,
    tenor: float,
    vol_surface_func,
    rate: float = 0.045,
    num_strikes: int = 50,
    strike_range: tuple[float, float] = (0.70, 1.30),
) -> VarSwapStrip:
    """
    Build a variance swap replicating portfolio from a vol surface.

    Args:
        forward: Forward price of the underlying.
        tenor: Time to expiry in years.
        vol_surface_func: Callable(strike) -> implied_vol for this tenor.
        rate: Risk-free rate.
        num_strikes: Number of strikes in the strip.
        strike_range: (low, high) as fraction of forward.

    Returns:
        VarSwapStrip with the replicating portfolio.
    """
    # Generate strike grid
    k_low = forward * strike_range[0]
    k_high = forward * strike_range[1]
    strikes = np.linspace(k_low, k_high, num_strikes)
    dk = strikes[1] - strikes[0]

    # Determine OTM side: puts below forward, calls at/above
    is_call = strikes >= forward

    # Get implied vols and price each option
    option_prices = np.zeros(num_strikes)
    deltas = np.zeros(num_strikes)
    gammas = np.zeros(num_strikes)
    vegas = np.zeros(num_strikes)
    thetas = np.zeros(num_strikes)

    for i, (k, ic) in enumerate(zip(strikes, is_call)):
        vol = vol_surface_func(k)
        option_prices[i] = bs.price(forward, k, vol, tenor, rate, bool(ic))

        g = bs.all_greeks(forward, k, vol, tenor, rate, bool(ic))
        deltas[i] = g["delta"]
        gammas[i] = g["gamma"]
        vegas[i] = g["vega"]
        thetas[i] = g["theta"]

    # Variance swap weights: 2/(T * K^2) * dK
    weights = 2.0 / (tenor * strikes ** 2) * dk

    # Fair variance = sum of weighted option prices
    fair_var = float(np.sum(weights * option_prices))

    # Adjust for discrete monitoring (Derman correction)
    # log(F/K0) term where K0 is the strike just below forward
    k0_idx = np.searchsorted(strikes, forward) - 1
    k0_idx = max(0, min(k0_idx, len(strikes) - 1))
    k0 = strikes[k0_idx]
    log_correction = -(1.0 / tenor) * (forward / k0 - 1 - np.log(forward / k0))
    fair_var += log_correction

    fair_var = max(fair_var, 1e-8)
    fair_vol = np.sqrt(fair_var)

    # Aggregate portfolio Greeks (weighted sum)
    strip_delta = float(np.sum(weights * deltas))
    strip_gamma = float(np.sum(weights * gammas))
    strip_vega = float(np.sum(weights * vegas))
    strip_theta = float(np.sum(weights * thetas))

    strip = VarSwapStrip(
        strikes=strikes,
        weights=weights,
        is_call=is_call,
        option_prices=option_prices,
        forward=forward,
        tenor=tenor,
        fair_var=fair_var,
        fair_vol=fair_vol,
        delta=strip_delta,
        gamma=strip_gamma,
        vega=strip_vega,
        theta=strip_theta,
    )

    logger.debug(
        "Var swap strip: F=%.0f T=%.2f fair_vol=%.2f%% %d strikes [%.0f-%.0f]",
        forward, tenor, fair_vol * 100, num_strikes, k_low, k_high,
    )

    return strip


def var_swap_pnl(
    notional: float,
    strike_var: float,
    realized_var: float,
) -> float:
    """
    Compute variance swap PnL.

    PnL = Notional * (RealizedVar - StrikeVar)

    For a SHORT variance swap: PnL = -Notional * (RealizedVar - StrikeVar)
      = Notional * (StrikeVar - RealizedVar)
      Positive when realized < implied (vol premium captured)

    Args:
        notional: Variance notional in dollars.
        strike_var: Variance strike (fair_var from replication).
        realized_var: Realized variance over the period.

    Returns:
        Dollar PnL.
    """
    return notional * (realized_var - strike_var)


def realized_variance(returns: np.ndarray, annualize: bool = True) -> float:
    """
    Compute realized variance from a return series.

    Uses the standard unbiased estimator:
      RV = (252/N) * sum(r_i^2)    [no mean adjustment, standard for var swaps]

    Args:
        returns: Array of daily log returns.
        annualize: If True, multiply by 252.

    Returns:
        Realized variance (annualized if requested).
    """
    if len(returns) == 0:
        return 0.0
    rv = float(np.mean(returns ** 2))
    if annualize:
        rv *= 252
    return rv


@dataclass
class DispersionVarSwap:
    """
    Dispersion trade using variance swaps.

    Structure:
      - SHORT index variance swap (sell correlation)
      - LONG constituent variance swaps (buy single-stock vol)

    The PnL depends on:
      correlation_pnl = index_var - weighted_sum(constituent_var)
      If realized correlation < implied correlation -> profit
    """
    index_notional: float
    constituent_notionals: dict  # {name: notional}
    index_strike_var: float
    constituent_strike_vars: dict  # {name: strike_var}
    weights: dict  # {name: index_weight}

    def daily_pnl(
        self,
        index_return: float,
        constituent_returns: dict[str, float],
    ) -> dict:
        """
        Compute daily PnL of the dispersion var swap.

        The key insight: a dispersion trade profits when the index moves
        less than the weighted sum of constituent moves (low correlation).

        Args:
            index_return: Daily log return of the index.
            constituent_returns: {name: daily_log_return}.

        Returns:
            Dict with PnL components.
        """
        # Index variance contribution (today)
        index_var_today = index_return ** 2 * 252

        # Constituent variance contribution (today)
        constituent_var_today = 0.0
        for name, notional in self.constituent_notionals.items():
            r = constituent_returns.get(name, 0.0)
            constituent_var_today += self.weights.get(name, 0.0) * (r ** 2 * 252)

        # Short index: profit when index realized var < strike
        index_pnl = -self.index_notional * (index_var_today - self.index_strike_var) / 252

        # Long constituents: profit when constituent realized var > strike
        constituent_pnl = 0.0
        for name, notional in self.constituent_notionals.items():
            r = constituent_returns.get(name, 0.0)
            strike_var = self.constituent_strike_vars.get(name, 0.04)
            constituent_pnl += notional * ((r ** 2 * 252) - strike_var) / 252

        # Correlation PnL: difference between index and weighted constituents
        # If realized corr < implied corr -> index var < weighted constituent var -> profit
        correlation_pnl = constituent_var_today - index_var_today

        return {
            "index_pnl": index_pnl,
            "constituent_pnl": constituent_pnl,
            "total_pnl": index_pnl + constituent_pnl,
            "index_var_daily": index_var_today,
            "constituent_var_daily": constituent_var_today,
            "correlation_pnl_signal": correlation_pnl,
        }


def build_dispersion_from_surface(
    vol_surface: pd.DataFrame,
    index_notional: float,
    constituent_total_notional: float,
    active_names: int = 12,
    tenor_months: int = 3,
) -> DispersionVarSwap:
    """
    Build a dispersion variance swap trade from the vol surface data.

    Args:
        vol_surface: Implied vol surface DataFrame.
        index_notional: Dollar notional for the index short leg.
        constituent_total_notional: Total notional across all constituent longs.
        active_names: Number of constituents to include.
        tenor_months: Option tenor.

    Returns:
        DispersionVarSwap ready for daily PnL computation.
    """
    # Extract ATM vols at the target tenor
    atm = vol_surface[
        (vol_surface["strike_delta"] == "ATM")
        & (vol_surface["tenor_months"] == tenor_months)
    ].set_index("name")

    if "SPX_INDEX" not in atm.index:
        logger.warning("SPX_INDEX not in surface, using default")
        index_strike_var = 0.18 ** 2
    else:
        index_iv = atm.loc["SPX_INDEX", "implied_vol"]
        index_strike_var = index_iv ** 2

    # Constituent strike variances
    constituents = atm.drop("SPX_INDEX", errors="ignore")

    # Select top N by implied vol (highest vol = most interesting for dispersion)
    if len(constituents) > active_names:
        constituents = constituents.nlargest(active_names, "implied_vol")

    notional_each = constituent_total_notional / max(len(constituents), 1)

    constituent_notionals = {}
    constituent_strike_vars = {}
    weights = {}

    for name in constituents.index:
        constituent_notionals[name] = notional_each
        iv = constituents.loc[name, "implied_vol"]
        constituent_strike_vars[name] = iv ** 2
        w = constituents.loc[name, "index_weight"] if "index_weight" in constituents.columns else 1.0 / len(constituents)
        weights[name] = w

    # Normalize weights
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}

    return DispersionVarSwap(
        index_notional=index_notional,
        constituent_notionals=constituent_notionals,
        index_strike_var=index_strike_var,
        constituent_strike_vars=constituent_strike_vars,
        weights=weights,
    )
