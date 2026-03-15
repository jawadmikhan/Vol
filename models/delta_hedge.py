"""
Delta-Hedging Simulator
=========================
Simulates discrete delta-hedging of option positions and tracks
the realized gamma PnL vs theta paid.

This is where short vol actually makes or loses money:
  Gamma PnL = sum of 0.5 * Gamma * dS^2 (from hedging)
  Theta PnL = sum of Theta * dt (time decay collected/paid)
  Net = Gamma PnL + Theta PnL

If realized vol < implied vol → net positive (vol premium captured)
If realized vol > implied vol → net negative (vol premium lost)

Supports configurable hedging frequency:
  - Continuous (theoretical limit)
  - Daily
  - Every N hours (intraday)
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from . import black_scholes as bs

logger = logging.getLogger(__name__)


@dataclass
class HedgeState:
    """Tracks the state of the delta hedge for a single position."""
    position_delta: float = 0.0     # Option delta (shares equivalent)
    hedge_shares: float = 0.0       # Shares held to hedge
    hedge_cost_basis: float = 0.0   # Cash spent on hedge shares
    cumulative_hedge_pnl: float = 0.0
    cumulative_gamma_pnl: float = 0.0
    cumulative_theta_pnl: float = 0.0
    cumulative_txn_cost: float = 0.0
    rebalance_count: int = 0


@dataclass
class DeltaHedger:
    """
    Simulates discrete delta-hedging for a portfolio of option positions.

    The hedger:
    1. Computes portfolio delta from BS model
    2. Trades underlying to flatten delta at each hedge interval
    3. Tracks realized gamma PnL from spot moves between hedges
    4. Tracks theta collected/paid
    5. Accounts for transaction costs on hedge trades

    Usage:
        hedger = DeltaHedger(hedge_frequency_days=1, txn_cost_bps=3.0)
        result = hedger.simulate(positions, spot_path, vol_surface_path, ...)
    """
    hedge_frequency_days: float = 1.0   # 1.0 = daily, 0.5 = twice daily
    txn_cost_bps: float = 3.0           # Cost per hedge trade in basis points
    rate: float = 0.045

    def simulate_day(
        self,
        state: HedgeState,
        prev_spot: float,
        curr_spot: float,
        portfolio_gamma_usd: float,
        portfolio_theta_usd: float,
        portfolio_delta_usd: float,
    ) -> dict:
        """
        Simulate one day of delta-hedging.

        This is the simplified version that works with aggregate portfolio Greeks
        (compatible with the existing strategy framework).

        Args:
            state: Current hedge state (mutated in place).
            prev_spot: Previous day's spot price.
            curr_spot: Current spot price.
            portfolio_gamma_usd: Dollar gamma of the portfolio.
            portfolio_theta_usd: Dollar theta (daily).
            portfolio_delta_usd: Dollar delta before hedging.

        Returns:
            Dict with daily hedge PnL components.
        """
        spot_move = curr_spot - prev_spot
        spot_return = spot_move / prev_spot if prev_spot > 0 else 0.0

        # Gamma PnL: 0.5 * Gamma_$ * (dS/S)^2 * S^2
        # Simplified: 0.5 * Gamma_$ * dS^2
        gamma_pnl = 0.5 * portfolio_gamma_usd * (spot_return ** 2)

        # Theta PnL (daily): theta is already in $/day
        theta_pnl = portfolio_theta_usd

        # Hedge rebalance: trade to flatten delta
        delta_to_hedge = portfolio_delta_usd
        shares_to_trade = abs(delta_to_hedge / curr_spot) if curr_spot > 0 else 0
        txn_cost = shares_to_trade * curr_spot * self.txn_cost_bps / 10000

        # Hedge PnL: the hedge position gains/loses with spot moves
        hedge_pnl = -state.hedge_shares * spot_move  # hedge is opposite to delta
        state.hedge_shares = -delta_to_hedge / curr_spot if curr_spot > 0 else 0

        # Net PnL from hedging activity
        net_hedge_pnl = gamma_pnl + theta_pnl - txn_cost

        # Update state
        state.cumulative_gamma_pnl += gamma_pnl
        state.cumulative_theta_pnl += theta_pnl
        state.cumulative_txn_cost += txn_cost
        state.cumulative_hedge_pnl += net_hedge_pnl
        state.rebalance_count += 1

        return {
            "gamma_pnl": gamma_pnl,
            "theta_pnl": theta_pnl,
            "hedge_pnl": hedge_pnl,
            "txn_cost": txn_cost,
            "net_hedge_pnl": net_hedge_pnl,
            "shares_traded": shares_to_trade,
            "spot_return": spot_return,
        }

    def simulate_path(
        self,
        spot_path: np.ndarray,
        gamma_path: np.ndarray,
        theta_path: np.ndarray,
        delta_path: np.ndarray,
    ) -> dict:
        """
        Simulate delta-hedging over a full price path.

        Args:
            spot_path: Array of daily spot prices.
            gamma_path: Array of daily portfolio dollar gamma.
            theta_path: Array of daily portfolio dollar theta.
            delta_path: Array of daily portfolio dollar delta.

        Returns:
            Dict with arrays of daily PnL components and summary stats.
        """
        n = len(spot_path)
        state = HedgeState()

        gamma_pnls = np.zeros(n)
        theta_pnls = np.zeros(n)
        txn_costs = np.zeros(n)
        net_pnls = np.zeros(n)

        for i in range(1, n):
            result = self.simulate_day(
                state=state,
                prev_spot=spot_path[i - 1],
                curr_spot=spot_path[i],
                portfolio_gamma_usd=gamma_path[i],
                portfolio_theta_usd=theta_path[i],
                portfolio_delta_usd=delta_path[i],
            )
            gamma_pnls[i] = result["gamma_pnl"]
            theta_pnls[i] = result["theta_pnl"]
            txn_costs[i] = result["txn_cost"]
            net_pnls[i] = result["net_hedge_pnl"]

        return {
            "gamma_pnl": gamma_pnls,
            "theta_pnl": theta_pnls,
            "txn_cost": txn_costs,
            "net_pnl": net_pnls,
            "total_gamma_pnl": state.cumulative_gamma_pnl,
            "total_theta_pnl": state.cumulative_theta_pnl,
            "total_txn_cost": state.cumulative_txn_cost,
            "total_net_pnl": state.cumulative_hedge_pnl,
            "rebalance_count": state.rebalance_count,
            "vol_premium_captured": state.cumulative_gamma_pnl + state.cumulative_theta_pnl,
        }
