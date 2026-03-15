"""
Transaction Cost Model
========================
Models realistic trading costs for options and equities.

Components:
  1. Bid-ask spread (varies by moneyness, tenor, underlying liquidity)
  2. Market impact (size-dependent)
  3. Commission (fixed per contract)
  4. Delta-hedging cost (equity trading costs from hedging)

Spread model calibrated to typical US equity option markets:
  - ATM options: ~2-5% of option value
  - OTM options: ~5-15% of option value
  - Deep OTM: ~15-30% of option value
  - SPX index options are tighter than single-stock
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CostParams:
    """Transaction cost parameters for a market segment."""
    # Bid-ask half-spread as fraction of option mid-price
    spread_atm_pct: float = 0.02       # ATM: 2% half-spread
    spread_otm_pct: float = 0.08       # OTM: 8% half-spread
    spread_deep_otm_pct: float = 0.20  # Deep OTM: 20% half-spread

    # Market impact (linear model): impact_bps = impact_coeff * sqrt(participation_rate)
    impact_coeff: float = 5.0          # basis points per sqrt(participation)

    # Fixed costs
    commission_per_contract: float = 0.65   # per contract
    exchange_fee_per_contract: float = 0.30
    clearing_fee_per_contract: float = 0.02

    # Equity hedging costs
    equity_spread_bps: float = 1.0     # equity half-spread
    equity_commission_per_share: float = 0.005


# Pre-defined cost profiles
SPX_INDEX_COSTS = CostParams(
    spread_atm_pct=0.015,
    spread_otm_pct=0.05,
    spread_deep_otm_pct=0.12,
    impact_coeff=3.0,
)

LIQUID_SINGLE_STOCK_COSTS = CostParams(
    spread_atm_pct=0.025,
    spread_otm_pct=0.08,
    spread_deep_otm_pct=0.20,
    impact_coeff=5.0,
)

ILLIQUID_SINGLE_STOCK_COSTS = CostParams(
    spread_atm_pct=0.05,
    spread_otm_pct=0.15,
    spread_deep_otm_pct=0.30,
    impact_coeff=10.0,
)


class TransactionCostModel:
    """
    Compute realistic transaction costs for option trades.

    Usage:
        tcm = TransactionCostModel()
        cost = tcm.option_cost(
            option_price=5.50, strike=4500, spot=4500,
            notional=10_000_000, is_index=True
        )
    """

    def __init__(self):
        self.index_costs = SPX_INDEX_COSTS
        self.liquid_costs = LIQUID_SINGLE_STOCK_COSTS
        self.illiquid_costs = ILLIQUID_SINGLE_STOCK_COSTS

    def option_cost(
        self,
        option_mid_price: float,
        strike: float,
        spot: float,
        notional: float,
        is_index: bool = True,
        daily_volume: float = 10000,
    ) -> dict:
        """
        Compute total transaction cost for an option trade.

        Args:
            option_mid_price: Mid-market option price.
            strike: Option strike.
            spot: Current underlying price.
            notional: Dollar notional of the trade.
            is_index: True for index options (tighter spreads).
            daily_volume: Average daily volume in contracts (for impact).

        Returns:
            Dict with cost breakdown.
        """
        params = self.index_costs if is_index else self.liquid_costs

        # Number of contracts (assuming 100 multiplier for equity, 100 for SPX)
        multiplier = 100
        num_contracts = abs(notional / (option_mid_price * multiplier)) if option_mid_price > 0 else 0

        # 1. Bid-ask spread cost
        moneyness = strike / spot if spot > 0 else 1.0
        spread_pct = self._spread_by_moneyness(moneyness, params)
        spread_cost = option_mid_price * spread_pct * num_contracts * multiplier

        # 2. Market impact
        participation = num_contracts / max(daily_volume, 1)
        impact_bps = params.impact_coeff * np.sqrt(min(participation, 1.0))
        impact_cost = notional * impact_bps / 10000

        # 3. Fixed costs
        commission = num_contracts * params.commission_per_contract
        exchange_fees = num_contracts * params.exchange_fee_per_contract
        clearing_fees = num_contracts * params.clearing_fee_per_contract

        total = spread_cost + impact_cost + commission + exchange_fees + clearing_fees

        return {
            "spread_cost": spread_cost,
            "impact_cost": impact_cost,
            "commission": commission,
            "exchange_fees": exchange_fees,
            "clearing_fees": clearing_fees,
            "total_cost": total,
            "total_bps": (total / notional * 10000) if notional > 0 else 0,
            "num_contracts": num_contracts,
            "spread_pct": spread_pct,
            "participation_rate": participation,
        }

    def hedge_cost(
        self,
        shares: float,
        share_price: float,
        params: CostParams | None = None,
    ) -> float:
        """
        Cost of a delta-hedge trade in the underlying equity/future.

        Args:
            shares: Number of shares to trade (absolute value).
            share_price: Current share price.
            params: Cost parameters (default: index costs).

        Returns:
            Total hedge cost in dollars.
        """
        if params is None:
            params = self.index_costs

        shares = abs(shares)
        notional = shares * share_price

        spread = notional * params.equity_spread_bps / 10000
        commission = shares * params.equity_commission_per_share

        return spread + commission

    def portfolio_entry_cost(
        self,
        positions: list[dict],
        spot: float,
        atm_vol: float,
    ) -> dict:
        """
        Compute total cost to enter a portfolio of positions.

        Args:
            positions: List of position dicts from strategy.construct_positions().
            spot: Current spot price.
            atm_vol: ATM implied vol for rough pricing.

        Returns:
            Dict with total cost and per-position breakdown.
        """
        from . import black_scholes as bs

        total_cost = 0.0
        position_costs = []

        for pos in positions:
            notional = abs(pos.get("notional_usd", 0))
            instrument = pos.get("instrument", "")
            is_index = "index" in pos.get("leg", "").lower() or "SPX" in instrument

            # Estimate option price
            if "straddle" in instrument.lower():
                call_price = bs.price(spot, spot, atm_vol, 0.25, self.index_costs.equity_spread_bps / 10000, True)
                put_price = bs.price(spot, spot, atm_vol, 0.25, self.index_costs.equity_spread_bps / 10000, False)
                option_price = call_price + put_price
            elif "put" in instrument.lower():
                option_price = bs.price(spot, spot * 0.95, atm_vol * 1.05, 0.25, 0.045, False)
            elif "call" in instrument.lower():
                option_price = bs.price(spot, spot * 1.10, atm_vol * 0.95, 0.25, 0.045, True)
            else:
                option_price = bs.price(spot, spot, atm_vol, 0.25, 0.045, True)

            cost = self.option_cost(
                option_mid_price=option_price,
                strike=spot,
                spot=spot,
                notional=notional,
                is_index=is_index,
            )

            total_cost += cost["total_cost"]
            position_costs.append({
                "leg": pos.get("leg", ""),
                **cost,
            })

        return {
            "total_cost": total_cost,
            "total_bps_of_notional": (total_cost / sum(abs(p.get("notional_usd", 0)) for p in positions) * 10000)
            if positions else 0,
            "positions": position_costs,
        }

    def _spread_by_moneyness(self, moneyness: float, params: CostParams) -> float:
        """Map moneyness to bid-ask half-spread percentage."""
        otm_distance = abs(moneyness - 1.0)

        if otm_distance <= 0.02:
            return params.spread_atm_pct
        elif otm_distance <= 0.10:
            # Linear interpolation between ATM and OTM
            t = (otm_distance - 0.02) / 0.08
            return params.spread_atm_pct + t * (params.spread_otm_pct - params.spread_atm_pct)
        elif otm_distance <= 0.25:
            t = (otm_distance - 0.10) / 0.15
            return params.spread_otm_pct + t * (params.spread_deep_otm_pct - params.spread_otm_pct)
        else:
            return params.spread_deep_otm_pct
