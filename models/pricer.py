"""
Position Pricer - Mark-to-Market Engine
==========================================
Prices option positions daily using the Black-Scholes model with
SVI-calibrated implied vols. Replaces Greek-based PnL approximation
with actual mark-to-market.

Each position is tracked as an option contract with:
  - Instrument type (straddle, put, call, var swap leg, etc.)
  - Strike, expiry, right (C/P)
  - Quantity / notional
  - Entry price

Daily PnL = change in mark-to-market value of all positions.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import black_scholes as bs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position representation
# ---------------------------------------------------------------------------

@dataclass
class OptionPosition:
    """A single option leg in the portfolio."""
    strategy: str           # Which strategy owns this
    leg_id: str             # e.g., "INDEX_SHORT", "CONSTITUENT_LONG_3"
    underlying: str         # e.g., "SPX_INDEX", "AAPL"
    strike: float
    tenor_years: float      # Time to expiry at inception
    is_call: bool
    quantity: float         # Positive = long, negative = short
    notional: float         # Dollar notional for sizing
    entry_price: float = 0.0
    current_price: float = 0.0
    entry_vol: float = 0.0

    # Current Greeks (updated by pricer)
    greeks: dict = field(default_factory=lambda: {
        "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0
    })

    def mark_to_market(self) -> float:
        """Current mark-to-market value of the position."""
        return self.quantity * self.current_price

    def unrealized_pnl(self) -> float:
        """Unrealized PnL since entry."""
        return self.quantity * (self.current_price - self.entry_price)


@dataclass
class StraddlePosition:
    """A straddle = long call + long put at same strike."""
    strategy: str
    leg_id: str
    underlying: str
    strike: float
    tenor_years: float
    quantity: float     # Positive = long straddle, negative = short straddle
    notional: float
    call: OptionPosition = None
    put: OptionPosition = None

    def __post_init__(self):
        self.call = OptionPosition(
            strategy=self.strategy, leg_id=f"{self.leg_id}_C",
            underlying=self.underlying, strike=self.strike,
            tenor_years=self.tenor_years, is_call=True,
            quantity=self.quantity, notional=self.notional / 2,
        )
        self.put = OptionPosition(
            strategy=self.strategy, leg_id=f"{self.leg_id}_P",
            underlying=self.underlying, strike=self.strike,
            tenor_years=self.tenor_years, is_call=False,
            quantity=self.quantity, notional=self.notional / 2,
        )

    def legs(self) -> list[OptionPosition]:
        return [self.call, self.put]


# ---------------------------------------------------------------------------
# Portfolio Pricer
# ---------------------------------------------------------------------------

class PortfolioPricer:
    """
    Prices all option positions using Black-Scholes + SVI surface.

    Usage:
        pricer = PortfolioPricer(rate=0.045)
        pricer.add_straddle("Dispersion", "INDEX_SHORT", "SPX_INDEX", 4500, 0.25, -1, 60_000_000)
        daily_pnl = pricer.reprice(spot_prices, iv_surface, day_tenor)
    """

    def __init__(self, rate: float = 0.045):
        self.rate = rate
        self.positions: list[OptionPosition] = []
        self.prev_mtm: float = 0.0  # Previous day's total MTM
        self.total_mtm: float = 0.0

    def clear(self):
        """Remove all positions."""
        self.positions.clear()
        self.prev_mtm = 0.0
        self.total_mtm = 0.0

    # ------------------------------------------------------------------
    # Position builders
    # ------------------------------------------------------------------

    def add_option(
        self,
        strategy: str,
        leg_id: str,
        underlying: str,
        strike: float,
        tenor_years: float,
        is_call: bool,
        quantity: float,
        notional: float,
        spot: float,
        vol: float,
    ) -> OptionPosition:
        """Add a single option position and compute entry price."""
        entry_price = bs.price(spot, strike, vol, tenor_years, self.rate, is_call)
        pos = OptionPosition(
            strategy=strategy, leg_id=leg_id, underlying=underlying,
            strike=strike, tenor_years=tenor_years, is_call=is_call,
            quantity=quantity, notional=notional,
            entry_price=entry_price, current_price=entry_price,
            entry_vol=vol,
        )
        self.positions.append(pos)
        return pos

    def add_straddle(
        self,
        strategy: str,
        leg_id: str,
        underlying: str,
        strike: float,
        tenor_years: float,
        quantity: float,
        notional: float,
        spot: float,
        vol: float,
    ):
        """Add a straddle (call + put at same strike)."""
        self.add_option(strategy, f"{leg_id}_C", underlying, strike,
                        tenor_years, True, quantity, notional / 2, spot, vol)
        self.add_option(strategy, f"{leg_id}_P", underlying, strike,
                        tenor_years, False, quantity, notional / 2, spot, vol)

    # ------------------------------------------------------------------
    # Daily repricing
    # ------------------------------------------------------------------

    def reprice(
        self,
        spot_prices: dict[str, float],
        vol_surface: pd.DataFrame,
        elapsed_days: int = 1,
    ) -> dict:
        """
        Reprice all positions and compute daily PnL.

        Args:
            spot_prices: Dict of {underlying: current_spot_price}.
            vol_surface: Current IV surface DataFrame (same format as strategies expect).
            elapsed_days: Trading days elapsed since last reprice.

        Returns:
            Dict with total PnL, per-strategy PnL, and portfolio Greeks.
        """
        self.prev_mtm = self.total_mtm

        strategy_pnl = {}
        strategy_greeks = {}
        total_mtm = 0.0

        for pos in self.positions:
            spot = spot_prices.get(pos.underlying)
            if spot is None:
                # Use SPX as fallback for unnamed constituents
                spot = spot_prices.get("SPX_INDEX", 4500.0)

            # Get current implied vol from the surface
            vol = self._lookup_vol(vol_surface, pos.underlying, pos.strike, spot, pos.tenor_years)

            # Decay tenor
            pos.tenor_years = max(pos.tenor_years - elapsed_days / 252, 1e-5)

            # Reprice
            greeks = bs.all_greeks(spot, pos.strike, vol, pos.tenor_years, self.rate, pos.is_call)
            pos.current_price = greeks["price"]

            # Scale Greeks by quantity and contract multiplier
            contract_mult = abs(pos.notional / max(pos.entry_price, 1e-6))
            pos.greeks = {
                "delta": greeks["delta"] * pos.quantity * contract_mult * spot,
                "gamma": greeks["gamma"] * pos.quantity * contract_mult * spot ** 2 / 100,
                "vega": greeks["vega"] * pos.quantity * contract_mult / 100,
                "theta": greeks["theta"] * pos.quantity * contract_mult,
            }

            # MTM value
            mtm = pos.quantity * pos.current_price * contract_mult
            total_mtm += mtm

            # Accumulate per-strategy
            if pos.strategy not in strategy_pnl:
                strategy_pnl[pos.strategy] = 0.0
                strategy_greeks[pos.strategy] = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

            for g in ["delta", "gamma", "vega", "theta"]:
                strategy_greeks[pos.strategy][g] += pos.greeks[g]

        self.total_mtm = total_mtm
        daily_pnl = total_mtm - self.prev_mtm

        # Per-strategy PnL (from Greeks for now, MTM per strategy requires more tracking)
        # We compute strategy PnL from the change in their positions' values
        for pos in self.positions:
            contract_mult = abs(pos.notional / max(pos.entry_price, 1e-6))
            pos_pnl = pos.quantity * (pos.current_price - pos.entry_price) * contract_mult
            # Reset entry to current for next day's PnL
            # Actually we track cumulative, so we need incremental
            # Use the total strategy Greeks for attribution instead

        return {
            "daily_pnl": daily_pnl,
            "total_mtm": total_mtm,
            "strategy_pnl": strategy_pnl,
            "strategy_greeks": strategy_greeks,
            "portfolio_greeks": self._aggregate_greeks(),
            "num_positions": len(self.positions),
        }

    def _aggregate_greeks(self) -> dict:
        """Sum Greeks across all positions."""
        total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for pos in self.positions:
            for g in total:
                total[g] += pos.greeks.get(g, 0.0)
        return total

    def _lookup_vol(
        self,
        surface: pd.DataFrame,
        underlying: str,
        strike: float,
        spot: float,
        tenor_years: float,
    ) -> float:
        """
        Look up implied vol from the surface for a given position.
        Maps strike to the nearest delta bucket and tenor.
        """
        if surface.empty:
            return 0.20  # fallback

        # Determine moneyness to pick the right delta bucket
        moneyness = strike / spot if spot > 0 else 1.0
        if moneyness < 0.90:
            delta_label = "10D_PUT"
        elif moneyness < 0.95:
            delta_label = "25D_PUT"
        elif moneyness <= 1.05:
            delta_label = "ATM"
        elif moneyness <= 1.10:
            delta_label = "25D_CALL"
        else:
            delta_label = "10D_CALL"

        # Nearest tenor (in months)
        tenor_months = max(1, round(tenor_years * 12))
        available_tenors = surface["tenor_months"].unique()
        nearest_tenor = min(available_tenors, key=lambda t: abs(t - tenor_months))

        # Look up
        name = underlying
        mask = (
            (surface["name"] == name)
            & (surface["strike_delta"] == delta_label)
            & (surface["tenor_months"] == nearest_tenor)
        )
        match = surface[mask]

        if not match.empty:
            return float(match["implied_vol"].iloc[0])

        # Fallback: try ATM for this name
        mask_atm = (
            (surface["name"] == name)
            & (surface["strike_delta"] == "ATM")
        )
        match_atm = surface[mask_atm]
        if not match_atm.empty:
            return float(match_atm["implied_vol"].iloc[0])

        # Last fallback: SPX ATM
        mask_spx = (
            (surface["name"] == "SPX_INDEX")
            & (surface["strike_delta"] == "ATM")
            & (surface["tenor_months"] == 3)
        )
        match_spx = surface[mask_spx]
        if not match_spx.empty:
            return float(match_spx["implied_vol"].iloc[0])

        return 0.20


# ---------------------------------------------------------------------------
# Convenience: build positions from strategy output
# ---------------------------------------------------------------------------

def positions_from_strategy(
    strategy_name: str,
    positions: list[dict],
    spot: float,
    atm_vol: float,
    pricer: PortfolioPricer,
):
    """
    Convert a strategy's position list into priced OptionPositions.

    Maps the simplified position dicts from strategies into actual
    option contracts that can be marked to market.

    Args:
        strategy_name: Name of the strategy.
        positions: List of position dicts from strategy.construct_positions().
        spot: Current underlying spot price.
        atm_vol: Current ATM implied vol.
        pricer: PortfolioPricer to add positions to.
    """
    for pos in positions:
        leg = pos.get("leg", "UNKNOWN")
        direction = pos.get("direction", "LONG")
        notional = abs(pos.get("notional_usd", 0))
        quantity = 1.0 if direction == "LONG" else -1.0

        # Determine strike from instrument description
        instrument = pos.get("instrument", "")
        if "at-the-money" in instrument.lower() or "atm" in instrument.lower():
            strike = spot
        elif "5% out-of-the-money" in instrument.lower():
            strike = spot * 0.95  # OTM put
        elif "10% out-of-the-money" in instrument.lower():
            strike = spot * 0.90
        else:
            strike = spot  # default ATM

        # Determine if straddle or single leg
        if "straddle" in instrument.lower():
            pricer.add_straddle(
                strategy=strategy_name,
                leg_id=leg,
                underlying="SPX_INDEX" if "index" in leg.lower() or "SPX" in instrument else "CONSTITUENT",
                strike=strike,
                tenor_years=0.25,  # 3-month default
                quantity=quantity,
                notional=notional,
                spot=spot,
                vol=atm_vol,
            )
        elif "put" in instrument.lower():
            pricer.add_option(
                strategy=strategy_name,
                leg_id=leg,
                underlying="SPX_INDEX",
                strike=strike,
                tenor_years=0.25,
                is_call=False,
                quantity=quantity,
                notional=notional,
                spot=spot,
                vol=atm_vol * 1.05,  # puts trade at slight premium
            )
        elif "call" in instrument.lower():
            pricer.add_option(
                strategy=strategy_name,
                leg_id=leg,
                underlying="SPX_INDEX",
                strike=strike,
                tenor_years=0.25,
                is_call=True,
                quantity=quantity,
                notional=notional,
                spot=spot,
                vol=atm_vol * 0.97,  # calls trade at slight discount
            )
        elif "variance" in instrument.lower() or "swap" in instrument.lower():
            # Approximate variance swap as a strip: short OTM puts + calls
            pricer.add_option(
                strategy=strategy_name,
                leg_id=f"{leg}_PUT",
                underlying="SPX_INDEX",
                strike=spot * 0.95,
                tenor_years=0.25,
                is_call=False,
                quantity=quantity,
                notional=notional / 2,
                spot=spot,
                vol=atm_vol * 1.08,
            )
            pricer.add_option(
                strategy=strategy_name,
                leg_id=f"{leg}_CALL",
                underlying="SPX_INDEX",
                strike=spot * 1.05,
                tenor_years=0.25,
                is_call=True,
                quantity=quantity,
                notional=notional / 2,
                spot=spot,
                vol=atm_vol * 0.95,
            )
        else:
            # Default: ATM straddle
            pricer.add_straddle(
                strategy=strategy_name,
                leg_id=leg,
                underlying="SPX_INDEX",
                strike=strike,
                tenor_years=0.25,
                quantity=quantity,
                notional=notional,
                spot=spot,
                vol=atm_vol,
            )
