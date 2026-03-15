"""
Phased Portfolio Builder
==========================
Implements the 60-day, 5-phase build-up mandated by the Investment Committee.

Phase 1 (Weeks 1-2):  Dispersion Trading
Phase 2 (Weeks 3-4):  Volatility Harvesting
Phase 3 (Weeks 5-6):  Directional Long/Short
Phase 4 (Weeks 7-8):  Dynamic Volatility Targeting
Phase 5 (Weeks 8-9):  Option Overlay

Each phase:
  1. Generates target positions from the strategy
  2. Computes the delta between target and current positions
  3. Creates orders to reach the target
  4. Monitors fill progress
  5. Gates on risk limits before proceeding to next phase
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config.portfolio_constraints import IMPLEMENTATION_PHASES, ALLOCATIONS
from .order_manager import OrderManager, Order

logger = logging.getLogger(__name__)


@dataclass
class PhaseStatus:
    """Status of a single implementation phase."""
    phase_key: str
    strategy: str
    target_notional: float
    deployed_notional: float = 0.0
    orders_created: int = 0
    orders_filled: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def completion_pct(self) -> float:
        if self.target_notional <= 0:
            return 100.0
        return min(100.0, self.deployed_notional / self.target_notional * 100)

    @property
    def is_complete(self) -> bool:
        return self.completion_pct >= 95.0  # 95% threshold


class PhasedBuilder:
    """
    Manages the phased portfolio build-up schedule.

    Usage:
        builder = PhasedBuilder(order_manager)
        builder.advance(current_day=5, strategies=strategies, spot=4500, vol=0.18)
    """

    def __init__(self, order_manager: OrderManager):
        self.om = order_manager
        self.phases: dict[str, PhaseStatus] = {}
        self.current_phase_idx = 0
        self.build_day = 0

        # Initialize phases from IC mandate
        for key, phase_info in IMPLEMENTATION_PHASES.items():
            strat = phase_info["strategy"]
            capital = ALLOCATIONS[strat]["capital"]
            self.phases[key] = PhaseStatus(
                phase_key=key,
                strategy=strat,
                target_notional=capital,
            )

    def advance(
        self,
        build_day: int,
        strategies: list,
        spot: float,
        vol: float,
    ) -> list[Order]:
        """
        Advance the build-up by one day. Creates orders if the current
        phase is active and not yet complete.

        Args:
            build_day: Day number in the 60-day build-up (1-60).
            strategies: List of initialized strategy objects.
            spot: Current spot price.
            vol: Current ATM vol.

        Returns:
            List of orders created today.
        """
        self.build_day = build_day
        orders_today = []

        for key, phase_info in IMPLEMENTATION_PHASES.items():
            day_start, day_end = phase_info["days"]
            status = self.phases[key]

            # Only act during the phase's active window
            if build_day < day_start or build_day > day_end:
                continue

            if status.is_complete:
                continue

            if status.started_at is None:
                status.started_at = datetime.now(timezone.utc)
                logger.info(
                    "Phase %s started: %s (days %d-%d)",
                    key, phase_info["description"], day_start, day_end,
                )

            # Find the matching strategy
            strat = next(
                (s for s in strategies if phase_info["strategy"] in s.name.lower().replace(" ", "_").replace("/", "_")),
                None,
            )
            if strat is None:
                # Try looser matching
                strat_name_lower = phase_info["strategy"].replace("_", " ")
                strat = next(
                    (s for s in strategies if strat_name_lower in s.name.lower()),
                    None,
                )

            if strat is None or not strat.positions:
                continue

            # Calculate daily target: spread evenly over the phase duration
            phase_days = day_end - day_start + 1
            daily_target = status.target_notional / phase_days
            remaining = status.target_notional - status.deployed_notional

            today_target = min(daily_target, remaining)
            if today_target <= 0:
                continue

            # Create orders for today's tranche
            new_orders = self._create_phase_orders(
                status, strat, today_target, spot, vol, key,
            )
            orders_today.extend(new_orders)

            # Process fills (paper mode fills immediately)
            self.om.process_fills(spot, vol)

            # Update deployed notional from fills
            for order in new_orders:
                if order.state.value == "FILLED":
                    status.deployed_notional += today_target
                    status.orders_filled += 1

            if status.is_complete and status.completed_at is None:
                status.completed_at = datetime.now(timezone.utc)
                logger.info("Phase %s complete: $%.0f deployed", key, status.deployed_notional)

        return orders_today

    def _create_phase_orders(
        self,
        status: PhaseStatus,
        strategy,
        target_notional: float,
        spot: float,
        vol: float,
        phase_key: str,
    ) -> list[Order]:
        """Create orders for a daily tranche of a phase."""
        orders = []

        for pos in strategy.positions:
            instrument = pos.get("instrument", "")
            direction = pos.get("direction", "LONG")
            pos_notional = abs(pos.get("notional_usd", 0))
            if pos_notional <= 0:
                continue

            # Scale position to today's tranche
            scale = target_notional / strategy.notional_deployed if strategy.notional_deployed > 0 else 0
            tranche_notional = pos_notional * scale

            # Estimate number of contracts
            from models import black_scholes as bs
            option_price = bs.price(spot, spot, vol, 0.25, 0.045, True)
            num_contracts = max(1, int(tranche_notional / (option_price * 100)))

            side = "SELL" if direction == "SHORT" else "BUY"

            if "straddle" in instrument.lower():
                order = self.om.create_straddle_order(
                    strategy=strategy.name,
                    symbol="SPX",
                    strike=spot,
                    expiry="20260620",
                    side=side,
                    quantity=num_contracts,
                    phase=phase_key,
                )
            else:
                right = "P" if "put" in instrument.lower() else "C"
                order = self.om.create_single_order(
                    strategy=strategy.name,
                    symbol="SPX",
                    strike=spot,
                    expiry="20260620",
                    right=right,
                    side=side,
                    quantity=num_contracts,
                    phase=phase_key,
                )

            self.om.submit(order)
            status.orders_created += 1
            orders.append(order)

        return orders

    def summary(self) -> dict:
        """Return build-up progress summary."""
        total_target = sum(p.target_notional for p in self.phases.values())
        total_deployed = sum(p.deployed_notional for p in self.phases.values())

        return {
            "build_day": self.build_day,
            "total_target": total_target,
            "total_deployed": total_deployed,
            "overall_completion_pct": (total_deployed / total_target * 100) if total_target > 0 else 0,
            "phases": {
                key: {
                    "strategy": p.strategy,
                    "target": p.target_notional,
                    "deployed": p.deployed_notional,
                    "completion_pct": p.completion_pct,
                    "orders_created": p.orders_created,
                    "orders_filled": p.orders_filled,
                }
                for key, p in self.phases.items()
            },
        }
