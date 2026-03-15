"""
Order Manager
===============
Manages the full order lifecycle: creation, validation, routing,
fill tracking, and position reconciliation.

Order states: PENDING -> SUBMITTED -> PARTIAL -> FILLED | CANCELLED | REJECTED

Supports:
  - Single-leg orders (calls, puts)
  - Multi-leg spreads (straddles, strangles, collars, var swap strips)
  - Phased execution (60-day IC mandate build-up)
  - Paper mode (simulated fills) and live mode (IBKR routing)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class OrderState(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    MID = "MID"             # Mid-price limit
    SPREAD = "SPREAD"       # Multi-leg spread order


@dataclass
class OrderLeg:
    """A single leg of an order."""
    symbol: str
    strike: float
    expiry: str             # YYYYMMDD
    right: str              # C or P
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None


@dataclass
class Order:
    """A complete order (single or multi-leg)."""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    strategy: str = ""
    legs: list[OrderLeg] = field(default_factory=list)
    state: OrderState = OrderState.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_at: datetime | None = None
    filled_at: datetime | None = None

    # Fill details
    fill_price: float = 0.0
    fill_quantity: int = 0
    commission: float = 0.0
    slippage_bps: float = 0.0

    # Execution metadata
    phase: str = ""         # PHASE_1 through PHASE_5
    urgency: str = "NORMAL" # NORMAL, URGENT, PASSIVE
    notes: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED)

    @property
    def is_multi_leg(self) -> bool:
        return len(self.legs) > 1

    @property
    def net_quantity(self) -> int:
        return sum(
            leg.quantity if leg.side == OrderSide.BUY else -leg.quantity
            for leg in self.legs
        )


class OrderManager:
    """
    Manages orders from creation through fill.

    Usage:
        om = OrderManager(mode="paper")
        order = om.create_straddle_order("Dispersion", "SPX", 4500, "20260620", "SELL", 50)
        om.submit(order)
        om.process_fills()  # In paper mode, fills immediately
    """

    def __init__(self, mode: str = "paper"):
        """
        Args:
            mode: "paper" for simulated fills, "live" for IBKR routing.
        """
        self.mode = mode
        self.orders: dict[str, Order] = {}
        self.fill_history: list[dict] = []
        self._ibkr_client = None

    def set_ibkr_client(self, client):
        """Attach IBKR client for live order routing."""
        self._ibkr_client = client
        self.mode = "live"

    # ------------------------------------------------------------------
    # Order creation
    # ------------------------------------------------------------------

    def create_single_order(
        self,
        strategy: str,
        symbol: str,
        strike: float,
        expiry: str,
        right: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: float | None = None,
        phase: str = "",
    ) -> Order:
        """Create a single-leg option order."""
        leg = OrderLeg(
            symbol=symbol, strike=strike, expiry=expiry, right=right,
            side=OrderSide[side], quantity=quantity,
            order_type=OrderType[order_type], limit_price=limit_price,
        )
        order = Order(strategy=strategy, legs=[leg], phase=phase)
        self.orders[order.order_id] = order
        logger.info("Created order %s: %s %s %s %.0f%s x%d",
                     order.order_id, side, symbol, right, strike, expiry, quantity)
        return order

    def create_straddle_order(
        self,
        strategy: str,
        symbol: str,
        strike: float,
        expiry: str,
        side: str,
        quantity: int,
        phase: str = "",
    ) -> Order:
        """Create a straddle order (call + put at same strike)."""
        order_side = OrderSide[side]
        call_leg = OrderLeg(
            symbol=symbol, strike=strike, expiry=expiry, right="C",
            side=order_side, quantity=quantity, order_type=OrderType.SPREAD,
        )
        put_leg = OrderLeg(
            symbol=symbol, strike=strike, expiry=expiry, right="P",
            side=order_side, quantity=quantity, order_type=OrderType.SPREAD,
        )
        order = Order(strategy=strategy, legs=[call_leg, put_leg], phase=phase)
        self.orders[order.order_id] = order
        logger.info("Created straddle %s: %s %s %.0f %s x%d",
                     order.order_id, side, symbol, strike, expiry, quantity)
        return order

    def create_collar_order(
        self,
        strategy: str,
        symbol: str,
        put_strike: float,
        call_strike: float,
        expiry: str,
        quantity: int,
        phase: str = "",
    ) -> Order:
        """Create a collar order (long put + short call)."""
        put_leg = OrderLeg(
            symbol=symbol, strike=put_strike, expiry=expiry, right="P",
            side=OrderSide.BUY, quantity=quantity, order_type=OrderType.SPREAD,
        )
        call_leg = OrderLeg(
            symbol=symbol, strike=call_strike, expiry=expiry, right="C",
            side=OrderSide.SELL, quantity=quantity, order_type=OrderType.SPREAD,
        )
        order = Order(strategy=strategy, legs=[put_leg, call_leg], phase=phase)
        self.orders[order.order_id] = order
        logger.info("Created collar %s: %s put %.0f / call %.0f %s x%d",
                     order.order_id, symbol, put_strike, call_strike, expiry, quantity)
        return order

    # ------------------------------------------------------------------
    # Order submission and fill processing
    # ------------------------------------------------------------------

    def submit(self, order: Order) -> bool:
        """Submit an order for execution."""
        if order.is_terminal:
            logger.warning("Cannot submit terminal order %s", order.order_id)
            return False

        order.state = OrderState.SUBMITTED
        order.submitted_at = datetime.now(timezone.utc)

        if self.mode == "paper":
            return True
        elif self.mode == "live" and self._ibkr_client:
            return self._submit_to_ibkr(order)
        else:
            logger.error("No IBKR client for live mode")
            order.state = OrderState.REJECTED
            return False

    def submit_all_pending(self) -> int:
        """Submit all pending orders. Returns count submitted."""
        count = 0
        for order in self.orders.values():
            if order.state == OrderState.PENDING:
                if self.submit(order):
                    count += 1
        return count

    def process_fills(self, spot: float = 0.0, vol: float = 0.20) -> list[dict]:
        """
        Process fills for submitted orders.

        In paper mode, fills immediately with simulated prices.
        In live mode, checks IBKR for fill status.

        Returns list of fill records.
        """
        fills = []

        for order in self.orders.values():
            if order.state != OrderState.SUBMITTED:
                continue

            if self.mode == "paper":
                fill = self._simulate_fill(order, spot, vol)
                fills.append(fill)
            elif self.mode == "live":
                fill = self._check_ibkr_fill(order)
                if fill:
                    fills.append(fill)

        self.fill_history.extend(fills)
        return fills

    def cancel(self, order_id: str) -> bool:
        """Cancel a pending or submitted order."""
        order = self.orders.get(order_id)
        if not order or order.is_terminal:
            return False
        order.state = OrderState.CANCELLED
        logger.info("Cancelled order %s", order_id)
        return True

    def cancel_all(self) -> int:
        """Cancel all non-terminal orders."""
        count = 0
        for order in self.orders.values():
            if not order.is_terminal:
                order.state = OrderState.CANCELLED
                count += 1
        return count

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def open_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if not o.is_terminal]

    def filled_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.state == OrderState.FILLED]

    def orders_by_strategy(self, strategy: str) -> list[Order]:
        return [o for o in self.orders.values() if o.strategy == strategy]

    def orders_by_phase(self, phase: str) -> list[Order]:
        return [o for o in self.orders.values() if o.phase == phase]

    def total_commission(self) -> float:
        return sum(o.commission for o in self.filled_orders())

    def total_slippage_bps(self) -> float:
        filled = self.filled_orders()
        if not filled:
            return 0.0
        return sum(o.slippage_bps for o in filled) / len(filled)

    # ------------------------------------------------------------------
    # Paper fill simulation
    # ------------------------------------------------------------------

    def _simulate_fill(self, order: Order, spot: float, vol: float) -> dict:
        """Simulate a fill with realistic slippage."""
        from models import black_scholes as bs

        total_premium = 0.0
        for leg in order.legs:
            is_call = leg.right == "C"
            mid_price = bs.price(spot, leg.strike, vol, 0.25, 0.045, is_call)

            # Apply half-spread slippage
            moneyness = abs(leg.strike / spot - 1.0) if spot > 0 else 0
            if moneyness < 0.02:
                spread_pct = 0.015
            elif moneyness < 0.10:
                spread_pct = 0.05
            else:
                spread_pct = 0.12

            if leg.side == OrderSide.BUY:
                fill_price = mid_price * (1 + spread_pct)
            else:
                fill_price = mid_price * (1 - spread_pct)

            total_premium += fill_price * leg.quantity * (1 if leg.side == OrderSide.BUY else -1)

        # Commission: $0.65 per contract per leg
        total_contracts = sum(leg.quantity for leg in order.legs)
        commission = total_contracts * 0.65

        slippage_bps = spread_pct * 10000

        order.state = OrderState.FILLED
        order.filled_at = datetime.now(timezone.utc)
        order.fill_price = total_premium
        order.fill_quantity = total_contracts
        order.commission = commission
        order.slippage_bps = slippage_bps

        fill_record = {
            "order_id": order.order_id,
            "strategy": order.strategy,
            "phase": order.phase,
            "num_legs": len(order.legs),
            "total_contracts": total_contracts,
            "fill_premium": total_premium,
            "commission": commission,
            "slippage_bps": slippage_bps,
            "filled_at": order.filled_at.isoformat(),
        }

        logger.info("Filled %s: %d contracts, premium $%.0f, comm $%.2f",
                     order.order_id, total_contracts, total_premium, commission)
        return fill_record

    def _submit_to_ibkr(self, order: Order) -> bool:
        """Submit order to IBKR. Placeholder for live integration."""
        logger.info("Would submit %s to IBKR (not implemented)", order.order_id)
        return False

    def _check_ibkr_fill(self, order: Order) -> dict | None:
        """Check IBKR for fill status. Placeholder for live integration."""
        return None
