"""
Margin Model
===============
Estimates portfolio margin requirements for the combined options book.

Implements a simplified SPAN-style margin calculation:
  1. Scanning risk: worst-case loss across price/vol scenarios
  2. Inter-month spread credit
  3. Short option minimum
  4. Net portfolio margin with cross-strategy offsets

Also supports portfolio margin (Reg T) estimation.

Margin cap from IC mandate: $180,000,000
"""

import logging
from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs

logger = logging.getLogger(__name__)

MARGIN_CAP = 180_000_000  # IC mandate


@dataclass
class MarginResult:
    """Margin computation result."""
    scanning_risk: float = 0.0
    inter_spread_credit: float = 0.0
    short_option_minimum: float = 0.0
    net_margin: float = 0.0
    margin_cap: float = MARGIN_CAP
    utilization_pct: float = 0.0
    within_cap: bool = True
    by_strategy: dict = None

    def __post_init__(self):
        if self.by_strategy is None:
            self.by_strategy = {}


# SPAN scan ranges (price move x vol move scenarios)
SPAN_SCENARIOS = [
    # (price_pct_move, vol_move_pts)
    (-0.08, +0.04),   # -8% price, +4 vol pts
    (-0.06, +0.03),
    (-0.04, +0.02),
    (-0.02, +0.01),
    (+0.00, +0.00),
    (+0.02, -0.01),
    (+0.04, -0.02),
    (+0.06, -0.03),
    (+0.08, -0.04),
    # Extreme scenarios (weighted at 35%)
    (-0.12, +0.06),
    (+0.12, -0.06),
    (-0.15, +0.08),
    (+0.15, -0.08),
]


def compute_margin(
    positions: list[dict],
    spot: float,
    atm_vol: float,
    rate: float = 0.045,
) -> MarginResult:
    """
    Compute SPAN-style margin for a portfolio of option positions.

    Args:
        positions: List of position dicts from strategies.
        spot: Current underlying spot price.
        atm_vol: Current ATM implied vol.
        rate: Risk-free rate.

    Returns:
        MarginResult with margin breakdown.
    """
    if not positions:
        return MarginResult()

    # Compute current portfolio value
    portfolio_value = 0.0
    strategy_margins = {}

    for pos in positions:
        strategy = pos.get("strategy", "Unknown")
        notional = abs(pos.get("notional_usd", 0))
        direction = pos.get("direction", "LONG")
        instrument = pos.get("instrument", "")
        vega = pos.get("vega_usd", 0)

        sign = 1.0 if direction == "LONG" else -1.0

        # Estimate margin for this position
        pos_margin = _position_margin(
            instrument, notional, spot, atm_vol, sign, vega, rate,
        )

        if strategy not in strategy_margins:
            strategy_margins[strategy] = 0.0
        strategy_margins[strategy] += pos_margin
        portfolio_value += pos_margin

    # Scanning risk: worst-case loss across SPAN scenarios
    scanning_risk = _scanning_risk(positions, spot, atm_vol, rate)

    # Inter-month spread credit (offsetting positions reduce margin)
    spread_credit = _spread_credit(positions, spot)

    # Short option minimum: max(short_notional * 0.10, short_premium)
    short_minimum = _short_option_minimum(positions, spot, atm_vol, rate)

    # Net margin: max of scanning risk and short minimum, minus credits
    net_margin = max(scanning_risk - spread_credit, short_minimum)

    utilization = net_margin / MARGIN_CAP * 100 if MARGIN_CAP > 0 else 0

    result = MarginResult(
        scanning_risk=scanning_risk,
        inter_spread_credit=spread_credit,
        short_option_minimum=short_minimum,
        net_margin=net_margin,
        utilization_pct=utilization,
        within_cap=net_margin <= MARGIN_CAP,
        by_strategy=strategy_margins,
    )

    logger.debug(
        "Margin: scan=$%.0f credit=$%.0f min=$%.0f net=$%.0f (%.1f%% of cap)",
        scanning_risk, spread_credit, short_minimum, net_margin, utilization,
    )

    return result


def _position_margin(
    instrument: str, notional: float, spot: float,
    vol: float, sign: float, vega: float, rate: float,
) -> float:
    """Estimate margin for a single position."""
    # Short options: margin = max(premium + OTM_amount, minimum_margin)
    # Long options: margin = premium paid (debit)
    if sign < 0:  # Short position
        # Short margin approximation:
        # Broad-based index: 15% of underlying - OTM amount + premium
        # Single stock: 20% of underlying - OTM amount + premium
        base_pct = 0.15  # Index
        margin = notional * base_pct

        # Adjust for vol level (higher vol = higher margin)
        vol_adj = max(0.8, min(1.5, vol / 0.20))
        margin *= vol_adj

        return margin
    else:
        # Long options: margin = premium (fully paid)
        # Premium is roughly vega * vol (very rough)
        premium = abs(vega) * vol * 100 if vega != 0 else notional * 0.05
        return min(premium, notional * 0.10)


def _scanning_risk(
    positions: list[dict], spot: float, vol: float, rate: float,
) -> float:
    """
    Compute worst-case portfolio loss across SPAN scenarios.
    """
    worst_loss = 0.0

    for price_pct, vol_move in SPAN_SCENARIOS:
        scenario_spot = spot * (1 + price_pct)
        scenario_vol = max(0.05, vol + vol_move)

        # Weight extreme scenarios at 35%
        weight = 0.35 if abs(price_pct) > 0.10 else 1.0

        scenario_loss = 0.0
        for pos in positions:
            notional = abs(pos.get("notional_usd", 0))
            direction = pos.get("direction", "LONG")
            sign = 1.0 if direction == "LONG" else -1.0

            instrument = pos.get("instrument", "").lower()

            if "straddle" in instrument:
                # Straddle PnL under scenario
                call_pnl = bs.price(scenario_spot, spot, scenario_vol, 0.25, rate, True) - \
                            bs.price(spot, spot, vol, 0.25, rate, True)
                put_pnl = bs.price(scenario_spot, spot, scenario_vol, 0.25, rate, False) - \
                           bs.price(spot, spot, vol, 0.25, rate, False)
                pnl = (call_pnl + put_pnl) * sign
            elif "put" in instrument:
                pnl = (bs.price(scenario_spot, spot * 0.95, scenario_vol, 0.25, rate, False) -
                       bs.price(spot, spot * 0.95, vol, 0.25, rate, False)) * sign
            elif "call" in instrument:
                pnl = (bs.price(scenario_spot, spot * 1.05, scenario_vol, 0.25, rate, True) -
                       bs.price(spot, spot * 1.05, vol, 0.25, rate, True)) * sign
            else:
                # Generic: use vega and delta
                vega_usd = pos.get("vega_usd", 0)
                delta_usd = pos.get("delta_usd", 0)
                pnl = delta_usd * price_pct + vega_usd * vol_move

            # Scale by notional ratio
            scale = notional / max(spot, 1) * 0.01
            scenario_loss += pnl * scale * weight

        worst_loss = max(worst_loss, abs(scenario_loss))

    return worst_loss


def _spread_credit(positions: list[dict], spot: float) -> float:
    """
    Compute inter-strategy offset credit.
    Long positions offset short positions in the same underlying.
    """
    long_notional = sum(
        abs(p.get("notional_usd", 0))
        for p in positions if p.get("direction") == "LONG"
    )
    short_notional = sum(
        abs(p.get("notional_usd", 0))
        for p in positions if p.get("direction") == "SHORT"
    )

    # Credit = overlap * 70% (not full offset due to basis risk)
    overlap = min(long_notional, short_notional)
    return overlap * 0.05  # 5% of overlap as margin credit


def _short_option_minimum(
    positions: list[dict], spot: float, vol: float, rate: float,
) -> float:
    """
    Minimum margin for short option positions.
    Typically: $250 per contract or 10% of notional, whichever is greater.
    """
    total_min = 0.0
    for pos in positions:
        if pos.get("direction") != "SHORT":
            continue
        notional = abs(pos.get("notional_usd", 0))
        # 10% of notional as minimum margin
        total_min += notional * 0.10

    return total_min


def print_margin_report(result: MarginResult):
    """Print formatted margin report."""
    print("\n" + "=" * 60)
    print("  MARGIN ANALYSIS")
    print("=" * 60)
    print(f"\n  Scanning Risk:        ${result.scanning_risk:>14,.0f}")
    print(f"  Spread Credit:        ${result.inter_spread_credit:>14,.0f}")
    print(f"  Short Option Min:     ${result.short_option_minimum:>14,.0f}")
    print(f"  {'=' * 40}")
    print(f"  Net Margin Required:  ${result.net_margin:>14,.0f}")
    print(f"  Margin Cap:           ${result.margin_cap:>14,.0f}")
    print(f"  Utilization:          {result.utilization_pct:>13.1f}%")
    print(f"  Status:               {'PASS' if result.within_cap else 'BREACH'}")

    if result.by_strategy:
        print(f"\n  {'Strategy':<35} {'Margin':>14}")
        print(f"  {'-' * 49}")
        for strat, margin in sorted(result.by_strategy.items(), key=lambda x: -x[1]):
            print(f"  {strat:<35} ${margin:>13,.0f}")

    print(f"\n{'=' * 60}")
