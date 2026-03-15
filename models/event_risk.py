"""
Single-Stock Event Risk Model
================================
Manages earnings calendar and event risk for the dispersion book.

Key features:
  1. Earnings calendar tracking (synthetic or from data provider)
  2. IV crush estimation around earnings
  3. Pre-earnings position sizing adjustment
  4. M&A / corporate action detection signals

For dispersion trading, single-stock events are the #1 risk:
  - Earnings: constituent IV spikes pre-event, crushes post-event
  - M&A: massive gap risk, can blow up a single-name long vol position
  - Dividends: affect forward pricing and put-call parity
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EarningsEvent:
    """A single earnings event."""
    symbol: str
    report_date: str        # YYYY-MM-DD
    report_time: str = "AMC"  # AMC (after market close) or BMO (before market open)
    estimated_move_pct: float = 0.0  # Expected earnings move (from options market)
    actual_move_pct: float | None = None
    iv_pre_event: float | None = None
    iv_post_event: float | None = None


# ---------------------------------------------------------------------------
# Earnings Calendar
# ---------------------------------------------------------------------------

class EarningsCalendar:
    """
    Tracks earnings dates and estimates event impact.

    In production, this would pull from:
      - Earnings Whispers / Wall Street Horizon
      - IBKR corporate events API
      - Yahoo Finance / Alpha Vantage

    For now, generates a synthetic calendar matching the 45-name universe.
    """

    def __init__(self):
        self.events: list[EarningsEvent] = []
        self._by_symbol: dict[str, list[EarningsEvent]] = {}

    def load_synthetic(self, symbols: list[str], start_date: str, end_date: str):
        """
        Generate a synthetic earnings calendar.
        Most companies report quarterly, clustered in Jan/Apr/Jul/Oct.
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Earnings season months (peak reporting)
        earnings_months = [1, 4, 7, 10]

        # Typical staggered move expectations by sector
        sector_moves = {
            "Technology": 0.06,       # 6% expected move
            "Financials": 0.04,
            "Healthcare": 0.05,
            "Consumer": 0.04,
            "Industrials": 0.03,
            "Energy": 0.05,
            "Communication": 0.05,
            "Materials": 0.04,
            "Utilities": 0.02,
            "RealEstate": 0.03,
        }

        np.random.seed(42)

        for symbol in symbols:
            # Determine sector from symbol prefix
            sector = "Technology"  # default
            for s in ["FIN", "HLTH", "CONS", "IND", "ENRG", "COMM", "MAT", "UTIL", "REIT"]:
                if s in symbol.upper():
                    sector_map = {
                        "FIN": "Financials", "HLTH": "Healthcare", "CONS": "Consumer",
                        "IND": "Industrials", "ENRG": "Energy", "COMM": "Communication",
                        "MAT": "Materials", "UTIL": "Utilities", "REIT": "RealEstate",
                    }
                    sector = sector_map.get(s, "Technology")
                    break

            base_move = sector_moves.get(sector, 0.04)

            # Generate quarterly earnings dates
            current = start
            while current <= end:
                if current.month in earnings_months:
                    # Random day in the second half of the month
                    day = np.random.randint(15, 28)
                    try:
                        report_date = current.replace(day=day)
                    except ValueError:
                        report_date = current.replace(day=28)

                    if start <= report_date <= end:
                        move = base_move + np.random.normal(0, 0.01)
                        event = EarningsEvent(
                            symbol=symbol,
                            report_date=report_date.strftime("%Y-%m-%d"),
                            report_time="AMC" if np.random.random() > 0.4 else "BMO",
                            estimated_move_pct=max(0.01, move),
                        )
                        self.events.append(event)

                        if symbol not in self._by_symbol:
                            self._by_symbol[symbol] = []
                        self._by_symbol[symbol].append(event)

                current += timedelta(days=30)

        logger.info("Generated %d earnings events for %d symbols", len(self.events), len(symbols))

    def upcoming_events(self, current_date: str, lookahead_days: int = 5) -> list[EarningsEvent]:
        """Get events in the next N trading days."""
        current = pd.to_datetime(current_date)
        cutoff = current + timedelta(days=lookahead_days)

        return [
            e for e in self.events
            if current <= pd.to_datetime(e.report_date) <= cutoff
        ]

    def has_earnings_soon(self, symbol: str, current_date: str, days: int = 5) -> bool:
        """Check if a symbol has earnings within N days."""
        events = self._by_symbol.get(symbol, [])
        current = pd.to_datetime(current_date)
        cutoff = current + timedelta(days=days)

        return any(
            current <= pd.to_datetime(e.report_date) <= cutoff
            for e in events
        )

    def events_for_symbol(self, symbol: str) -> list[EarningsEvent]:
        """Get all events for a symbol."""
        return self._by_symbol.get(symbol, [])


# ---------------------------------------------------------------------------
# IV Crush Model
# ---------------------------------------------------------------------------

def estimate_iv_crush(
    pre_earnings_iv: float,
    estimated_move_pct: float,
    tenor_days: int = 30,
) -> dict:
    """
    Estimate the implied vol change around an earnings event.

    Pre-earnings: IV inflates to price in the expected move
    Post-earnings: IV crushes as event uncertainty resolves

    The event premium in variance terms:
      event_var = (expected_move)^2 / remaining_fraction

    Args:
        pre_earnings_iv: Current implied vol (annualized).
        estimated_move_pct: Expected earnings move (e.g., 0.06 for 6%).
        tenor_days: Days to expiry of the option.

    Returns:
        Dict with IV crush estimates.
    """
    # Event variance contribution
    # An earnings event contributes ~1 day of extraordinary variance
    event_var_daily = estimated_move_pct ** 2
    normal_var_daily = (pre_earnings_iv ** 2) / 252

    # Pre-earnings: total variance includes event
    # If earnings is in T days, event fraction = 1/T of remaining variance
    if tenor_days > 0:
        event_fraction = 1.0 / tenor_days
        pre_event_total_var = normal_var_daily * (tenor_days - 1) + event_var_daily
        pre_event_iv = np.sqrt(pre_event_total_var / tenor_days * 252)
    else:
        pre_event_iv = pre_earnings_iv

    # Post-earnings: event variance removed
    post_event_total_var = normal_var_daily * max(tenor_days - 1, 1)
    post_event_iv = np.sqrt(post_event_total_var / max(tenor_days - 1, 1) * 252)

    crush_magnitude = pre_event_iv - post_event_iv
    crush_pct = crush_magnitude / pre_event_iv * 100 if pre_event_iv > 0 else 0

    return {
        "pre_event_iv": pre_event_iv,
        "post_event_iv": post_event_iv,
        "crush_magnitude": crush_magnitude,
        "crush_pct": crush_pct,
        "event_var_contribution": event_var_daily,
        "normal_daily_var": normal_var_daily,
    }


# ---------------------------------------------------------------------------
# Position Sizing Adjustment
# ---------------------------------------------------------------------------

def adjust_position_for_earnings(
    base_notional: float,
    symbol: str,
    calendar: EarningsCalendar,
    current_date: str,
    max_reduction_pct: float = 0.50,
) -> tuple[float, str]:
    """
    Adjust position sizing ahead of earnings.

    Rules:
      - 5+ days before: no adjustment
      - 3-5 days before: reduce by 25%
      - 1-2 days before: reduce by 50%
      - Day of: consider full exit

    Args:
        base_notional: Original target notional.
        symbol: Stock symbol.
        calendar: EarningsCalendar instance.
        current_date: Current date string.
        max_reduction_pct: Maximum position reduction.

    Returns:
        (adjusted_notional, reason_string)
    """
    events = calendar._by_symbol.get(symbol, [])
    current = pd.to_datetime(current_date)

    for event in events:
        event_date = pd.to_datetime(event.report_date)
        days_to_event = (event_date - current).days

        if days_to_event < 0 or days_to_event > 5:
            continue

        if days_to_event == 0:
            reduction = max_reduction_pct
            reason = f"Earnings today ({event.report_time})"
        elif days_to_event <= 2:
            reduction = max_reduction_pct * 0.75
            reason = f"Earnings in {days_to_event}d"
        elif days_to_event <= 5:
            reduction = max_reduction_pct * 0.40
            reason = f"Earnings in {days_to_event}d"
        else:
            continue

        adjusted = base_notional * (1 - reduction)
        logger.debug(
            "%s: reducing from $%.0f to $%.0f (%s)",
            symbol, base_notional, adjusted, reason,
        )
        return adjusted, reason

    return base_notional, "No upcoming earnings"
