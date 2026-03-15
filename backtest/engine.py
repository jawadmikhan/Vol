"""
Backtest Engine — Walk-Forward Simulation
==========================================
Replays historical data day-by-day through the strategy pipeline,
computing daily PnL via Greeks-based attribution.

The engine:
  1. Walks forward through vol_regime_signals dates
  2. For each day, constructs a data dict (IV surface scaled by VIX, rolling correlations)
  3. Feeds data to strategies → signals → positions → Greeks
  4. Computes daily PnL from Greek exposures × market moves
  5. Enforces risk limits (vega bounds, drawdown stops)
  6. Records everything for analytics

Usage:
    from backtest.engine import BacktestEngine
    engine = BacktestEngine()
    results = engine.run()
"""

import sys
import os
import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.portfolio_constraints import (
    ALLOCATIONS, TOTAL_CAPITAL, NET_VEGA_FLOOR, NET_VEGA_CEILING,
    GROSS_NOTIONAL_CAP, VOL_HARVEST_SHORT_VARIANCE_NOTIONAL_CAP,
    MAX_PORTFOLIO_DRAWDOWN,
)
from strategies.dispersion import DispersionStrategy
from strategies.volatility_harvesting import VolatilityHarvestingStrategy
from strategies.directional_long_short import DirectionalLongShortStrategy
from strategies.dynamic_vol_targeting import DynamicVolTargetingStrategy
from strategies.option_overlay import OptionOverlayStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Daily snapshot — one row per day in the results
# ---------------------------------------------------------------------------

@dataclass
class DailySnapshot:
    date: str
    day_index: int

    # Market state
    vix: float = 0.0
    vix_change: float = 0.0
    realized_vol_20d: float = 0.0
    rv_change: float = 0.0
    regime: str = "TRANSITIONAL"
    term_slope: float = 0.0
    avg_correlation: float = 0.0
    corr_change: float = 0.0

    # Portfolio PnL (daily)
    vega_pnl: float = 0.0
    gamma_pnl: float = 0.0
    theta_pnl: float = 0.0
    correlation_pnl: float = 0.0
    total_pnl: float = 0.0

    # Per-strategy PnL
    strategy_pnl: dict = field(default_factory=dict)

    # Portfolio Greeks (end of day)
    net_vega: float = 0.0
    net_gamma: float = 0.0
    net_delta: float = 0.0
    net_theta: float = 0.0
    gross_notional: float = 0.0

    # Risk
    cumulative_pnl: float = 0.0
    peak_pnl: float = 0.0
    drawdown: float = 0.0
    drawdown_pct: float = 0.0
    vega_within_bounds: bool = True
    notional_within_cap: bool = True

    # Positions
    num_positions: int = 0
    active_strategies: int = 0


class BacktestEngine:
    """
    Walk-forward backtester for the systematic volatility portfolio.

    Processes historical regime signals day-by-day, re-running strategy
    signals and constructing positions at a configurable rebalance frequency.
    Daily PnL is computed via Greeks × market moves (attribution model).
    """

    def __init__(
        self,
        rebalance_days: int = 5,
        initial_capital: float = TOTAL_CAPITAL,
        suppress_prints: bool = True,
    ):
        """
        Args:
            rebalance_days: Re-run strategy signals every N days (default weekly).
            initial_capital: Starting capital (default $250M from constraints).
            suppress_prints: Suppress strategy print statements during backtest.
        """
        self.rebalance_days = rebalance_days
        self.initial_capital = initial_capital
        self.suppress_prints = suppress_prints

    def run(self, data: dict) -> pd.DataFrame:
        """
        Execute the full walk-forward backtest.

        Args:
            data: The standard data dict from load_reference_data() or LiveDataAdapter.

        Returns:
            DataFrame with one row per trading day, containing all DailySnapshot fields.
        """
        regime_df = data["vol_regime_signals"].copy()
        rv_history = data["realized_vol_history"].copy()
        base_iv_surface = data["implied_vol_surface"].copy()
        base_corr_matrix = data["correlation_matrix"].copy()

        # Parse dates
        regime_df["date"] = pd.to_datetime(regime_df["date"])
        if "date" in rv_history.columns:
            rv_history["date"] = pd.to_datetime(rv_history["date"])
            rv_history.set_index("date", inplace=True)

        dates = regime_df["date"].tolist()
        n_days = len(dates)

        logger.info("Backtest: %d days from %s to %s", n_days, dates[0].date(), dates[-1].date())

        # Initialize strategies (will be rebuilt on rebalance days)
        strategies = None
        snapshots = []
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        prev_vix = None
        prev_rv = None
        prev_avg_corr = None
        stopped_out = False

        for i, date in enumerate(dates):
            row = regime_df.iloc[i]
            vix = row["vix_front_month"]
            rv = row.get("realized_vol_20d", 0.0)
            if pd.isna(rv):
                rv = 0.0
            if pd.isna(vix):
                vix = 18.0
            regime = row.get("regime", "TRANSITIONAL")
            term_slope = row.get("term_structure_slope", 0.0)
            if pd.isna(term_slope):
                term_slope = 0.0

            # Compute daily market moves
            vix_change = (vix - prev_vix) if prev_vix is not None else 0.0
            rv_change = (rv - prev_rv) if prev_rv is not None else 0.0

            # Rolling correlation from returns data
            avg_corr = self._compute_avg_correlation(rv_history, date, base_corr_matrix)
            corr_change = (avg_corr - prev_avg_corr) if prev_avg_corr is not None else 0.0

            # ------------------------------------------------------------------
            # Rebalance: re-run strategies on rebalance days or first day
            # ------------------------------------------------------------------
            if (i % self.rebalance_days == 0 or strategies is None) and not stopped_out:
                day_data = self._build_day_data(
                    data, base_iv_surface, base_corr_matrix,
                    rv_history, regime_df, i, vix,
                )
                strategies = self._run_strategies(day_data)

            # ------------------------------------------------------------------
            # Compute daily PnL from Greeks × market moves
            # ------------------------------------------------------------------
            snap = DailySnapshot(
                date=str(date.date()),
                day_index=i,
                vix=vix,
                vix_change=vix_change,
                realized_vol_20d=rv if not np.isnan(rv) else 0.0,
                rv_change=rv_change,
                regime=regime,
                term_slope=term_slope,
                avg_correlation=avg_corr,
                corr_change=corr_change,
            )

            if strategies and not stopped_out:
                self._compute_daily_pnl(snap, strategies, vix_change, rv_change, corr_change)
                self._record_greeks(snap, strategies)

            # Cumulative tracking
            cumulative_pnl += snap.total_pnl
            peak_pnl = max(peak_pnl, cumulative_pnl)
            drawdown = peak_pnl - cumulative_pnl
            drawdown_pct = drawdown / self.initial_capital if self.initial_capital > 0 else 0.0

            snap.cumulative_pnl = cumulative_pnl
            snap.peak_pnl = peak_pnl
            snap.drawdown = drawdown
            snap.drawdown_pct = drawdown_pct

            # Vega bounds check
            snap.vega_within_bounds = NET_VEGA_FLOOR <= snap.net_vega <= NET_VEGA_CEILING
            snap.notional_within_cap = snap.gross_notional <= GROSS_NOTIONAL_CAP

            # Drawdown stop
            if drawdown_pct > MAX_PORTFOLIO_DRAWDOWN and not stopped_out:
                logger.warning(
                    "DRAWDOWN STOP: %.2f%% exceeds limit of %.0f%% on %s",
                    drawdown_pct * 100, MAX_PORTFOLIO_DRAWDOWN * 100, date.date(),
                )
                stopped_out = True

            snapshots.append(snap)

            # Update previous values
            prev_vix = vix
            prev_rv = rv
            prev_avg_corr = avg_corr

        # Convert to DataFrame
        results = pd.DataFrame([vars(s) for s in snapshots])
        logger.info(
            "Backtest complete: %d days, final PnL $%.0f, max DD %.2f%%",
            n_days, cumulative_pnl, results["drawdown_pct"].max() * 100,
        )
        return results

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_day_data(
        self,
        full_data: dict,
        base_iv_surface: pd.DataFrame,
        base_corr_matrix: pd.DataFrame,
        rv_history: pd.DataFrame,
        regime_df: pd.DataFrame,
        day_idx: int,
        current_vix: float,
    ) -> dict:
        """
        Build the data dict for a single day, mimicking what strategies expect.

        IV surface is scaled by current VIX relative to the base VIX assumption (18).
        Correlation matrix uses the base (could be enhanced with rolling computation).
        Regime signals use the current day's row.
        """
        baseline_vix = 18.0
        vix_ratio = current_vix / baseline_vix if baseline_vix > 0 else 1.0

        # Scale implied vols by VIX ratio (higher VIX → higher IVs across the surface)
        scaled_surface = base_iv_surface.copy()
        scaled_surface["implied_vol"] = scaled_surface["implied_vol"] * vix_ratio

        # Regime signals: just the current day as a single-row DataFrame
        regime_row = regime_df.iloc[[day_idx]].copy()

        # Realized vol history: everything up to current day
        current_date = pd.to_datetime(regime_df.iloc[day_idx]["date"])
        rv_slice = rv_history[rv_history.index <= current_date].copy()

        return {
            "implied_vol_surface": scaled_surface,
            "realized_vol_history": rv_slice,
            "correlation_matrix": base_corr_matrix,
            "vol_regime_signals": regime_row,
            "portfolio_constraints": full_data.get("portfolio_constraints", {}),
            "option_overlay_specs": full_data.get("option_overlay_specs", {}),
        }

    def _run_strategies(self, day_data: dict) -> list:
        """Initialize and run all 5 strategies on the given day's data."""
        import io
        from contextlib import redirect_stdout

        strategies = [
            DispersionStrategy(
                capital=ALLOCATIONS["dispersion"]["capital"],
                universe_size=45,
                active_names=12,
            ),
            VolatilityHarvestingStrategy(
                capital=ALLOCATIONS["volatility_harvesting"]["capital"],
                short_variance_notional_cap=VOL_HARVEST_SHORT_VARIANCE_NOTIONAL_CAP,
            ),
            DirectionalLongShortStrategy(
                capital=ALLOCATIONS["directional_long_short"]["capital"],
            ),
            DynamicVolTargetingStrategy(
                capital=ALLOCATIONS["dynamic_vol_targeting"]["capital"],
            ),
            OptionOverlayStrategy(
                capital=ALLOCATIONS["option_overlay"]["capital"],
                portfolio_notional=TOTAL_CAPITAL,
            ),
        ]

        output = io.StringIO() if self.suppress_prints else None

        for strat in strategies:
            try:
                if self.suppress_prints:
                    with redirect_stdout(output):
                        signals = strat.generate_signals(day_data)
                        strat.construct_positions(signals)
                else:
                    signals = strat.generate_signals(day_data)
                    strat.construct_positions(signals)
            except Exception as e:
                logger.debug("Strategy %s failed on this day: %s", strat.name, e)

        return strategies

    def _compute_daily_pnl(
        self,
        snap: DailySnapshot,
        strategies: list,
        vix_change: float,
        rv_change: float,
        corr_change: float,
    ):
        """Compute daily PnL using the Greeks-based attribution model."""
        total_vega_pnl = 0.0
        total_gamma_pnl = 0.0
        total_theta_pnl = 0.0
        total_corr_pnl = 0.0

        strategy_pnl = {}

        for strat in strategies:
            g = strat.greeks

            vega_pnl = g["vega"] * vix_change
            gamma_pnl = 0.5 * g["gamma"] * (rv_change ** 2)
            theta_pnl = g["theta"]  # Daily theta

            corr_pnl = 0.0
            if "Dispersion" in strat.name and corr_change != 0:
                corr_sensitivity = -strat.capital * 0.15
                corr_pnl = corr_sensitivity * corr_change

            strat_total = vega_pnl + gamma_pnl + theta_pnl + corr_pnl
            strategy_pnl[strat.name] = strat_total

            total_vega_pnl += vega_pnl
            total_gamma_pnl += gamma_pnl
            total_theta_pnl += theta_pnl
            total_corr_pnl += corr_pnl

        snap.vega_pnl = total_vega_pnl
        snap.gamma_pnl = total_gamma_pnl
        snap.theta_pnl = total_theta_pnl
        snap.correlation_pnl = total_corr_pnl
        snap.total_pnl = total_vega_pnl + total_gamma_pnl + total_theta_pnl + total_corr_pnl
        snap.strategy_pnl = strategy_pnl

    def _record_greeks(self, snap: DailySnapshot, strategies: list):
        """Record end-of-day portfolio Greeks and position counts."""
        net_vega = sum(s.greeks["vega"] for s in strategies)
        net_gamma = sum(s.greeks["gamma"] for s in strategies)
        net_delta = sum(s.greeks["delta"] for s in strategies)
        net_theta = sum(s.greeks["theta"] for s in strategies)
        gross_notional = sum(s.notional_deployed for s in strategies)

        snap.net_vega = net_vega
        snap.net_gamma = net_gamma
        snap.net_delta = net_delta
        snap.net_theta = net_theta
        snap.gross_notional = gross_notional
        snap.num_positions = sum(len(s.positions) for s in strategies)
        snap.active_strategies = sum(1 for s in strategies if len(s.positions) > 0)

    def _compute_avg_correlation(
        self,
        rv_history: pd.DataFrame,
        current_date: pd.Timestamp,
        base_corr: pd.DataFrame,
    ) -> float:
        """
        Compute average pairwise correlation.
        Uses rolling 90-day returns if enough history; falls back to base matrix.
        """
        # Get return columns
        return_cols = [c for c in rv_history.columns if c.endswith("_return")]

        if len(return_cols) < 5:
            # Fall back to base correlation matrix
            vals = base_corr.values
            upper = vals[np.triu_indices(vals.shape[0], k=1)]
            return float(np.mean(upper))

        # Get 90-day window of returns
        mask = rv_history.index <= current_date
        window = rv_history.loc[mask, return_cols].tail(90)

        if len(window) < 30:
            vals = base_corr.values
            upper = vals[np.triu_indices(vals.shape[0], k=1)]
            return float(np.mean(upper))

        corr = window.corr()
        upper = corr.values[np.triu_indices(corr.shape[0], k=1)]
        return float(np.nanmean(upper))
