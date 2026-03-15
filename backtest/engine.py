"""
Backtest Engine - Walk-Forward Simulation
==========================================
Replays historical data day-by-day through the strategy pipeline.

Supports two PnL modes:
  - "greeks": PnL from Greek exposures x market moves (fast, approximate)
  - "mtm":    Mark-to-market via Black-Scholes repricing (accurate, slower)

The MTM mode:
  1. Constructs a spot price path from synthetic returns
  2. Builds option positions in the PortfolioPricer on rebalance days
  3. Reprices all positions daily with BS + current IV surface
  4. Layers delta-hedging PnL (gamma scalping vs theta)
  5. Applies transaction costs on entry and rebalance

Usage:
    engine = BacktestEngine(pnl_mode="mtm")
    results = engine.run(data)
"""

import sys
import os
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
# Daily snapshot - one row per day in the results
# ---------------------------------------------------------------------------

@dataclass
class DailySnapshot:
    date: str
    day_index: int

    # Market state
    vix: float = 0.0
    vix_change: float = 0.0
    spot: float = 0.0
    spot_return: float = 0.0
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
    hedge_txn_cost: float = 0.0
    entry_txn_cost: float = 0.0
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

    Args:
        pnl_mode: "greeks" for Greek-based PnL, "mtm" for mark-to-market.
        rebalance_days: Re-run strategy signals every N days.
        initial_capital: Starting capital.
        hedge_frequency_days: Delta-hedging frequency (MTM mode only).
        hedge_txn_cost_bps: Transaction cost per hedge trade in bps.
        suppress_prints: Suppress strategy print statements.
    """

    def __init__(
        self,
        pnl_mode: str = "greeks",
        rebalance_days: int = 5,
        initial_capital: float = TOTAL_CAPITAL,
        hedge_frequency_days: float = 1.0,
        hedge_txn_cost_bps: float = 3.0,
        suppress_prints: bool = True,
    ):
        self.pnl_mode = pnl_mode
        self.rebalance_days = rebalance_days
        self.initial_capital = initial_capital
        self.hedge_frequency_days = hedge_frequency_days
        self.hedge_txn_cost_bps = hedge_txn_cost_bps
        self.suppress_prints = suppress_prints

    def run(self, data: dict) -> pd.DataFrame:
        """Execute the full walk-forward backtest."""
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

        # Build spot price path from returns
        spot_path = self._build_spot_path(rv_history, dates)

        logger.info(
            "Backtest [%s mode]: %d days from %s to %s",
            self.pnl_mode, n_days, dates[0].date(), dates[-1].date(),
        )

        # Initialize state
        strategies = None
        pricer = None
        hedger = None
        hedge_state = None
        tcm = None
        snapshots = []
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        prev_vix = None
        prev_rv = None
        prev_avg_corr = None
        stopped_out = False

        # Initialize MTM components if needed
        if self.pnl_mode == "mtm":
            from models.pricer import PortfolioPricer, positions_from_strategy
            from models.delta_hedge import DeltaHedger, HedgeState
            from models.transaction_costs import TransactionCostModel

            pricer = PortfolioPricer()
            hedger = DeltaHedger(
                hedge_frequency_days=self.hedge_frequency_days,
                txn_cost_bps=self.hedge_txn_cost_bps,
            )
            hedge_state = HedgeState()
            tcm = TransactionCostModel()

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

            spot = spot_path[i]
            spot_return = (spot / spot_path[i - 1] - 1.0) if i > 0 else 0.0

            # Daily market moves
            vix_change = (vix - prev_vix) if prev_vix is not None else 0.0
            rv_change = (rv - prev_rv) if prev_rv is not None else 0.0
            avg_corr = self._compute_avg_correlation(rv_history, date, base_corr_matrix)
            corr_change = (avg_corr - prev_avg_corr) if prev_avg_corr is not None else 0.0

            # ------------------------------------------------------------------
            # Rebalance
            # ------------------------------------------------------------------
            is_rebalance = (i % self.rebalance_days == 0 or strategies is None) and not stopped_out
            entry_cost = 0.0

            if is_rebalance:
                day_data = self._build_day_data(
                    data, base_iv_surface, base_corr_matrix,
                    rv_history, regime_df, i, vix,
                )
                strategies = self._run_strategies(day_data)

                # MTM: rebuild pricer positions from strategy output
                if self.pnl_mode == "mtm" and strategies:
                    from models.pricer import PortfolioPricer, positions_from_strategy
                    pricer = PortfolioPricer()
                    atm_vol = vix / 100.0 if vix > 1 else 0.18

                    for strat in strategies:
                        if strat.positions:
                            positions_from_strategy(
                                strat.name, strat.positions,
                                spot, atm_vol, pricer,
                            )

                    # Do an initial reprice with the current surface to set
                    # a consistent MTM baseline (avoids vol mismatch on day 1)
                    iv_surface = day_data["implied_vol_surface"]
                    spot_prices = {"SPX_INDEX": spot, "CONSTITUENT": spot}
                    pricer.reprice(spot_prices, iv_surface, elapsed_days=0)
                    # Now prev_mtm == total_mtm, so next reprice gives true incremental PnL

                    # Entry transaction costs (amortize over rebalance period)
                    if tcm:
                        for strat in strategies:
                            if strat.positions:
                                cost_result = tcm.portfolio_entry_cost(
                                    strat.positions, spot, atm_vol,
                                )
                                entry_cost += cost_result["total_cost"]
                        # Amortize entry cost over the rebalance period
                        entry_cost = entry_cost / max(self.rebalance_days, 1)

            # ------------------------------------------------------------------
            # Daily PnL
            # ------------------------------------------------------------------
            snap = DailySnapshot(
                date=str(date.date()),
                day_index=i,
                vix=vix,
                vix_change=vix_change,
                spot=spot,
                spot_return=spot_return,
                realized_vol_20d=rv if not np.isnan(rv) else 0.0,
                rv_change=rv_change,
                regime=regime,
                term_slope=term_slope,
                avg_correlation=avg_corr,
                corr_change=corr_change,
                entry_txn_cost=entry_cost,
            )

            if strategies and not stopped_out:
                if self.pnl_mode == "mtm" and pricer and pricer.positions:
                    self._compute_mtm_pnl(
                        snap, strategies, pricer, hedger, hedge_state,
                        spot, spot_path[i - 1] if i > 0 else spot,
                        day_data if is_rebalance else self._build_day_data(
                            data, base_iv_surface, base_corr_matrix,
                            rv_history, regime_df, i, vix,
                        ),
                        entry_cost,
                    )
                else:
                    self._compute_greeks_pnl(snap, strategies, vix_change, rv_change, corr_change)
                self._record_greeks(snap, strategies, pricer)

            # Cumulative tracking
            cumulative_pnl += snap.total_pnl
            peak_pnl = max(peak_pnl, cumulative_pnl)
            drawdown = peak_pnl - cumulative_pnl
            drawdown_pct = drawdown / self.initial_capital if self.initial_capital > 0 else 0.0

            snap.cumulative_pnl = cumulative_pnl
            snap.peak_pnl = peak_pnl
            snap.drawdown = drawdown
            snap.drawdown_pct = drawdown_pct
            snap.vega_within_bounds = NET_VEGA_FLOOR <= snap.net_vega <= NET_VEGA_CEILING
            snap.notional_within_cap = snap.gross_notional <= GROSS_NOTIONAL_CAP

            if drawdown_pct > MAX_PORTFOLIO_DRAWDOWN and not stopped_out:
                logger.warning(
                    "DRAWDOWN STOP: %.2f%% on %s", drawdown_pct * 100, date.date(),
                )
                stopped_out = True

            snapshots.append(snap)
            prev_vix = vix
            prev_rv = rv
            prev_avg_corr = avg_corr

        results = pd.DataFrame([vars(s) for s in snapshots])
        logger.info(
            "Backtest complete: %d days, final PnL $%.0f, max DD %.2f%%",
            n_days, cumulative_pnl, results["drawdown_pct"].max() * 100,
        )
        return results

    # ------------------------------------------------------------------
    # Spot price path construction
    # ------------------------------------------------------------------

    def _build_spot_path(self, rv_history: pd.DataFrame, dates: list) -> np.ndarray:
        """Build a spot price path from synthetic return data."""
        return_col = None
        for col in rv_history.columns:
            if "SPX_INDEX_return" in col:
                return_col = col
                break

        if return_col is None:
            # No return data - synthesize from VIX (approximate)
            return np.full(len(dates), 4500.0)

        # Start at 4500 and compound returns
        spot_path = np.zeros(len(dates))
        spot_path[0] = 4500.0

        for i, date in enumerate(dates):
            if i == 0:
                continue
            # Find the closest return for this date
            if date in rv_history.index:
                r = rv_history.loc[date, return_col]
            else:
                # Nearest date
                idx = rv_history.index.get_indexer([date], method="nearest")[0]
                r = rv_history.iloc[idx][return_col] if idx >= 0 else 0.0

            if pd.isna(r):
                r = 0.0
            spot_path[i] = spot_path[i - 1] * (1 + r)

        return spot_path

    # ------------------------------------------------------------------
    # MTM PnL computation
    # ------------------------------------------------------------------

    def _compute_mtm_pnl(
        self,
        snap: DailySnapshot,
        strategies: list,
        pricer,
        hedger,
        hedge_state,
        curr_spot: float,
        prev_spot: float,
        day_data: dict,
        entry_cost: float,
    ):
        """Compute daily PnL using mark-to-market repricing."""
        from models.delta_hedge import HedgeState

        iv_surface = day_data["implied_vol_surface"]

        # Reprice all positions
        spot_prices = {"SPX_INDEX": curr_spot, "CONSTITUENT": curr_spot}
        reprice_result = pricer.reprice(spot_prices, iv_surface, elapsed_days=1)

        mtm_pnl = reprice_result["daily_pnl"]

        # Delta-hedging simulation
        port_greeks = reprice_result["portfolio_greeks"]
        hedge_result = hedger.simulate_day(
            state=hedge_state,
            prev_spot=prev_spot,
            curr_spot=curr_spot,
            portfolio_gamma_usd=port_greeks["gamma"],
            portfolio_theta_usd=port_greeks["theta"],
            portfolio_delta_usd=port_greeks["delta"],
        )

        # PnL decomposition
        # In MTM mode, the repricing captures all Greek effects together.
        # We decompose approximately using portfolio Greeks:
        port_greeks = reprice_result["portfolio_greeks"]
        snap.vega_pnl = port_greeks.get("vega", 0) * snap.vix_change
        snap.gamma_pnl = hedge_result["gamma_pnl"]
        snap.theta_pnl = port_greeks.get("theta", 0)
        snap.hedge_txn_cost = hedge_result["txn_cost"]
        snap.correlation_pnl = 0.0

        for strat in strategies:
            if "Dispersion" in strat.name and snap.corr_change != 0:
                corr_sensitivity = -strat.capital * 0.15
                snap.correlation_pnl = corr_sensitivity * snap.corr_change

        # Total PnL = MTM change - all transaction costs
        # MTM change is the authoritative PnL; Greek decomposition is approximate
        snap.total_pnl = mtm_pnl - hedge_result["txn_cost"] - entry_cost

        # Per-strategy PnL (approximate from strategy Greeks)
        strat_greeks = reprice_result.get("strategy_greeks", {})
        strategy_pnl = {}
        spot_return = (curr_spot / prev_spot - 1.0) if prev_spot > 0 else 0.0
        for strat in strategies:
            sg = strat_greeks.get(strat.name, strat.greeks)
            if isinstance(sg, dict):
                # Approximate per-strategy PnL from their Greeks
                s_pnl = (
                    sg.get("vega", 0) * snap.vix_change
                    + 0.5 * sg.get("gamma", 0) * (spot_return ** 2)
                    + sg.get("theta", 0)
                )
            else:
                s_pnl = 0.0
            strategy_pnl[strat.name] = s_pnl

        snap.strategy_pnl = strategy_pnl
        snap.num_positions = reprice_result.get("num_positions", 0)

    # ------------------------------------------------------------------
    # Greeks-based PnL (legacy mode)
    # ------------------------------------------------------------------

    def _compute_greeks_pnl(
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
            theta_pnl = g["theta"]
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

    # ------------------------------------------------------------------
    # Greeks recording
    # ------------------------------------------------------------------

    def _record_greeks(self, snap: DailySnapshot, strategies: list, pricer=None):
        """Record end-of-day portfolio Greeks and position counts."""
        if pricer and pricer.positions:
            pg = pricer._aggregate_greeks()
            snap.net_vega = pg["vega"]
            snap.net_gamma = pg["gamma"]
            snap.net_delta = pg["delta"]
            snap.net_theta = pg["theta"]
        else:
            snap.net_vega = sum(s.greeks["vega"] for s in strategies)
            snap.net_gamma = sum(s.greeks["gamma"] for s in strategies)
            snap.net_delta = sum(s.greeks["delta"] for s in strategies)
            snap.net_theta = sum(s.greeks["theta"] for s in strategies)

        snap.gross_notional = sum(s.notional_deployed for s in strategies)
        if not snap.num_positions:
            snap.num_positions = sum(len(s.positions) for s in strategies)
        snap.active_strategies = sum(1 for s in strategies if len(s.positions) > 0)

    # ------------------------------------------------------------------
    # Data construction
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
        """Build the data dict for a single day with SVI-inspired surface dynamics."""
        baseline_vix = 18.0
        vix_ratio = current_vix / baseline_vix if baseline_vix > 0 else 1.0
        regime_row = regime_df.iloc[[day_idx]].copy()
        regime = regime_row.iloc[0].get("regime", "TRANSITIONAL")

        scaled_surface = base_iv_surface.copy()

        for idx, row in scaled_surface.iterrows():
            base_iv = row["implied_vol"]
            delta_label = row["strike_delta"]
            tenor = row["tenor_months"]
            is_index = row["name"] == "SPX_INDEX"

            beta = 1.0 if is_index else 0.7 + np.random.uniform(-0.1, 0.1)
            atm_shift = (vix_ratio - 1.0) * beta

            skew_multiplier = 1.0
            if regime == "CRISIS":
                skew_effect = {
                    "10D_PUT": 0.15, "25D_PUT": 0.08, "ATM": 0.0,
                    "25D_CALL": -0.03, "10D_CALL": -0.05,
                }
                skew_multiplier = 1.0 + skew_effect.get(delta_label, 0.0)
            elif regime == "LOW_VOL_HARVESTING":
                skew_effect = {
                    "10D_PUT": -0.04, "25D_PUT": -0.02, "ATM": 0.0,
                    "25D_CALL": 0.01, "10D_CALL": 0.02,
                }
                skew_multiplier = 1.0 + skew_effect.get(delta_label, 0.0)

            tenor_factor = 1.0
            if regime == "CRISIS":
                tenor_factor = 1.0 + 0.08 * max(0, (6 - tenor) / 6)
            elif regime == "LOW_VOL_HARVESTING":
                tenor_factor = 1.0 - 0.02 * max(0, (6 - tenor) / 6)

            new_iv = base_iv * (1.0 + atm_shift) * skew_multiplier * tenor_factor
            new_iv = max(new_iv, 0.03)
            scaled_surface.at[idx, "implied_vol"] = new_iv

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
                universe_size=45, active_names=12,
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
                logger.debug("Strategy %s failed: %s", strat.name, e)

        return strategies

    def _compute_avg_correlation(self, rv_history, current_date, base_corr):
        """Compute average pairwise correlation."""
        return_cols = [c for c in rv_history.columns if c.endswith("_return")]
        if len(return_cols) < 5:
            vals = base_corr.values
            upper = vals[np.triu_indices(vals.shape[0], k=1)]
            return float(np.mean(upper))

        mask = rv_history.index <= current_date
        window = rv_history.loc[mask, return_cols].tail(90)
        if len(window) < 30:
            vals = base_corr.values
            upper = vals[np.triu_indices(vals.shape[0], k=1)]
            return float(np.mean(upper))

        corr = window.corr()
        upper = corr.values[np.triu_indices(corr.shape[0], k=1)]
        return float(np.nanmean(upper))
