"""
Live orchestrator — replaces main.py's synthetic data with IBKR live feeds.

Connects to IB Gateway, fetches market data, computes signals, runs risk,
and persists everything to TimescaleDB.

Usage:
    python -m infrastructure.run              # one-shot: fetch + compute + store
    python -m infrastructure.run --schedule   # run on market-hours schedule
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.ibkr.client import IBKRClient
from infrastructure.data_adapter import LiveDataAdapter
from infrastructure.db.connection import insert_dataframe, close_pool

from config.portfolio_constraints import CONSTRAINTS
from strategies.dispersion import DispersionTrading
from strategies.volatility_harvesting import VolatilityHarvesting
from strategies.directional_long_short import DirectionalLongShort
from strategies.dynamic_vol_targeting import DynamicVolTargeting
from strategies.option_overlay import OptionOverlay
from risk.greeks_engine import GreeksEngine
from risk.scenario_analysis import ScenarioAnalysis
from risk.attribution import PnLAttribution

import pandas as pd

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vol-engine")


def run_cycle():
    """Execute one full portfolio cycle: fetch data → signals → risk → persist."""
    logger.info("=" * 70)
    logger.info("STARTING PORTFOLIO CYCLE — %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Connect to IBKR
    # ------------------------------------------------------------------
    client = IBKRClient(
        host=os.getenv("IBKR_HOST", "127.0.0.1"),
        port=int(os.getenv("IBKR_PORT", "4002")),
        client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
    )
    client.connect()

    try:
        # ------------------------------------------------------------------
        # 2. Fetch all live data (replaces synthetic data)
        # ------------------------------------------------------------------
        adapter = LiveDataAdapter(client, use_cache=True)
        data = adapter.fetch_all()

        logger.info("Data fetch complete:")
        logger.info("  IV surface:    %d rows", len(data.get("implied_vol_surface", [])))
        logger.info("  RV history:    %d rows", len(data.get("realized_vol_history", [])))
        logger.info("  Correlation:   %s shape", getattr(data.get("correlation_matrix"), "shape", "N/A"))
        logger.info("  Regime:        %s", data.get("vol_regime_signals", {}).to_dict("records") if hasattr(data.get("vol_regime_signals"), "to_dict") else "N/A")

        # ------------------------------------------------------------------
        # 3. Initialize strategies (same as main.py)
        # ------------------------------------------------------------------
        strategies = [
            DispersionTrading("Dispersion Trading", CONSTRAINTS["strategy_allocations"]["dispersion"]),
            VolatilityHarvesting("Volatility Harvesting", CONSTRAINTS["strategy_allocations"]["volatility_harvesting"]),
            DirectionalLongShort("Directional Long/Short Vol", CONSTRAINTS["strategy_allocations"]["directional_long_short"]),
            DynamicVolTargeting("Dynamic Volatility Targeting", CONSTRAINTS["strategy_allocations"]["dynamic_vol_targeting"]),
            OptionOverlay("Option Overlay", CONSTRAINTS["strategy_allocations"]["option_overlay"]),
        ]

        # ------------------------------------------------------------------
        # 4. Generate signals and construct positions
        # ------------------------------------------------------------------
        all_positions = []
        for strategy in strategies:
            logger.info("Processing strategy: %s", strategy.name)

            signals = strategy.generate_signals(data)
            positions = strategy.construct_positions(signals)
            all_positions.extend(positions)

            # Persist signals
            if isinstance(signals, pd.DataFrame) and not signals.empty:
                signals_db = signals.copy()
                signals_db["ts"] = datetime.now(timezone.utc)
                signals_db["strategy"] = strategy.name
                # Store key signal columns
                for col in signals.columns:
                    if col not in ("ts", "strategy", "date"):
                        try:
                            insert_dataframe("strategy_signals", pd.DataFrame([{
                                "ts": datetime.now(timezone.utc),
                                "strategy": strategy.name,
                                "signal_name": col,
                                "signal_value": float(signals[col].iloc[-1]) if pd.notna(signals[col].iloc[-1]) else None,
                                "metadata": None,
                            }]))
                        except (ValueError, TypeError):
                            pass

            logger.info("  → %d positions generated", len(positions))

        # ------------------------------------------------------------------
        # 5. Risk — aggregate Greeks
        # ------------------------------------------------------------------
        greeks_engine = GreeksEngine(strategies)
        greeks_df = greeks_engine.aggregate_greeks()
        vega_check = greeks_engine.validate_vega_bounds()
        notional_check = greeks_engine.validate_notional_cap()

        logger.info("Portfolio Greeks:")
        logger.info("  Net Vega:      $%,.0f", vega_check["net_vega_usd"])
        logger.info("  Vega bounds:   %s", "OK" if vega_check["within_bounds"] else "BREACH")
        logger.info("  Gross Notional: $%,.0f", notional_check["gross_notional_usd"])
        logger.info("  Notional cap:  %s", "OK" if notional_check["within_cap"] else "BREACH")

        # Persist Greeks
        greeks_persist = greeks_df.copy()
        greeks_persist["ts"] = datetime.now(timezone.utc)
        greeks_persist["vega_headroom_floor"] = vega_check.get("headroom_to_floor")
        greeks_persist["vega_headroom_ceiling"] = vega_check.get("headroom_to_ceiling")
        insert_dataframe("portfolio_greeks", greeks_persist)

        # ------------------------------------------------------------------
        # 6. Risk — scenario analysis
        # ------------------------------------------------------------------
        scenario = ScenarioAnalysis(strategies, CONSTRAINTS)
        scenario_df = scenario.run_all_scenarios()

        logger.info("Scenario Analysis:")
        for _, row in scenario_df.iterrows():
            logger.info(
                "  %s → %s: $%,.0f (%.2f%%)",
                row.get("scenario", ""),
                row.get("strategy", ""),
                row.get("pnl_usd", 0),
                row.get("pnl_pct_of_capital", 0) * 100,
            )

        # Persist scenarios
        scenario_persist = scenario_df.copy()
        scenario_persist["ts"] = datetime.now(timezone.utc)
        insert_dataframe("scenario_results", scenario_persist)

        # ------------------------------------------------------------------
        # 7. PnL Attribution
        # ------------------------------------------------------------------
        attribution = PnLAttribution(strategies)
        attr_df = attribution.compute_daily_attribution(data)

        if isinstance(attr_df, pd.DataFrame) and not attr_df.empty:
            attr_persist = attr_df.copy()
            attr_persist["ts"] = datetime.now(timezone.utc)
            insert_dataframe("pnl_attribution", attr_persist)
            logger.info("PnL attribution persisted")

        # ------------------------------------------------------------------
        # Done
        # ------------------------------------------------------------------
        logger.info("=" * 70)
        logger.info("CYCLE COMPLETE — %s", datetime.now(timezone.utc).isoformat())
        logger.info("=" * 70)

    finally:
        client.disconnect()


def run_scheduled():
    """Run cycles on a schedule during market hours."""
    logger.info("Starting scheduled mode — will run at market open and every 30 min")

    while True:
        now = datetime.now(timezone.utc)
        hour_et = (now.hour - 4) % 24  # rough UTC → ET conversion

        # Run during US market hours (9:30 AM - 4:00 PM ET)
        if 9 <= hour_et <= 16:
            try:
                run_cycle()
            except Exception as e:
                logger.error("Cycle failed: %s", e, exc_info=True)

            # Wait 30 minutes between cycles
            time.sleep(30 * 60)
        else:
            # Outside market hours — check every 15 minutes
            logger.debug("Outside market hours (ET hour=%d), sleeping", hour_et)
            time.sleep(15 * 60)


def main():
    parser = argparse.ArgumentParser(description="Vol Portfolio — Live Engine")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run on a recurring schedule during market hours",
    )
    args = parser.parse_args()

    try:
        if args.schedule:
            run_scheduled()
        else:
            run_cycle()
    finally:
        close_pool()


if __name__ == "__main__":
    main()
