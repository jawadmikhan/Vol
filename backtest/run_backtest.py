"""
Run Backtest — Entry Point
============================
Loads synthetic reference data and runs the walk-forward backtest.

Usage:
    python -m backtest.run_backtest
    python -m backtest.run_backtest --rebalance-days 1 --no-plot
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generators.synthetic_data import generate_all as generate_reference_data
from backtest.engine import BacktestEngine
from backtest.analytics import print_report, plot_backtest, monthly_returns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backtest")


def load_reference_data(data_dir: Path) -> dict:
    """Load all generated reference data files."""
    data = {}
    data["implied_vol_surface"] = pd.read_csv(data_dir / "implied_vol_surface.csv")
    data["realized_vol_history"] = pd.read_csv(data_dir / "realized_vol_history.csv")
    data["correlation_matrix"] = pd.read_csv(data_dir / "correlation_matrix.csv", index_col=0)
    data["vol_regime_signals"] = pd.read_csv(data_dir / "vol_regime_signals.csv")

    with open(data_dir / "portfolio_constraints.json") as f:
        data["portfolio_constraints"] = json.load(f)

    with open(data_dir / "option_overlay_specs.json") as f:
        data["option_overlay_specs"] = json.load(f)

    return data


def main():
    parser = argparse.ArgumentParser(description="Run Volatility Portfolio Backtest")
    parser.add_argument(
        "--rebalance-days", type=int, default=5,
        help="Rebalance strategies every N days (default: 5 = weekly)",
    )
    parser.add_argument(
        "--mtm", action="store_true",
        help="Use mark-to-market mode (Black-Scholes repricing + delta hedging)",
    )
    parser.add_argument(
        "--hedge-cost-bps", type=float, default=3.0,
        help="Transaction cost per delta-hedge trade in basis points (default: 3.0)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip chart generation",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results CSV to this path",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load or generate reference data
    # ------------------------------------------------------------------
    data_dir = Path(__file__).resolve().parent.parent / "data" / "reference"
    if not (data_dir / "implied_vol_surface.csv").exists():
        logger.info("Generating synthetic reference data...")
        generate_reference_data()

    logger.info("Loading reference data...")
    data = load_reference_data(data_dir)
    logger.info(
        "  IV surface: %d rows, RV history: %d rows, Regime signals: %d days",
        len(data["implied_vol_surface"]),
        len(data["realized_vol_history"]),
        len(data["vol_regime_signals"]),
    )

    # ------------------------------------------------------------------
    # 2. Run backtest
    # ------------------------------------------------------------------
    pnl_mode = "mtm" if args.mtm else "greeks"
    engine = BacktestEngine(
        pnl_mode=pnl_mode,
        rebalance_days=args.rebalance_days,
        hedge_txn_cost_bps=args.hedge_cost_bps,
        suppress_prints=True,
    )

    logger.info("Running backtest [%s] (rebalance every %d days)...", pnl_mode, args.rebalance_days)
    results = engine.run(data)

    # ------------------------------------------------------------------
    # 3. Print performance report
    # ------------------------------------------------------------------
    metrics = print_report(results)

    # Monthly returns
    monthly = monthly_returns(results)
    if not monthly.empty:
        print("\n  MONTHLY RETURNS (%)")
        print("  " + "-" * 50)
        print(monthly.to_string(float_format="%.2f"))

    # ------------------------------------------------------------------
    # 4. Generate charts
    # ------------------------------------------------------------------
    if not args.no_plot:
        logger.info("Generating backtest charts...")
        plot_backtest(results, save=True)

    # ------------------------------------------------------------------
    # 5. Save results CSV
    # ------------------------------------------------------------------
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / "backtest_results.csv"

    results.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
