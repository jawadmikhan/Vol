"""
Transaction Cost Analysis (TCA)
=================================
Post-trade analysis of execution quality.

Metrics:
  - Implementation shortfall (IS): slippage vs. arrival price
  - VWAP comparison
  - Fill rate by phase/strategy
  - Commission analysis
  - Spread capture ratio
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_fills(fill_history: list[dict]) -> dict:
    """
    Compute TCA metrics from fill history.

    Args:
        fill_history: List of fill records from OrderManager.

    Returns:
        Dict of TCA metrics.
    """
    if not fill_history:
        return {"total_fills": 0}

    df = pd.DataFrame(fill_history)

    total_contracts = df["total_contracts"].sum()
    total_commission = df["commission"].sum()
    total_premium = df["fill_premium"].abs().sum()
    avg_slippage = df["slippage_bps"].mean()

    # Per-strategy breakdown
    strategy_stats = {}
    for strategy in df["strategy"].unique():
        mask = df["strategy"] == strategy
        subset = df[mask]
        strategy_stats[strategy] = {
            "fills": len(subset),
            "contracts": subset["total_contracts"].sum(),
            "commission": subset["commission"].sum(),
            "avg_slippage_bps": subset["slippage_bps"].mean(),
            "total_premium": subset["fill_premium"].abs().sum(),
        }

    # Per-phase breakdown
    phase_stats = {}
    if "phase" in df.columns:
        for phase in df["phase"].unique():
            if not phase:
                continue
            mask = df["phase"] == phase
            subset = df[mask]
            phase_stats[phase] = {
                "fills": len(subset),
                "contracts": subset["total_contracts"].sum(),
                "commission": subset["commission"].sum(),
                "avg_slippage_bps": subset["slippage_bps"].mean(),
            }

    # Cost breakdown
    total_cost = total_commission + (avg_slippage / 10000 * total_premium)
    cost_as_pct_premium = (total_cost / total_premium * 100) if total_premium > 0 else 0

    return {
        "total_fills": len(df),
        "total_contracts": int(total_contracts),
        "total_commission": total_commission,
        "total_premium": total_premium,
        "avg_slippage_bps": avg_slippage,
        "total_estimated_cost": total_cost,
        "cost_pct_of_premium": cost_as_pct_premium,
        "by_strategy": strategy_stats,
        "by_phase": phase_stats,
    }


def print_tca_report(fill_history: list[dict]):
    """Print a formatted TCA report."""
    metrics = analyze_fills(fill_history)

    if metrics["total_fills"] == 0:
        print("  No fills to analyze.")
        return metrics

    print("\n" + "=" * 70)
    print("  TRANSACTION COST ANALYSIS")
    print("=" * 70)

    print(f"\n  Total Fills:          {metrics['total_fills']}")
    print(f"  Total Contracts:      {metrics['total_contracts']:,}")
    print(f"  Total Premium:        ${metrics['total_premium']:,.0f}")
    print(f"  Total Commission:     ${metrics['total_commission']:,.2f}")
    print(f"  Avg Slippage:         {metrics['avg_slippage_bps']:.1f} bps")
    print(f"  Total Est. Cost:      ${metrics['total_estimated_cost']:,.0f}")
    print(f"  Cost % of Premium:    {metrics['cost_pct_of_premium']:.2f}%")

    if metrics["by_strategy"]:
        print(f"\n  {'Strategy':<35} {'Fills':>6} {'Contracts':>10} {'Commission':>12} {'Slippage':>10}")
        print(f"  {'-' * 73}")
        for strat, stats in metrics["by_strategy"].items():
            print(f"  {strat:<35} {stats['fills']:>6} {stats['contracts']:>10,} "
                  f"${stats['commission']:>10,.2f} {stats['avg_slippage_bps']:>9.1f}bp")

    if metrics["by_phase"]:
        print(f"\n  {'Phase':<15} {'Fills':>6} {'Contracts':>10} {'Commission':>12}")
        print(f"  {'-' * 43}")
        for phase, stats in metrics["by_phase"].items():
            print(f"  {phase:<15} {stats['fills']:>6} {stats['contracts']:>10,} "
                  f"${stats['commission']:>10,.2f}")

    print(f"\n{'=' * 70}")
    return metrics
