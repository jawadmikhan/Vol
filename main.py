"""
Systematic Volatility Portfolio — Main Orchestrator
=====================================================
Runs the full 48-step, 7-stage workflow:
  Stage 1: Generate reference data and screen opportunities
  Stage 2: Calibrate models and backtest
  Stage 3: Validate constraints and stress-test
  Stage 4: Construct positions across all 5 sub-strategies
  Stage 5: Generate execution schedule (60-day phased build-up)
  Stage 6: Compute Profit and Loss attribution
  Stage 7: Print portfolio summary

Usage:
    python main.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure imports work from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.portfolio_constraints import (
    TOTAL_CAPITAL, GROSS_NOTIONAL_CAP, NET_VEGA_FLOOR, NET_VEGA_CEILING,
    ALLOCATIONS, IMPLEMENTATION_PHASES, SCENARIO_VIX_SPIKE, SCENARIO_VIX_COLLAPSE,
    VOL_HARVEST_SHORT_VARIANCE_NOTIONAL_CAP,
)
from data.generators.synthetic_data import generate_all as generate_reference_data
from strategies.dispersion import DispersionStrategy
from strategies.volatility_harvesting import VolatilityHarvestingStrategy
from strategies.directional_long_short import DirectionalLongShortStrategy
from strategies.dynamic_vol_targeting import DynamicVolTargetingStrategy
from strategies.option_overlay import OptionOverlayStrategy
from risk.greeks_engine import GreeksEngine
from risk.scenario_analysis import ScenarioAnalysis
from risk.attribution import PnLAttribution


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


def print_implementation_timeline():
    """
    Step 5.4: Generate execution schedule.
    Print the 60-day phased build-up with capital deployment milestones.
    """
    print("\n" + "=" * 80)
    print("IMPLEMENTATION TIMELINE — 60-Day Phased Build-Up")
    print("=" * 80)

    cumulative_capital = 0
    for phase_key, phase in IMPLEMENTATION_PHASES.items():
        strat_name = phase["strategy"]
        capital = ALLOCATIONS[strat_name]["capital"]
        cumulative_capital += capital
        pct_deployed = cumulative_capital / TOTAL_CAPITAL * 100

        print(f"\n  {phase_key.upper()} | Weeks {phase['weeks']} (Days {phase['days'][0]}-{phase['days'][1]})")
        print(f"    Strategy:            {ALLOCATIONS[strat_name]['description']}")
        print(f"    Capital This Phase:  ${capital:,.0f}")
        print(f"    Cumulative Deployed: ${cumulative_capital:,.0f} ({pct_deployed:.0f}%)")


def print_executive_summary(strategies: list, greeks_df: pd.DataFrame):
    """Print the executive summary as it would appear in the Investment Committee memorandum."""
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY — Systematic Volatility Portfolio Proposal")
    print("=" * 80)
    print(f"\n  Total Capital:       ${TOTAL_CAPITAL:,.0f}")
    print(f"  Gross Notional Cap:  ${GROSS_NOTIONAL_CAP:,.0f}")
    print(f"  Net Vega Bounds:     ${NET_VEGA_FLOOR:,.0f} to ${NET_VEGA_CEILING:,.0f}")
    print(f"  Number of Strategies: {len(strategies)}")
    print(f"  Implementation:       60-day phased build-up")

    print(f"\n  {'Strategy':<35} {'Capital':>14} {'Weight':>8} {'Notional':>14}")
    print(f"  {'-'*71}")
    for strat in strategies:
        weight = strat.capital / TOTAL_CAPITAL * 100
        print(f"  {strat.name:<35} ${strat.capital:>12,.0f} {weight:>6.0f}% "
              f"${strat.notional_deployed:>12,.0f}")

    total_notional = sum(s.notional_deployed for s in strategies)
    print(f"  {'-'*71}")
    print(f"  {'TOTAL':<35} ${TOTAL_CAPITAL:>12,.0f} {'100%':>6} ${total_notional:>12,.0f}")


def main():
    """Run the complete portfolio workflow."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " SYSTEMATIC VOLATILITY PORTFOLIO — FULL WORKFLOW EXECUTION ".center(78) + "║")
    print("║" + " $250 Million | 5 Sub-Strategies | 48 Steps | 7 Stages ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # =========================================================================
    # STAGE 1: Generate Reference Data (Steps 1.1–1.7)
    # =========================================================================
    print("\n\n" + "▓" * 80)
    print("  STAGE 1: IDEA GENERATION AND SOURCING")
    print("▓" * 80)

    data_dir = Path(__file__).parent / "data" / "reference"
    if not (data_dir / "implied_vol_surface.csv").exists():
        print("\nGenerating synthetic reference data...\n")
        generate_reference_data()
    else:
        print("\nReference data already exists. Loading...\n")

    data = load_reference_data(data_dir)
    print(f"  Loaded {len(data)} reference datasets.")

    # =========================================================================
    # STAGE 2–4: Strategy Construction (Steps 2.1–4.11)
    # =========================================================================
    print("\n\n" + "▓" * 80)
    print("  STAGES 2–4: RESEARCH, REVIEW, AND THESIS DEVELOPMENT")
    print("▓" * 80)

    # Initialize all 5 strategies
    strategies = []

    # Strategy 1: Dispersion Trading
    print("\n" + "-" * 60)
    print("  [1/5] DISPERSION TRADING")
    print("-" * 60)
    dispersion = DispersionStrategy(
        capital=ALLOCATIONS["dispersion"]["capital"],
        universe_size=45,
        active_names=12,
    )
    signals = dispersion.generate_signals(data)
    dispersion.construct_positions(signals)
    strategies.append(dispersion)

    # Strategy 2: Volatility Harvesting
    print("\n" + "-" * 60)
    print("  [2/5] VOLATILITY HARVESTING")
    print("-" * 60)
    vol_harvest = VolatilityHarvestingStrategy(
        capital=ALLOCATIONS["volatility_harvesting"]["capital"],
        short_variance_notional_cap=VOL_HARVEST_SHORT_VARIANCE_NOTIONAL_CAP,
    )
    signals = vol_harvest.generate_signals(data)
    vol_harvest.construct_positions(signals)
    strategies.append(vol_harvest)

    # Strategy 3: Directional Long/Short Volatility
    print("\n" + "-" * 60)
    print("  [3/5] DIRECTIONAL LONG/SHORT VOLATILITY")
    print("-" * 60)
    directional = DirectionalLongShortStrategy(
        capital=ALLOCATIONS["directional_long_short"]["capital"],
    )
    signals = directional.generate_signals(data)
    directional.construct_positions(signals)
    strategies.append(directional)

    # Strategy 4: Dynamic Volatility Targeting
    print("\n" + "-" * 60)
    print("  [4/5] DYNAMIC VOLATILITY TARGETING")
    print("-" * 60)
    vol_target = DynamicVolTargetingStrategy(
        capital=ALLOCATIONS["dynamic_vol_targeting"]["capital"],
    )
    signals = vol_target.generate_signals(data)
    vol_target.construct_positions(signals)
    strategies.append(vol_target)

    # Strategy 5: Option Overlay
    print("\n" + "-" * 60)
    print("  [5/5] OPTION OVERLAY")
    print("-" * 60)
    overlay = OptionOverlayStrategy(
        capital=ALLOCATIONS["option_overlay"]["capital"],
        portfolio_notional=TOTAL_CAPITAL,
    )
    signals = overlay.generate_signals(data)
    overlay.construct_positions(signals)
    strategies.append(overlay)

    # =========================================================================
    # STAGE 5: Risk Aggregation and Scenario Analysis (Steps 4.6, 4.9, 5.1)
    # =========================================================================
    print("\n\n" + "▓" * 80)
    print("  STAGE 5: DECISION, APPROVAL, AND EXECUTION")
    print("▓" * 80)

    # Greeks aggregation
    greeks_engine = GreeksEngine(strategies, NET_VEGA_FLOOR, NET_VEGA_CEILING)
    greeks_df = greeks_engine.print_report()

    # Scenario analysis
    scenario_engine = ScenarioAnalysis(strategies)
    scenario_engine.print_report()

    # Implementation timeline
    print_implementation_timeline()

    # =========================================================================
    # STAGE 6: Profit and Loss Attribution (Steps 6.1–6.5)
    # =========================================================================
    print("\n\n" + "▓" * 80)
    print("  STAGE 6: MONITORING AND REVIEW")
    print("▓" * 80)

    attribution = PnLAttribution(strategies)
    attribution.print_report(vix_change=0.5, realized_vol_change=0.3)

    # =========================================================================
    # Executive Summary
    # =========================================================================
    print_executive_summary(strategies, greeks_df)

    # =========================================================================
    # Validation
    # =========================================================================
    print("\n\n" + "▓" * 80)
    print("  VALIDATION SUMMARY")
    print("▓" * 80)

    vega_check = greeks_engine.validate_vega_bounds()
    notional_check = greeks_engine.validate_notional_cap()

    checks = [
        ("Net Vega within bounds", vega_check["within_bounds"]),
        ("Gross notional within cap", notional_check["within_cap"]),
        ("All 5 strategies constructed", len(strategies) == 5),
        ("All strategies have positions", all(len(s.positions) > 0 for s in strategies)),
    ]

    all_pass = True
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  All validation checks passed.")
    else:
        print("\n  WARNING: Some validation checks failed. Review positions.")

    print("\n" + "═" * 80)
    print("  WORKFLOW COMPLETE")
    print("═" * 80)
    print()

    return strategies


if __name__ == "__main__":
    strategies = main()
