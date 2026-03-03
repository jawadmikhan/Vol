"""
Scenario Analysis
==================
Stress-test the portfolio under two mandated scenarios:
  1. VIX spike to 45 (with correlation shock)
  2. VIX collapse to 11

Each scenario produces explicit portfolio-level Profit and Loss figures.

Workflow steps covered: 3.2, 4.9
"""

import pandas as pd
import numpy as np


class ScenarioAnalysis:
    """
    Step 4.9: Run full scenario analysis.
    Computes Profit and Loss for each sub-strategy and the portfolio total
    under VIX spike and VIX collapse scenarios.
    """

    def __init__(self, strategies: list, baseline_vix: float = 18.0):
        self.strategies = strategies
        self.baseline_vix = baseline_vix

    def run_scenario(self, vix_level: float, scenario_name: str,
                     correlation_shock: float = 0.0) -> pd.DataFrame:
        """Run a single scenario across all strategies."""
        rows = []
        for strat in self.strategies:
            pnl = strat.run_scenario(vix_level, correlation_shock)
            rows.append({
                "strategy": strat.name,
                "scenario": scenario_name,
                "vix_level": vix_level,
                "correlation_shock": correlation_shock,
                "pnl_usd": pnl,
                "pnl_pct_of_capital": round(pnl / strat.capital * 100, 2) if strat.capital > 0 else 0,
            })

        df = pd.DataFrame(rows)

        # Add total
        total_pnl = df["pnl_usd"].sum()
        total_capital = sum(s.capital for s in self.strategies)
        df = pd.concat([df, pd.DataFrame([{
            "strategy": "PORTFOLIO TOTAL",
            "scenario": scenario_name,
            "vix_level": vix_level,
            "correlation_shock": correlation_shock,
            "pnl_usd": total_pnl,
            "pnl_pct_of_capital": round(total_pnl / total_capital * 100, 2),
        }])], ignore_index=True)

        return df

    def run_all_mandated_scenarios(self) -> pd.DataFrame:
        """
        Run the two Investment Committee mandated scenarios:
          1. VIX spike to 45 with +0.25 correlation shock
          2. VIX collapse to 11 with -0.10 correlation shift
        """
        results = []

        # Scenario 1: VIX spike to 45
        # In a crisis, correlations spike — model a +0.25 correlation shock
        spike = self.run_scenario(
            vix_level=45,
            scenario_name="VIX Spike to 45",
            correlation_shock=0.25,
        )
        results.append(spike)

        # Scenario 2: VIX collapse to 11
        # In extreme low vol, correlations decrease slightly
        collapse = self.run_scenario(
            vix_level=11,
            scenario_name="VIX Collapse to 11",
            correlation_shock=-0.10,
        )
        results.append(collapse)

        return pd.concat(results, ignore_index=True)

    def print_report(self):
        """Print formatted scenario analysis report."""
        results = self.run_all_mandated_scenarios()

        print("\n" + "=" * 95)
        print("SCENARIO ANALYSIS REPORT (All Profit and Loss in United States dollars)")
        print("=" * 95)

        for scenario_name in results["scenario"].unique():
            scenario_data = results[results["scenario"] == scenario_name]

            print(f"\n--- {scenario_name} ---")
            vix = scenario_data["vix_level"].iloc[0]
            corr = scenario_data["correlation_shock"].iloc[0]
            print(f"  VIX Level: {vix}  |  Correlation Shock: {corr:+.2f}")
            print()

            print(f"  {'Strategy':<35} {'Profit and Loss':>18} {'% of Capital':>14}")
            print(f"  {'-'*67}")

            for _, row in scenario_data.iterrows():
                name = row["strategy"]
                if name == "PORTFOLIO TOTAL":
                    print(f"  {'-'*67}")
                print(f"  {name:<35} ${row['pnl_usd']:>16,.0f} {row['pnl_pct_of_capital']:>12.1f}%")

        return results
