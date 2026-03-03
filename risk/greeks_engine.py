"""
Greeks Engine — Portfolio-Level Risk Aggregation
=================================================
Aggregates Delta, Gamma, Vega, Theta, and Correlation sensitivities
across all 5 sub-strategies in dollar terms.

Workflow steps covered: 2.3, 4.6
"""

import pandas as pd
import numpy as np


class GreeksEngine:
    """Compute and validate portfolio-level Greeks against Investment Committee mandate."""

    def __init__(self, strategies: list, vega_floor: float = -2_000_000,
                 vega_ceiling: float = 4_000_000):
        self.strategies = strategies
        self.vega_floor = vega_floor
        self.vega_ceiling = vega_ceiling

    def aggregate_greeks(self) -> pd.DataFrame:
        """
        Step 4.6: Verify portfolio Greeks and constraints.
        Returns a DataFrame with per-strategy and total Greeks in dollar terms.
        """
        rows = []
        for strat in self.strategies:
            greeks = strat.greeks
            rows.append({
                "strategy": strat.name,
                "capital_allocated": strat.capital,
                "notional_deployed": strat.notional_deployed,
                "delta_usd": greeks["delta"],
                "gamma_usd": greeks["gamma"],
                "vega_usd": greeks["vega"],
                "theta_usd": greeks["theta"],
            })

        df = pd.DataFrame(rows)

        # Add total row
        total = {
            "strategy": "PORTFOLIO TOTAL",
            "capital_allocated": df["capital_allocated"].sum(),
            "notional_deployed": df["notional_deployed"].sum(),
            "delta_usd": df["delta_usd"].sum(),
            "gamma_usd": df["gamma_usd"].sum(),
            "vega_usd": df["vega_usd"].sum(),
            "theta_usd": df["theta_usd"].sum(),
        }
        df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)

        return df

    def validate_vega_bounds(self) -> dict:
        """Check that net portfolio vega stays within Investment Committee mandate."""
        total_vega = sum(s.greeks["vega"] for s in self.strategies)
        within_bounds = self.vega_floor <= total_vega <= self.vega_ceiling

        return {
            "net_vega_usd": round(total_vega, 2),
            "vega_floor": self.vega_floor,
            "vega_ceiling": self.vega_ceiling,
            "within_bounds": within_bounds,
            "headroom_to_floor": round(total_vega - self.vega_floor, 2),
            "headroom_to_ceiling": round(self.vega_ceiling - total_vega, 2),
        }

    def validate_notional_cap(self, gross_notional_cap: float = 500_000_000) -> dict:
        """Check gross notional across all strategies."""
        total_notional = sum(s.notional_deployed for s in self.strategies)
        return {
            "gross_notional_usd": round(total_notional, 2),
            "cap_usd": gross_notional_cap,
            "within_cap": total_notional <= gross_notional_cap,
            "utilization_pct": round(total_notional / gross_notional_cap * 100, 1),
        }

    def print_report(self):
        """Print formatted risk report."""
        df = self.aggregate_greeks()
        vega_check = self.validate_vega_bounds()
        notional_check = self.validate_notional_cap()

        print("\n" + "=" * 90)
        print("PORTFOLIO GREEKS REPORT (All values in United States dollars)")
        print("=" * 90)

        # Format as table
        print(f"\n{'Strategy':<35} {'Delta':>12} {'Gamma':>12} {'Vega':>12} {'Theta':>12}")
        print("-" * 83)
        for _, row in df.iterrows():
            name = row["strategy"]
            if name == "PORTFOLIO TOTAL":
                print("-" * 83)
            print(f"{name:<35} {row['delta_usd']:>12,.0f} {row['gamma_usd']:>12,.0f} "
                  f"{row['vega_usd']:>12,.0f} {row['theta_usd']:>12,.0f}")

        print(f"\n--- Vega Constraint Check ---")
        status = "PASS" if vega_check["within_bounds"] else "FAIL"
        print(f"  Net Vega:         ${vega_check['net_vega_usd']:,.0f}")
        print(f"  Allowed Range:    ${vega_check['vega_floor']:,.0f} to ${vega_check['vega_ceiling']:,.0f}")
        print(f"  Status:           {status}")

        print(f"\n--- Notional Cap Check ---")
        status = "PASS" if notional_check["within_cap"] else "FAIL"
        print(f"  Gross Notional:   ${notional_check['gross_notional_usd']:,.0f}")
        print(f"  Cap:              ${notional_check['cap_usd']:,.0f}")
        print(f"  Utilization:      {notional_check['utilization_pct']}%")
        print(f"  Status:           {status}")

        return df
