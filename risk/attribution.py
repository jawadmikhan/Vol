"""
Profit and Loss Attribution
============================
Multi-strategy attribution decomposing returns into:
  - Vega (implied vol changes)
  - Gamma (realized vol / convexity)
  - Theta (time decay)
  - Correlation (dispersion-specific)
  - Residual

Workflow steps covered: 4.10, 6.2
"""

import pandas as pd
import numpy as np


class PnLAttribution:
    """
    Step 4.10: Design attribution methodology.
    Step 6.2: Track multi-strategy Profit and Loss attribution.
    """

    def __init__(self, strategies: list):
        self.strategies = strategies

    def compute_daily_attribution(self, vix_change: float = 0.5,
                                   realized_vol_change: float = 0.3,
                                   correlation_change: float = 0.0) -> pd.DataFrame:
        """
        Decompose a single day's Profit and Loss into Greek components.

        Parameters:
            vix_change: change in VIX (points)
            realized_vol_change: change in 20-day realized vol (points)
            correlation_change: change in average pairwise correlation
        """
        rows = []
        for strat in self.strategies:
            g = strat.greeks

            # Vega Profit and Loss: vega * delta_implied_vol
            vega_pnl = g["vega"] * vix_change

            # Gamma Profit and Loss: 0.5 * gamma * (delta_realized)^2
            gamma_pnl = 0.5 * g["gamma"] * (realized_vol_change ** 2)

            # Theta Profit and Loss: theta * 1 day
            theta_pnl = g["theta"]

            # Correlation Profit and Loss (only for dispersion)
            corr_pnl = 0.0
            if "Dispersion" in strat.name and correlation_change != 0:
                corr_sensitivity = -strat.capital * 0.15
                corr_pnl = corr_sensitivity * correlation_change

            total = vega_pnl + gamma_pnl + theta_pnl + corr_pnl
            residual = 0.0  # Would capture higher-order effects in production

            rows.append({
                "strategy": strat.name,
                "vega_pnl": round(vega_pnl, 2),
                "gamma_pnl": round(gamma_pnl, 2),
                "theta_pnl": round(theta_pnl, 2),
                "correlation_pnl": round(corr_pnl, 2),
                "residual_pnl": round(residual, 2),
                "total_pnl": round(total, 2),
            })

        df = pd.DataFrame(rows)

        # Portfolio total
        total_row = {
            "strategy": "PORTFOLIO TOTAL",
            "vega_pnl": df["vega_pnl"].sum(),
            "gamma_pnl": df["gamma_pnl"].sum(),
            "theta_pnl": df["theta_pnl"].sum(),
            "correlation_pnl": df["correlation_pnl"].sum(),
            "residual_pnl": df["residual_pnl"].sum(),
            "total_pnl": df["total_pnl"].sum(),
        }
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

        return df

    def print_report(self, vix_change: float = 0.5, realized_vol_change: float = 0.3):
        """Print daily attribution report."""
        df = self.compute_daily_attribution(vix_change, realized_vol_change)

        print("\n" + "=" * 105)
        print("DAILY PROFIT AND LOSS ATTRIBUTION (All values in United States dollars)")
        print(f"  VIX Change: {vix_change:+.1f} points  |  Realized Vol Change: {realized_vol_change:+.1f} points")
        print("=" * 105)

        print(f"\n{'Strategy':<35} {'Vega':>12} {'Gamma':>12} {'Theta':>12} "
              f"{'Correlation':>12} {'Total':>12}")
        print("-" * 95)

        for _, row in df.iterrows():
            name = row["strategy"]
            if name == "PORTFOLIO TOTAL":
                print("-" * 95)
            print(f"{name:<35} {row['vega_pnl']:>12,.0f} {row['gamma_pnl']:>12,.0f} "
                  f"{row['theta_pnl']:>12,.0f} {row['correlation_pnl']:>12,.0f} "
                  f"{row['total_pnl']:>12,.0f}")

        return df
