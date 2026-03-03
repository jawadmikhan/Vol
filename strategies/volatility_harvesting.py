"""
Volatility Harvesting Strategy
===============================
Short variance swaps to capture the structural realized-versus-implied
volatility spread. Includes convexity adjustment between variance notional
and vega notional.

Workflow steps covered: 1.7, 2.6, 4.4
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class VolatilityHarvestingStrategy(BaseStrategy):
    """
    Volatility Harvesting — systematically sell variance to capture the
    volatility risk premium (implied > realized in most regimes).
    """

    def __init__(self, capital: float, short_variance_notional_cap: float = 80_000_000):
        super().__init__("Volatility Harvesting", capital)
        self.short_variance_notional_cap = short_variance_notional_cap
        self.min_edge_bps = 200                 # Minimum 200 basis point edge to enter
        self.variance_strike = None
        self.vega_notional = None

    def generate_signals(self, data: dict) -> pd.DataFrame:
        """
        Step 1.7: Screen volatility harvesting opportunities.
        Compute realized-versus-implied spread and assess edge.
        """
        regime_signals = data["vol_regime_signals"]
        vol_surface = data["implied_vol_surface"]

        # Current regime data (last observation)
        latest = regime_signals.dropna().iloc[-1]
        rv_iv_spread = latest.get("rv_iv_spread", 0)
        current_regime = latest.get("regime", "TRANSITIONAL")
        vix_level = latest.get("vix_front_month", 18)

        # Index 3-month at-the-money implied vol
        atm_3m_index = vol_surface[
            (vol_surface["name"] == "SPX_INDEX") &
            (vol_surface["strike_delta"] == "ATM") &
            (vol_surface["tenor_months"] == 3)
        ]
        implied_vol = atm_3m_index["implied_vol"].values[0] if len(atm_3m_index) > 0 else 0.18

        # Historical realized vol (annualized from regime signals)
        recent_rv = regime_signals["realized_vol_20d"].dropna().tail(60)
        realized_vol = recent_rv.mean() / 100 if len(recent_rv) > 0 else 0.15

        # Edge in basis points
        edge_bps = (implied_vol - realized_vol) * 10_000

        # Entry signal: edge exceeds minimum and not in crisis regime
        entry = edge_bps >= self.min_edge_bps and current_regime != "CRISIS"

        signals = pd.DataFrame({
            "implied_vol": [implied_vol],
            "realized_vol": [realized_vol],
            "edge_bps": [round(edge_bps, 1)],
            "regime": [current_regime],
            "vix_level": [vix_level],
            "entry_signal": [entry],
        })

        print(f"  Volatility Harvesting Signals:")
        print(f"    Implied Vol (3-month at-the-money): {implied_vol:.4f}")
        print(f"    Realized Vol (20-day average):      {realized_vol:.4f}")
        print(f"    Edge:                               {edge_bps:.0f} basis points")
        print(f"    Entry Signal:                       {entry}")

        return signals

    def construct_positions(self, signals: pd.DataFrame) -> list:
        """
        Step 4.4: Design volatility harvesting positions.
        Short variance swap with convexity adjustment.

        Key practitioner insight: variance swap pays on realized VARIANCE,
        not volatility. The convexity adjustment matters for sizing:
          Vega Notional = Variance Notional / (2 * Strike)
        """
        if not signals["entry_signal"].iloc[0]:
            print("  No entry signal — volatility harvesting book flat.")
            self.positions = []
            return self.positions

        implied_vol = signals["implied_vol"].iloc[0]
        regime = signals["regime"].iloc[0]

        # Variance strike (squared implied vol, expressed in vol points)
        self.variance_strike = implied_vol * 100  # e.g., 18.0 vol points

        # Size: regime-dependent fraction of notional cap
        regime_sizing = {
            "LOW_VOL_HARVESTING": 0.90,
            "TRANSITIONAL": 0.60,
            "CRISIS": 0.0,
        }
        size_factor = regime_sizing.get(regime, 0.60)

        variance_notional = self.short_variance_notional_cap * size_factor

        # Convexity adjustment: Vega Notional = Variance Notional / (2 * Strike)
        # This is the critical practitioner distinction — models often conflate these
        self.vega_notional = variance_notional / (2 * self.variance_strike)

        # Dollar vega (sensitivity per 1 vol point move)
        dollar_vega = -self.vega_notional  # Short vega

        # Dollar gamma (variance swaps have constant dollar gamma)
        # Gamma = 2 * Variance Notional / S^2 (simplified for index level)
        index_level = 4500
        dollar_gamma = -2 * variance_notional / (index_level ** 2)

        # Theta: short variance earns theta = (implied^2 - realized^2) * notional / 365
        realized_vol = signals["realized_vol"].iloc[0]
        daily_theta = (implied_vol**2 - realized_vol**2) * variance_notional / 365

        self.positions = [
            {
                "leg": "SHORT_VARIANCE_SWAP",
                "instrument": f"SPX 3-month variance swap, strike {self.variance_strike:.1f}",
                "direction": "SHORT",
                "variance_notional_usd": round(variance_notional, 2),
                "vega_notional_usd": round(self.vega_notional, 2),
                "convexity_adjustment_ratio": round(self.vega_notional / variance_notional, 6),
                "delta_usd": 0.0,
                "gamma_usd": round(dollar_gamma, 2),
                "vega_usd": round(dollar_vega, 2),
                "theta_usd": round(daily_theta, 2),
            }
        ]

        self.notional_deployed = variance_notional
        self.greeks = self.compute_greeks()

        print(f"  Volatility Harvesting Positions Constructed:")
        print(f"    Variance Notional:       ${variance_notional:,.0f}")
        print(f"    Variance Strike:          {self.variance_strike:.1f} vol points")
        print(f"    Vega Notional:           ${self.vega_notional:,.0f}")
        print(f"    Convexity Adjustment:     {self.vega_notional / variance_notional:.4f}")
        print(f"    Dollar Vega:             ${dollar_vega:,.0f}")
        print(f"    Daily Theta:             ${daily_theta:,.0f}")

        return self.positions

    def compute_greeks(self) -> dict:
        greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for pos in self.positions:
            greeks["delta"] += pos["delta_usd"]
            greeks["gamma"] += pos["gamma_usd"]
            greeks["vega"] += pos["vega_usd"]
            greeks["theta"] += pos["theta_usd"]
        self.greeks = greeks
        return greeks

    def run_scenario(self, vix_level: float, correlation_shock: float = 0.0) -> float:
        """
        Variance swap Profit and Loss under VIX scenario.
        Short variance is hurt by realized vol exceeding strike.
        Convexity means losses accelerate — variance swap pays on vol^2.
        """
        if not self.positions:
            return 0.0

        baseline_vix = 18.0
        vix_move = vix_level - baseline_vix

        # Linear vega component
        vega_pnl = self.greeks["vega"] * vix_move

        # Convexity component (variance payoff is quadratic)
        # Loss = Notional * (realized_var - strike_var) / strike_var
        # Approximated as gamma * move^2
        convexity_pnl = 0.5 * self.greeks["gamma"] * (vix_move ** 2) * 10000

        total_pnl = vega_pnl + convexity_pnl
        return round(total_pnl, 2)
