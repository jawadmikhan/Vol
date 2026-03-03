"""
Dispersion Trading Strategy
============================
Core strategy: exploit structural mispricing between index and constituent
implied volatility via correlation overpricing.

Entry requires decomposing index implied volatility into a weighted sum of
constituent implied volatilities adjusted for realized pairwise correlations.

Workflow steps covered: 1.1–1.5, 2.1–2.4, 4.2–4.3
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class DispersionStrategy(BaseStrategy):
    """
    Dispersion Trading — sell index volatility, buy constituent volatility
    when implied correlation exceeds realized correlation by a statistically
    significant margin.
    """

    def __init__(self, capital: float, universe_size: int = 45, active_names: int = 12):
        super().__init__("Dispersion Trading", capital)
        self.universe_size = universe_size
        self.active_names = active_names
        self.implied_correlation = None
        self.realized_correlation = None
        self.dispersion_z_score = None
        self.entry_threshold_z = 1.5           # Enter when Z-score > 1.5
        self.exit_threshold_z = 0.5            # Exit when Z-score < 0.5

    def generate_signals(self, data: dict) -> pd.DataFrame:
        """
        Step 1.2: Compute realized versus implied correlation.
        Step 1.3: Calculate dispersion Z-score.
        Step 1.5: Generate dispersion entry signals.

        Implied correlation = (index_var - sum(w_i^2 * sigma_i^2)) / (2 * sum(w_i * w_j * sigma_i * sigma_j))
        """
        vol_surface = data["implied_vol_surface"]
        corr_matrix = data["correlation_matrix"]
        regime_signals = data["vol_regime_signals"]

        # Extract at-the-money, 3-month implied vols
        atm_3m = vol_surface[
            (vol_surface["strike_delta"] == "ATM") & (vol_surface["tenor_months"] == 3)
        ].set_index("name")

        # Index implied vol
        index_iv = atm_3m.loc["SPX_INDEX", "implied_vol"]

        # Constituent implied vols and weights
        constituents = atm_3m.drop("SPX_INDEX", errors="ignore")
        weights = constituents["index_weight"].values
        weights = weights / weights.sum()
        sigma_i = constituents["implied_vol"].values

        # Implied correlation via variance decomposition
        # Index variance = sum(w_i^2 * sigma_i^2) + 2 * sum(w_i * w_j * sigma_i * sigma_j * rho_implied)
        # Solving for rho_implied:
        weighted_var_sum = np.sum(weights**2 * sigma_i**2)
        cross_term_sum = 0.0
        n = len(weights)
        for i in range(n):
            for j in range(i + 1, n):
                cross_term_sum += weights[i] * weights[j] * sigma_i[i] * sigma_i[j]

        if cross_term_sum > 0:
            self.implied_correlation = (index_iv**2 - weighted_var_sum) / (2 * cross_term_sum)
        else:
            self.implied_correlation = 0.5

        self.implied_correlation = np.clip(self.implied_correlation, 0.0, 1.0)

        # Realized correlation from correlation matrix (average pairwise)
        corr_values = corr_matrix.values
        n_assets = corr_values.shape[0]
        upper_tri = corr_values[np.triu_indices(n_assets, k=1)]
        self.realized_correlation = float(np.mean(upper_tri))

        # Dispersion Z-score
        correlation_spread = self.implied_correlation - self.realized_correlation
        spread_std = max(0.05, np.std(upper_tri) * 0.5)  # Estimated spread volatility
        self.dispersion_z_score = correlation_spread / spread_std

        # Select top names by idiosyncratic vol (highest implied minus realized spread)
        regime_data = regime_signals.iloc[-1] if len(regime_signals) > 0 else {}
        current_regime = regime_data.get("regime", "TRANSITIONAL")

        # Build signal dataframe
        signals = pd.DataFrame({
            "implied_correlation": [self.implied_correlation],
            "realized_correlation": [self.realized_correlation],
            "correlation_spread": [correlation_spread],
            "dispersion_z_score": [self.dispersion_z_score],
            "regime": [current_regime],
            "entry_signal": [self.dispersion_z_score > self.entry_threshold_z],
            "exit_signal": [self.dispersion_z_score < self.exit_threshold_z],
        })

        print(f"  Dispersion Signals:")
        print(f"    Implied Correlation:  {self.implied_correlation:.4f}")
        print(f"    Realized Correlation: {self.realized_correlation:.4f}")
        print(f"    Spread:               {correlation_spread:.4f}")
        print(f"    Z-Score:              {self.dispersion_z_score:.2f}")
        print(f"    Entry Signal:         {bool(signals['entry_signal'].iloc[0])}")
        print(f"    Current Regime:       {current_regime}")

        return signals

    def construct_positions(self, signals: pd.DataFrame) -> list:
        """
        Step 4.2: Design index short volatility leg.
        Step 4.3: Design constituent long volatility leg.

        Position structure:
        - SHORT index straddles/variance at at-the-money, 3-month tenor
        - LONG constituent straddles on top 12 names ranked by vol spread
        """
        if not signals["entry_signal"].iloc[0]:
            print("  No entry signal — dispersion book flat.")
            self.positions = []
            return self.positions

        # Size based on regime
        regime = signals["regime"].iloc[0]
        regime_sizing = {
            "LOW_VOL_HARVESTING": 1.0,         # Full size in low vol
            "TRANSITIONAL": 0.7,                # Reduced in transitional
            "CRISIS": 0.3,                      # Minimal in crisis (correlation spikes)
        }
        size_factor = regime_sizing.get(regime, 0.7)

        # Index short leg
        index_notional = self.capital * 0.60 * size_factor
        index_vega = -index_notional * 0.0004   # Approximate vega per dollar notional

        # Constituent long legs (spread across 12 names)
        constituent_notional_each = (self.capital * 0.40 * size_factor) / self.active_names
        constituent_vega_each = constituent_notional_each * 0.0006  # Constituents have higher vega per dollar

        self.positions = [
            {
                "leg": "INDEX_SHORT",
                "instrument": "SPX 3-month at-the-money straddle",
                "direction": "SHORT",
                "notional_usd": round(index_notional, 2),
                "delta_usd": 0.0,                # Delta-hedged
                "gamma_usd": round(-index_notional * 0.00002, 2),
                "vega_usd": round(index_vega, 2),
                "theta_usd": round(-index_vega * 0.04, 2),  # Theta roughly 4% of vega daily
            }
        ]

        for i in range(self.active_names):
            self.positions.append({
                "leg": f"CONSTITUENT_LONG_{i+1}",
                "instrument": f"Constituent_{i+1} 3-month at-the-money straddle",
                "direction": "LONG",
                "notional_usd": round(constituent_notional_each, 2),
                "delta_usd": 0.0,
                "gamma_usd": round(constituent_notional_each * 0.00003, 2),
                "vega_usd": round(constituent_vega_each, 2),
                "theta_usd": round(-constituent_vega_each * 0.04, 2),
            })

        self.notional_deployed = index_notional + constituent_notional_each * self.active_names
        self.greeks = self.compute_greeks()

        print(f"  Dispersion Positions Constructed:")
        print(f"    Index short notional:      ${index_notional:,.0f}")
        print(f"    Constituent long notional:  ${constituent_notional_each * self.active_names:,.0f}")
        print(f"    Total notional deployed:    ${self.notional_deployed:,.0f}")
        print(f"    Net Vega:                   ${self.greeks['vega']:,.0f}")

        return self.positions

    def compute_greeks(self) -> dict:
        """Aggregate Greeks across all dispersion legs in dollar terms."""
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
        Estimate Profit and Loss under VIX scenario.
        In a VIX spike, correlations increase — hurts the short-correlation dispersion trade.
        In a VIX collapse, correlations decrease — benefits the trade.
        """
        baseline_vix = 18.0
        vix_move = vix_level - baseline_vix

        # Vega Profit and Loss
        vega_pnl = self.greeks["vega"] * vix_move

        # Correlation shock impact (dispersion is short correlation)
        # Higher correlation = index vol rises relative to constituents = loss
        corr_sensitivity = -self.capital * 0.15  # Approximate correlation vega
        corr_pnl = corr_sensitivity * correlation_shock

        # Gamma Profit and Loss (convexity from constituent longs partially offsets)
        gamma_pnl = 0.5 * self.greeks["gamma"] * (vix_move ** 2)

        total_pnl = vega_pnl + corr_pnl + gamma_pnl
        return round(total_pnl, 2)
