"""
Directional Long/Short Volatility Strategy
============================================
Regime-dependent directional volatility exposure using VIX term structure,
VVIX, and realized-versus-implied spread for classification.

Three regimes with distinct position sizing rules:
  1. Low-Volatility Harvesting: short volatility bias
  2. Transitional: neutral / reduced exposure
  3. Crisis: long volatility bias

Workflow steps covered: 1.4, 1.6, 2.6, 3.4, 4.5
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class DirectionalLongShortStrategy(BaseStrategy):
    """
    Directional Long/Short Volatility — take directional vol views based
    on regime classification using VIX term structure, VVIX, and the
    ratio of realized-to-implied volatility.
    """

    def __init__(self, capital: float):
        super().__init__("Directional Long/Short Volatility", capital)
        self.current_regime = None
        self.regime_history = []
        self.stop_loss_credit = 2.0             # Stop at 2x entry credit for shorts
        self.stop_loss_debit = 1.5              # Stop at 1.5x entry debit for longs

    def generate_signals(self, data: dict) -> pd.DataFrame:
        """
        Step 1.4: Classify current volatility regime.
        Uses three inputs: VIX level, term structure slope, VVIX.

        Regime classification:
          LOW_VOL_HARVESTING: VIX < 16, contango, VVIX < 90
          TRANSITIONAL:       VIX 16-25, mixed signals
          CRISIS:             VIX > 25 OR backwardation OR VVIX > 120
        """
        regime_signals = data["vol_regime_signals"]

        # Use last 20 days for regime assessment (avoid single-day noise)
        recent = regime_signals.dropna().tail(20)

        if len(recent) == 0:
            self.current_regime = "TRANSITIONAL"
            return pd.DataFrame({"regime": ["TRANSITIONAL"], "entry_signal": [False]})

        avg_vix = recent["vix_front_month"].mean()
        avg_slope = recent["term_structure_slope"].mean()
        avg_vvix = recent["vvix"].mean()
        avg_rv_iv = recent["rv_iv_spread"].mean()
        latest_regime = recent["regime"].mode().iloc[0] if len(recent["regime"].mode()) > 0 else "TRANSITIONAL"

        # Multi-factor regime classification
        crisis_score = 0
        if avg_vix > 25:
            crisis_score += 2
        if avg_slope < -0.02:
            crisis_score += 2
        if avg_vvix > 120:
            crisis_score += 1
        if avg_rv_iv > 3:
            crisis_score += 1

        low_vol_score = 0
        if avg_vix < 16:
            low_vol_score += 2
        if avg_slope > 0.03:
            low_vol_score += 1
        if avg_vvix < 85:
            low_vol_score += 1

        if crisis_score >= 3:
            self.current_regime = "CRISIS"
        elif low_vol_score >= 3:
            self.current_regime = "LOW_VOL_HARVESTING"
        else:
            self.current_regime = "TRANSITIONAL"

        # Directional signal
        if self.current_regime == "CRISIS":
            direction = "LONG_VOL"
        elif self.current_regime == "LOW_VOL_HARVESTING":
            direction = "SHORT_VOL"
        else:
            direction = "NEUTRAL"

        signals = pd.DataFrame({
            "avg_vix": [round(avg_vix, 2)],
            "avg_slope": [round(avg_slope, 4)],
            "avg_vvix": [round(avg_vvix, 2)],
            "avg_rv_iv_spread": [round(avg_rv_iv, 2)],
            "crisis_score": [crisis_score],
            "low_vol_score": [low_vol_score],
            "regime": [self.current_regime],
            "direction": [direction],
            "entry_signal": [direction != "NEUTRAL"],
        })

        print(f"  Directional Long/Short Signals:")
        print(f"    Average VIX (20-day):     {avg_vix:.2f}")
        print(f"    Term Structure Slope:      {avg_slope:.4f}")
        print(f"    Average VVIX:              {avg_vvix:.2f}")
        print(f"    Regime:                    {self.current_regime}")
        print(f"    Direction:                 {direction}")

        return signals

    def construct_positions(self, signals: pd.DataFrame) -> list:
        """
        Step 4.5: Design directional volatility positions.
        Position sizing varies by regime.
        """
        direction = signals["direction"].iloc[0]

        if direction == "NEUTRAL":
            print("  Neutral regime — directional book flat.")
            self.positions = []
            return self.positions

        regime = signals["regime"].iloc[0]
        vix = signals["avg_vix"].iloc[0]

        # Position sizing by regime
        if direction == "LONG_VOL":
            # Crisis: buy VIX calls or long variance
            notional = self.capital * 0.80          # Aggressive in crisis
            instrument = "VIX 1-month 25-delta call spreads"
            vega_per_dollar = 0.0008                # Higher vega sensitivity for calls
            sign = 1

        elif direction == "SHORT_VOL":
            # Low vol: sell VIX puts or short straddles
            notional = self.capital * 0.50          # Conservative sizing for shorts
            instrument = "SPX 1-month at-the-money short straddle"
            vega_per_dollar = 0.0005
            sign = -1

        else:
            notional = 0
            instrument = "FLAT"
            vega_per_dollar = 0
            sign = 0

        dollar_vega = sign * notional * vega_per_dollar
        dollar_gamma = sign * notional * 0.00002    # Approximate
        dollar_theta = -sign * abs(dollar_vega) * 0.035  # Theta as fraction of vega

        # Entry price for stop-loss tracking
        entry_credit_debit = abs(dollar_vega) * vix * 0.01

        self.positions = [
            {
                "leg": f"DIRECTIONAL_{direction}",
                "instrument": instrument,
                "direction": direction,
                "notional_usd": round(notional, 2),
                "delta_usd": 0.0,
                "gamma_usd": round(dollar_gamma, 2),
                "vega_usd": round(dollar_vega, 2),
                "theta_usd": round(dollar_theta, 2),
                "entry_vix": round(vix, 2),
                "stop_loss_level": round(
                    vix * (self.stop_loss_credit if sign < 0 else 1 / self.stop_loss_debit), 2
                ),
            }
        ]

        self.notional_deployed = notional
        self.greeks = self.compute_greeks()

        print(f"  Directional Positions Constructed:")
        print(f"    Direction:           {direction}")
        print(f"    Notional:            ${notional:,.0f}")
        print(f"    Dollar Vega:         ${dollar_vega:,.0f}")
        print(f"    Stop-Loss VIX:       {self.positions[0]['stop_loss_level']}")

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
        if not self.positions:
            return 0.0

        entry_vix = self.positions[0].get("entry_vix", 18.0)
        vix_move = vix_level - entry_vix

        vega_pnl = self.greeks["vega"] * vix_move
        gamma_pnl = 0.5 * self.greeks["gamma"] * (vix_move ** 2)

        return round(vega_pnl + gamma_pnl, 2)
