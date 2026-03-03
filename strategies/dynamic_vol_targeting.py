"""
Dynamic Volatility Targeting Strategy
======================================
Maintain portfolio at 12% annualized volatility target by adjusting
exposure based on 20-day realized volatility. Rebalance triggered when
realized vol deviates by more than 2 percentage points from target.

Workflow steps covered: 4.7, 6.3, 7.1–7.3
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class DynamicVolTargetingStrategy(BaseStrategy):
    """
    Dynamic Volatility Targeting — systematically scale portfolio exposure
    inversely to realized volatility to maintain a constant risk budget.

    Target leverage = Vol_Target / Realized_Vol
    """

    def __init__(self, capital: float, vol_target: float = 0.12,
                 rebalance_threshold: float = 0.02, lookback_days: int = 20):
        super().__init__("Dynamic Volatility Targeting", capital)
        self.vol_target = vol_target
        self.rebalance_threshold = rebalance_threshold
        self.lookback_days = lookback_days
        self.current_leverage = 1.0
        self.rebalance_count = 0
        self.transaction_cost_bps = 5           # 5 basis points per rebalance event

    def generate_signals(self, data: dict) -> pd.DataFrame:
        """
        Step 4.7: Specify volatility-targeting framework.
        Compute 20-day realized vol and determine if rebalancing is needed.
        """
        realized_vol_history = data["realized_vol_history"]

        # Extract index realized vol (last column matching index)
        index_rv_col = [c for c in realized_vol_history.columns if "SPX_INDEX_realized_vol" in c]
        if index_rv_col:
            rv_series = realized_vol_history[index_rv_col[0]].dropna()
        else:
            # Fallback: use regime signals
            regime_signals = data["vol_regime_signals"]
            rv_series = regime_signals["realized_vol_20d"].dropna() / 100

        current_rv = rv_series.iloc[-1] if len(rv_series) > 0 else self.vol_target
        if current_rv < 0.01:
            current_rv = current_rv  # Already in decimal
        elif current_rv > 1:
            current_rv = current_rv / 100  # Convert from percentage

        # Target leverage
        target_leverage = self.vol_target / max(current_rv, 0.05)
        target_leverage = np.clip(target_leverage, 0.3, 2.0)  # Floor and cap leverage

        # Deviation from current
        deviation = abs(current_rv - self.vol_target)
        needs_rebalance = deviation > self.rebalance_threshold

        # Historical rebalance frequency estimation
        if len(rv_series) > self.lookback_days:
            rv_deviations = abs(rv_series - self.vol_target)
            estimated_annual_rebalances = (rv_deviations > self.rebalance_threshold).sum()
            estimated_annual_rebalances = min(estimated_annual_rebalances, 252)
        else:
            estimated_annual_rebalances = 24  # Roughly biweekly

        # Transaction cost impact
        annual_tc_drag_bps = estimated_annual_rebalances * self.transaction_cost_bps

        signals = pd.DataFrame({
            "current_realized_vol": [round(current_rv, 4)],
            "vol_target": [self.vol_target],
            "deviation_pp": [round(deviation * 100, 2)],
            "target_leverage": [round(target_leverage, 3)],
            "needs_rebalance": [needs_rebalance],
            "estimated_annual_rebalances": [estimated_annual_rebalances],
            "estimated_annual_tc_drag_bps": [round(annual_tc_drag_bps, 1)],
            "entry_signal": [True],  # Always active
        })

        print(f"  Dynamic Volatility Targeting Signals:")
        print(f"    Current Realized Vol:       {current_rv:.4f}")
        print(f"    Volatility Target:          {self.vol_target:.4f}")
        print(f"    Deviation:                  {deviation*100:.2f} percentage points")
        print(f"    Target Leverage:            {target_leverage:.3f}x")
        print(f"    Rebalance Needed:           {needs_rebalance}")
        print(f"    Estimated Annual TC Drag:   {annual_tc_drag_bps:.0f} basis points")

        return signals

    def construct_positions(self, signals: pd.DataFrame) -> list:
        """
        Position = capital * leverage invested in equity futures / equity index exposure.
        The strategy adjusts equity exposure to target constant portfolio vol.
        """
        leverage = signals["target_leverage"].iloc[0]
        self.current_leverage = leverage

        equity_exposure = self.capital * leverage
        # The equity futures position generates delta and is vol-neutral
        # Vega comes indirectly from the rebalancing behavior

        # Effective vega from the rebalancing mechanism:
        # When vol rises, we de-lever (reduces exposure) — acts like short gamma
        # This is an implicit short vol position embedded in vol targeting
        implicit_short_gamma = -self.capital * 0.000005 * leverage

        self.positions = [
            {
                "leg": "EQUITY_FUTURES_EXPOSURE",
                "instrument": f"SPX equity futures, {leverage:.2f}x leverage",
                "direction": "LONG" if leverage > 0 else "FLAT",
                "notional_usd": round(equity_exposure, 2),
                "leverage": round(leverage, 3),
                "delta_usd": round(equity_exposure * 0.01, 2),     # Dollar delta per 1% SPX move
                "gamma_usd": round(implicit_short_gamma, 2),
                "vega_usd": 0.0,                                    # No direct vega
                "theta_usd": 0.0,
            }
        ]

        self.notional_deployed = abs(equity_exposure)
        self.greeks = self.compute_greeks()

        print(f"  Dynamic Volatility Targeting Positions:")
        print(f"    Leverage:            {leverage:.3f}x")
        print(f"    Equity Exposure:     ${equity_exposure:,.0f}")
        print(f"    Implicit Gamma:      ${implicit_short_gamma:,.0f}")

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
        """Vol targeting Profit and Loss is primarily driven by equity moves, not direct VIX."""
        if not self.positions:
            return 0.0

        # Under VIX spike, equities typically fall
        baseline_vix = 18.0
        vix_move = vix_level - baseline_vix

        # Rough equity-vol relationship: VIX +1 ≈ SPX -0.5%
        equity_move_pct = -vix_move * 0.005
        delta_pnl = self.greeks["delta"] * equity_move_pct * 100

        # Gamma from rebalancing lag
        gamma_pnl = 0.5 * self.greeks["gamma"] * (vix_move ** 2)

        return round(delta_pnl + gamma_pnl, 2)
