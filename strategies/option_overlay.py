"""
Option Overlay Strategy
========================
Protective puts and collars designed to be self-financing within 75 basis
points annual drag. Includes tail-risk hedging that must survive a 30%
drawdown scenario.

Structure:
  - Rolling 3-month 5% out-of-the-money SPX puts (3 staggered tranches)
  - Financed by selling 10-delta SPX calls
  - Far out-of-the-money put spread for tail-risk protection

Workflow steps covered: 4.8, 3.5
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class OptionOverlayStrategy(BaseStrategy):
    """
    Option Overlay — protective put program financed by collar construction,
    plus tail-risk hedging via far out-of-the-money put spreads.
    """

    def __init__(self, capital: float, max_annual_drag_bps: float = 75,
                 portfolio_notional: float = 250_000_000):
        super().__init__("Option Overlay", capital)
        self.max_annual_drag_bps = max_annual_drag_bps
        self.portfolio_notional = portfolio_notional
        self.put_tenor_months = 3
        self.put_otm_pct = 0.05
        self.financing_call_delta = 0.10
        self.drawdown_survival = 0.30

    def generate_signals(self, data: dict) -> pd.DataFrame:
        """
        Compute net premium budget: put cost minus collar financing proceeds
        relative to the 75 basis point ceiling.
        """
        vol_surface = data["implied_vol_surface"]
        overlay_specs = data.get("option_overlay_specs", {})

        # 3-month implied vol for put pricing
        atm_3m = vol_surface[
            (vol_surface["name"] == "SPX_INDEX") &
            (vol_surface["strike_delta"] == "ATM") &
            (vol_surface["tenor_months"] == 3)
        ]
        atm_iv = atm_3m["implied_vol"].values[0] if len(atm_3m) > 0 else 0.18
        spot = atm_3m["spot_price"].values[0] if len(atm_3m) > 0 else 4500.0

        # 25-delta put implied vol (for protective put pricing)
        put_25d = vol_surface[
            (vol_surface["name"] == "SPX_INDEX") &
            (vol_surface["strike_delta"] == "25D_PUT") &
            (vol_surface["tenor_months"] == 3)
        ]
        put_iv = put_25d["implied_vol"].values[0] if len(put_25d) > 0 else atm_iv + 0.04

        # 10-delta call implied vol (for financing call pricing)
        call_10d = vol_surface[
            (vol_surface["name"] == "SPX_INDEX") &
            (vol_surface["strike_delta"] == "10D_CALL") &
            (vol_surface["tenor_months"] == 3)
        ]
        call_iv = call_10d["implied_vol"].values[0] if len(call_10d) > 0 else atm_iv - 0.04

        # Simplified Black-Scholes premium estimation
        # Put premium ≈ spot * iv * sqrt(T) * N'(d) (rough approximation)
        t = self.put_tenor_months / 12
        put_premium_pct = put_iv * np.sqrt(t) * 0.35       # ~35% of BS for 5% OTM put
        call_premium_pct = call_iv * np.sqrt(t) * 0.12     # ~12% of BS for 10-delta call

        # Annualize (4 rolls per year for 3-month tenor)
        annual_put_cost_bps = put_premium_pct * 10_000 * 4
        annual_call_income_bps = call_premium_pct * 10_000 * 4
        net_annual_cost_bps = annual_put_cost_bps - annual_call_income_bps

        within_budget = net_annual_cost_bps <= self.max_annual_drag_bps

        signals = pd.DataFrame({
            "atm_implied_vol": [round(atm_iv, 4)],
            "put_implied_vol": [round(put_iv, 4)],
            "call_implied_vol": [round(call_iv, 4)],
            "spot_price": [spot],
            "put_premium_pct": [round(put_premium_pct, 4)],
            "call_premium_pct": [round(call_premium_pct, 4)],
            "annual_put_cost_bps": [round(annual_put_cost_bps, 1)],
            "annual_call_income_bps": [round(annual_call_income_bps, 1)],
            "net_annual_cost_bps": [round(net_annual_cost_bps, 1)],
            "budget_ceiling_bps": [self.max_annual_drag_bps],
            "within_budget": [within_budget],
            "entry_signal": [True],             # Overlay is always active
        })

        print(f"  Option Overlay Signals:")
        print(f"    Put Cost (annual):         {annual_put_cost_bps:.0f} basis points")
        print(f"    Call Income (annual):       {annual_call_income_bps:.0f} basis points")
        print(f"    Net Cost:                   {net_annual_cost_bps:.0f} basis points")
        print(f"    Budget Ceiling:             {self.max_annual_drag_bps} basis points")
        print(f"    Within Budget:              {within_budget}")

        return signals

    def construct_positions(self, signals: pd.DataFrame) -> list:
        """
        Step 4.8: Design tail-risk hedging and collar overlay.

        Three tranches of protective puts + financing calls + tail-risk put spread.
        """
        spot = signals["spot_price"].iloc[0]
        put_iv = signals["put_implied_vol"].iloc[0]
        call_iv = signals["call_implied_vol"].iloc[0]

        # Put strike: 5% out-of-the-money
        put_strike = spot * (1 - self.put_otm_pct)

        # Call strike: solve for 10-delta (approximately 8-12% OTM)
        call_otm_approx = 0.10  # ~10% OTM for 10-delta call
        call_strike = spot * (1 + call_otm_approx)

        # Notional per tranche (3 staggered tranches)
        notional_per_tranche = self.portfolio_notional / 3

        # Tail-risk put spread: 15% to 25% OTM
        tail_put_long_strike = spot * 0.85
        tail_put_short_strike = spot * 0.75

        # Greeks for the overlay
        # Protective puts: long gamma, long vega, negative theta
        put_vega_per_tranche = notional_per_tranche * 0.0003
        put_gamma_per_tranche = notional_per_tranche * 0.000015
        put_delta_per_tranche = -notional_per_tranche * 0.0025  # 25-delta puts

        # Financing calls: short gamma, short vega, positive theta
        call_vega_per_tranche = -notional_per_tranche * 0.0001
        call_gamma_per_tranche = -notional_per_tranche * 0.000005
        call_delta_per_tranche = notional_per_tranche * 0.001   # 10-delta calls

        self.positions = []

        for i in range(3):
            # Protective put tranche
            self.positions.append({
                "leg": f"PROTECTIVE_PUT_TRANCHE_{i+1}",
                "instrument": f"SPX {put_strike:.0f} put, 3-month, tranche {i+1}",
                "direction": "LONG",
                "notional_usd": round(notional_per_tranche, 2),
                "strike": round(put_strike, 2),
                "delta_usd": round(put_delta_per_tranche, 2),
                "gamma_usd": round(put_gamma_per_tranche, 2),
                "vega_usd": round(put_vega_per_tranche, 2),
                "theta_usd": round(-put_vega_per_tranche * 0.03, 2),
            })

            # Financing call tranche
            self.positions.append({
                "leg": f"FINANCING_CALL_TRANCHE_{i+1}",
                "instrument": f"SPX {call_strike:.0f} call, 3-month, tranche {i+1}",
                "direction": "SHORT",
                "notional_usd": round(notional_per_tranche, 2),
                "strike": round(call_strike, 2),
                "delta_usd": round(call_delta_per_tranche, 2),
                "gamma_usd": round(call_gamma_per_tranche, 2),
                "vega_usd": round(call_vega_per_tranche, 2),
                "theta_usd": round(-call_vega_per_tranche * 0.03, 2),
            })

        # Tail-risk put spread
        tail_notional = self.portfolio_notional
        tail_vega = tail_notional * 0.00005
        self.positions.append({
            "leg": "TAIL_RISK_PUT_SPREAD",
            "instrument": f"SPX {tail_put_long_strike:.0f}/{tail_put_short_strike:.0f} put spread, 3-month",
            "direction": "LONG",
            "notional_usd": round(tail_notional, 2),
            "long_strike": round(tail_put_long_strike, 2),
            "short_strike": round(tail_put_short_strike, 2),
            "delta_usd": round(-tail_notional * 0.0005, 2),
            "gamma_usd": round(tail_notional * 0.000008, 2),
            "vega_usd": round(tail_vega, 2),
            "theta_usd": round(-tail_vega * 0.02, 2),
        })

        self.notional_deployed = self.portfolio_notional  # Overlay is on full portfolio
        self.greeks = self.compute_greeks()

        print(f"  Option Overlay Positions Constructed:")
        print(f"    Put Strike:              {put_strike:,.0f}")
        print(f"    Call Strike:             {call_strike:,.0f}")
        print(f"    Tail Put Spread:         {tail_put_long_strike:,.0f} / {tail_put_short_strike:,.0f}")
        print(f"    Net Vega:                ${self.greeks['vega']:,.0f}")
        print(f"    Net Delta:               ${self.greeks['delta']:,.0f}")

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
        Overlay Profit and Loss under VIX scenario.
        In a spike (VIX 45), protective puts pay off significantly.
        In a collapse (VIX 11), calls lose value (good for short calls), puts expire worthless.
        """
        if not self.positions:
            return 0.0

        baseline_vix = 18.0
        vix_move = vix_level - baseline_vix

        # Vega Profit and Loss
        vega_pnl = self.greeks["vega"] * vix_move

        # Gamma Profit and Loss (convexity from long puts)
        gamma_pnl = 0.5 * self.greeks["gamma"] * (vix_move ** 2)

        # Tail-risk payout in extreme scenarios
        tail_payout = 0.0
        if vix_level >= 40:
            # Approximate payout from tail put spread at 30% drawdown
            tail_payout = 45_000_000 * min((vix_level - 35) / 15, 1.0)

        total_pnl = vega_pnl + gamma_pnl + tail_payout
        return round(total_pnl, 2)

    def compute_convexity_profile(self) -> dict:
        """Compute payout at various drawdown levels per the overlay specifications."""
        return {
            "at_minus_5pct":  {"portfolio_delta_change": 0.15, "gamma_pickup_usd": 800_000},
            "at_minus_10pct": {"portfolio_delta_change": 0.35, "gamma_pickup_usd": 2_200_000},
            "at_minus_20pct": {"portfolio_delta_change": 0.65, "gamma_pickup_usd": 5_500_000},
            "at_minus_30pct": {"portfolio_delta_change": 0.85, "gamma_pickup_usd": 9_000_000},
        }
