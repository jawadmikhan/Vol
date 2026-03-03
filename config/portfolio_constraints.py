"""
Portfolio Constraints — Investment Committee Mandate
====================================================
All parameters derived from the Systematic Volatility Portfolio Proposal
Delivery Plan. No abbreviations per project convention.
"""


# =============================================================================
# Capital and Notional Limits
# =============================================================================

TOTAL_CAPITAL = 250_000_000                     # $250 million total book
GROSS_NOTIONAL_CAP = 500_000_000                # $500 million across all strategies
SETTLEMENT_CURRENCY = "USD"                     # United States dollars

# =============================================================================
# Strategy Allocations (percentage of total capital)
# =============================================================================

ALLOCATIONS = {
    "dispersion": {
        "weight": 0.40,
        "capital": 100_000_000,                 # $100 million — maximum 40% of capital
        "description": "Index-constituent dispersion trading via correlation overpricing",
    },
    "volatility_harvesting": {
        "weight": 0.20,
        "capital": 50_000_000,                  # $50 million
        "short_variance_notional_cap": 80_000_000,  # $80 million variance notional limit
        "description": "Short variance capturing realized-versus-implied spread",
    },
    "directional_long_short": {
        "weight": 0.15,
        "capital": 37_500_000,                  # $37.5 million
        "description": "Regime-dependent directional volatility exposure",
    },
    "dynamic_vol_targeting": {
        "weight": 0.15,
        "capital": 37_500_000,                  # $37.5 million
        "description": "Maintain annualized volatility target via systematic rebalancing",
    },
    "option_overlay": {
        "weight": 0.10,
        "capital": 25_000_000,                  # $25 million
        "description": "Protective puts, collars, and tail-risk hedging",
    },
}

# =============================================================================
# Vega Constraints
# =============================================================================

NET_VEGA_FLOOR = -2_000_000                     # -$2 million per 1-point VIX move
NET_VEGA_CEILING = 4_000_000                    # +$4 million per 1-point VIX move

# =============================================================================
# Dispersion Sleeve Parameters
# =============================================================================

DISPERSION_MAX_WEIGHT = 0.40                    # Maximum 40% of total capital
DISPERSION_UNIVERSE_SIZE = 45                   # 45-name constituent universe
DISPERSION_ACTIVE_NAMES = 12                    # 12-name active dispersion book
DISPERSION_CORRELATION_LOOKBACK_DAYS = 90       # 90-day realized correlation window

# =============================================================================
# Volatility Harvesting Parameters
# =============================================================================

VOL_HARVEST_SHORT_VARIANCE_NOTIONAL_CAP = 80_000_000  # $80 million notional cap
VOL_HARVEST_MIN_EDGE_BPS = 200                  # Minimum 200 basis point realized-vs-implied edge

# =============================================================================
# Dynamic Volatility Targeting Parameters
# =============================================================================

VOL_TARGET_ANNUALIZED = 0.12                    # 12% annualized volatility target
VOL_TARGET_REBALANCE_THRESHOLD = 0.02           # Rebalance when 20-day realized deviates by >2pp
VOL_TARGET_LOOKBACK_DAYS = 20                   # 20-day realized volatility window

# =============================================================================
# Option Overlay Parameters
# =============================================================================

OVERLAY_MAX_ANNUAL_DRAG_BPS = 75                # Self-financing within 75 basis points annual drag
OVERLAY_PUT_TENOR_MONTHS = 3                    # 3-month rolling protective puts
OVERLAY_PUT_OTM_PERCENT = 0.05                  # 5% out-of-the-money puts
OVERLAY_FINANCING_CALL_DELTA = 0.10             # 10-delta financing calls
OVERLAY_DRAWDOWN_SURVIVAL = 0.30                # Tail-risk hedge must survive 30% drawdown

# =============================================================================
# Scenario Analysis Parameters
# =============================================================================

SCENARIO_VIX_SPIKE = 45                         # VIX spike scenario level
SCENARIO_VIX_COLLAPSE = 11                      # VIX collapse scenario level
SCENARIO_VIX_BASELINE = 18                      # Current baseline VIX assumption

# =============================================================================
# Implementation Timeline
# =============================================================================

IMPLEMENTATION_DAYS = 60                        # 60-day phased build-up
IMPLEMENTATION_PHASES = {
    "phase_1": {"weeks": "1-2", "days": (1, 14),   "strategy": "dispersion",            "description": "Execute index and constituent option legs"},
    "phase_2": {"weeks": "3-4", "days": (15, 28),  "strategy": "volatility_harvesting",  "description": "Establish variance swap positions"},
    "phase_3": {"weeks": "5-6", "days": (29, 42),  "strategy": "directional_long_short", "description": "Build directional volatility book"},
    "phase_4": {"weeks": "7-8", "days": (43, 56),  "strategy": "dynamic_vol_targeting",  "description": "Activate volatility targeting framework"},
    "phase_5": {"weeks": "8-9", "days": (50, 60),  "strategy": "option_overlay",          "description": "Layer protective puts and collar overlays"},
}

# =============================================================================
# Regime Classification Thresholds
# =============================================================================

REGIME_LOW_VOL_VIX_UPPER = 16                   # VIX below 16 = low-volatility harvesting regime
REGIME_TRANSITIONAL_VIX_UPPER = 25              # VIX 16-25 = transitional regime
REGIME_CRISIS_VIX_LOWER = 25                    # VIX above 25 = crisis regime

# VIX term structure regime signals
REGIME_CONTANGO_THRESHOLD = 0.02                # Front month > 2% below second month = contango
REGIME_BACKWARDATION_THRESHOLD = -0.02          # Front month > 2% above second month = backwardation

# =============================================================================
# Risk Limits
# =============================================================================

MAX_PORTFOLIO_DRAWDOWN = 0.15                   # 15% maximum portfolio drawdown
MARGIN_REQUIREMENT_CAP = 180_000_000            # $180 million total margin across all strategies
POSITION_STOP_LOSS_DEBIT = 1.5                  # Stop at 1.5x entry debit (long positions)
POSITION_STOP_LOSS_CREDIT = 2.0                 # Stop at 2x entry credit (short positions)
