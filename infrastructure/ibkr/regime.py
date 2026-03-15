"""
Compute volatility regime signals from live IBKR data.

Produces the same regime classification used by the strategies:
  LOW_VOL_HARVESTING, TRANSITIONAL, CRISIS

Matches the vol_regime_signals.csv schema:
  date, vix_front_month, vix_second_month, term_structure_slope,
  vvix, realized_vol_20d, rv_iv_spread, regime
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def classify_regime(
    vix: float,
    term_slope: float,
    vvix: float | None,
    rv_iv_spread: float,
) -> str:
    """
    Multi-factor regime classification matching directional_long_short.py logic.

    CRISIS if score >= 3:
        VIX > 25 (+2), term_slope < -2% (+2), VVIX > 120 (+1), RV/IV > 3% (+1)

    LOW_VOL_HARVESTING if score >= 3:
        VIX < 16 (+2), term_slope > 3% (+1), VVIX < 85 (+1)

    Otherwise: TRANSITIONAL
    """
    # Crisis scoring
    crisis_score = 0
    if vix > 25:
        crisis_score += 2
    if term_slope < -2.0:
        crisis_score += 2
    if vvix is not None and vvix > 120:
        crisis_score += 1
    if rv_iv_spread > 0.03:
        crisis_score += 1

    if crisis_score >= 3:
        return "CRISIS"

    # Low-vol scoring
    low_vol_score = 0
    if vix < 16:
        low_vol_score += 2
    if term_slope > 3.0:
        low_vol_score += 1
    if vvix is not None and vvix < 85:
        low_vol_score += 1

    if low_vol_score >= 3:
        return "LOW_VOL_HARVESTING"

    return "TRANSITIONAL"


def compute_realized_vol(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute annualized realized vol from a price DataFrame.

    Args:
        prices: DataFrame with date index and 'SPX_INDEX' column (or similar).
        window: Rolling window in trading days.

    Returns:
        Series of annualized realized volatility.
    """
    if "SPX_INDEX" in prices.columns:
        col = "SPX_INDEX"
    else:
        col = prices.columns[0]

    returns = prices[col].pct_change().dropna()
    rv = returns.rolling(window=window).std() * np.sqrt(252)
    return rv


def build_regime_signal(
    vix_data: dict,
    prices: pd.DataFrame,
    spx_implied_vol: float | None = None,
) -> dict:
    """
    Build a single regime signal row from live data.

    Args:
        vix_data: Output of IBKRClient.fetch_vix_data()
        prices: Historical price DataFrame (for realized vol computation)
        spx_implied_vol: SPX ATM 3-month implied vol (from vol surface)

    Returns:
        Dict matching vol_regime_signals schema.
    """
    rv = compute_realized_vol(prices, window=20)
    rv_20d = rv.iloc[-1] if len(rv) > 0 else None

    vix = vix_data.get("vix_front_month")
    term_slope = vix_data.get("term_structure_slope", 0.0)
    vvix = vix_data.get("vvix")

    # RV/IV spread: realized_vol - implied_vol (same as synthetic data convention)
    if rv_20d is not None and spx_implied_vol is not None:
        rv_iv_spread = rv_20d - spx_implied_vol
    elif rv_20d is not None and vix is not None:
        rv_iv_spread = rv_20d - (vix / 100.0)
    else:
        rv_iv_spread = 0.0

    regime = classify_regime(
        vix=vix or 18.0,
        term_slope=term_slope,
        vvix=vvix,
        rv_iv_spread=rv_iv_spread,
    )

    signal = {
        "ts": datetime.now(timezone.utc),
        "vix_front_month": vix,
        "vix_second_month": vix_data.get("vix_second_month"),
        "term_structure_slope": term_slope,
        "vvix": vvix,
        "realized_vol_20d": rv_20d,
        "rv_iv_spread": rv_iv_spread,
        "regime": regime,
    }

    logger.info(
        "Regime signal: VIX=%.1f, slope=%.2f%%, VVIX=%s, RV20=%.3f, spread=%.3f → %s",
        vix or 0, term_slope, vvix, rv_20d or 0, rv_iv_spread, regime,
    )

    return signal
