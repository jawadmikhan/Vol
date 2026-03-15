"""
Build an implied volatility surface from raw IBKR option chain data.

Transforms raw option ticks into the same implied_vol_surface.csv format
that the strategies expect:
  name, strike_delta, tenor_months, implied_vol, forward_price, spot_price,
  index_weight, sector
"""

import logging

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from .contracts import symbol_sector, symbol_weight, DELTA_LABELS, TENOR_MONTHS

logger = logging.getLogger(__name__)

# Target delta values for interpolation
TARGET_DELTAS = {
    "10D_PUT": -0.10,
    "25D_PUT": -0.25,
    "ATM": 0.50,     # Call delta ~ 0.50 for ATM
    "25D_CALL": 0.25,
    "10D_CALL": 0.10,
}


def build_surface(chain_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Convert a raw option chain DataFrame (from IBKRClient.fetch_option_chain)
    into the implied_vol_surface format the strategies consume.

    Args:
        chain_df: Raw chain with columns [underlying, expiry, strike, right,
                  implied_vol, delta, spot, tenor_months, ...]
        symbol: The symbol name for the output (e.g. 'AAPL' or 'SPX_INDEX')

    Returns:
        DataFrame matching implied_vol_surface schema.
    """
    if chain_df.empty:
        return pd.DataFrame()

    rows = []
    spot = chain_df["spot"].dropna().iloc[0] if "spot" in chain_df.columns else None

    for tenor in TENOR_MONTHS:
        tenor_data = chain_df[chain_df["tenor_months"] == tenor].copy()
        if tenor_data.empty:
            continue

        # Separate puts and calls, drop rows without IV or delta
        puts = tenor_data[
            (tenor_data["right"] == "P")
            & tenor_data["implied_vol"].notna()
            & tenor_data["delta"].notna()
        ].copy()
        calls = tenor_data[
            (tenor_data["right"] == "C")
            & tenor_data["implied_vol"].notna()
            & tenor_data["delta"].notna()
        ].copy()

        # Build delta-to-IV mapping using interpolation
        iv_by_delta = {}

        # Put side: 10D_PUT and 25D_PUT
        if len(puts) >= 3:
            puts_sorted = puts.sort_values("delta")
            try:
                cs_put = CubicSpline(
                    puts_sorted["delta"].values,
                    puts_sorted["implied_vol"].values,
                    extrapolate=True,
                )
                iv_by_delta["10D_PUT"] = float(cs_put(TARGET_DELTAS["10D_PUT"]))
                iv_by_delta["25D_PUT"] = float(cs_put(TARGET_DELTAS["25D_PUT"]))
            except Exception:
                # Fallback: nearest neighbor
                for label, target in [("10D_PUT", -0.10), ("25D_PUT", -0.25)]:
                    idx = (puts_sorted["delta"] - target).abs().idxmin()
                    iv_by_delta[label] = puts_sorted.loc[idx, "implied_vol"]

        # Call side: ATM, 25D_CALL, 10D_CALL
        if len(calls) >= 3:
            calls_sorted = calls.sort_values("delta", ascending=False)
            try:
                cs_call = CubicSpline(
                    calls_sorted["delta"].values,
                    calls_sorted["implied_vol"].values,
                    extrapolate=True,
                )
                iv_by_delta["ATM"] = float(cs_call(TARGET_DELTAS["ATM"]))
                iv_by_delta["25D_CALL"] = float(cs_call(TARGET_DELTAS["25D_CALL"]))
                iv_by_delta["10D_CALL"] = float(cs_call(TARGET_DELTAS["10D_CALL"]))
            except Exception:
                for label, target in [("ATM", 0.50), ("25D_CALL", 0.25), ("10D_CALL", 0.10)]:
                    idx = (calls_sorted["delta"] - target).abs().idxmin()
                    iv_by_delta[label] = calls_sorted.loc[idx, "implied_vol"]

        # If we don't have ATM from calls, try from puts (delta ~ -0.50)
        if "ATM" not in iv_by_delta and len(puts) >= 2:
            idx = (puts["delta"].abs() - 0.50).abs().idxmin()
            iv_by_delta["ATM"] = puts.loc[idx, "implied_vol"]

        # Estimate forward price: F = S * exp(r*T) ≈ S for short tenors
        # More accurate: use put-call parity if we have ATM quotes
        forward = spot * (1 + 0.05 * tenor / 12) if spot else None  # rough risk-free

        for delta_label, iv in iv_by_delta.items():
            rows.append({
                "name": symbol,
                "strike_delta": delta_label,
                "tenor_months": tenor,
                "implied_vol": iv,
                "forward_price": forward,
                "spot_price": spot,
                "index_weight": symbol_weight(symbol),
                "sector": symbol_sector(symbol),
            })

    return pd.DataFrame(rows)


def build_full_surface(client, symbols: list[str] | None = None) -> pd.DataFrame:
    """
    Fetch option chains and build the complete vol surface for all symbols.

    Args:
        client: An IBKRClient instance (connected).
        symbols: List of symbols to process. Defaults to full universe.

    Returns:
        DataFrame in implied_vol_surface format for all symbols.
    """
    from .contracts import constituent_symbols

    if symbols is None:
        symbols = ["SPX_INDEX"] + constituent_symbols()

    surfaces = []
    for sym in symbols:
        logger.info("Building vol surface for %s", sym)
        chain = client.fetch_option_chain(sym)
        if chain.empty:
            logger.warning("Empty chain for %s, skipping", sym)
            continue
        surface = build_surface(chain, sym)
        surfaces.append(surface)

    if not surfaces:
        return pd.DataFrame()

    full = pd.concat(surfaces, ignore_index=True)
    logger.info("Built vol surface: %d rows across %d symbols", len(full), len(surfaces))
    return full
