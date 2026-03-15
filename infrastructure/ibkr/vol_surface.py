"""
Build an implied volatility surface from raw IBKR option chain data.

Uses SVI (Stochastic Volatility Inspired) parameterization per tenor slice,
with SSVI for arbitrage-free interpolation across the term structure.

Transforms raw option ticks into the same implied_vol_surface.csv format
that the strategies expect:
  name, strike_delta, tenor_months, implied_vol, forward_price, spot_price,
  index_weight, sector
"""

import logging

import numpy as np
import pandas as pd

from models.svi import calibrate_svi, extract_delta_vols, calibrate_ssvi, SVIParams
from .contracts import symbol_sector, symbol_weight, DELTA_LABELS, TENOR_MONTHS

logger = logging.getLogger(__name__)


def build_surface(chain_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Convert a raw option chain DataFrame (from IBKRClient.fetch_option_chain)
    into the implied_vol_surface format the strategies consume.

    Fits SVI to each tenor slice, then extracts standard delta-bucket vols.

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
    svi_slices = []
    spot = chain_df["spot"].dropna().iloc[0] if "spot" in chain_df.columns else None

    for tenor_months in TENOR_MONTHS:
        tenor_data = chain_df[chain_df["tenor_months"] == tenor_months].copy()
        if tenor_data.empty:
            continue

        # Filter to rows with valid IV and strike
        valid = tenor_data[
            tenor_data["implied_vol"].notna()
            & tenor_data["strike"].notna()
            & (tenor_data["implied_vol"] > 0)
        ]

        if len(valid) < 3:
            logger.warning(
                "%s tenor %dM: only %d valid quotes, skipping SVI",
                symbol, tenor_months, len(valid),
            )
            continue

        # Estimate forward price from put-call parity or approximate
        forward = _estimate_forward(valid, spot, tenor_months)

        tenor_years = tenor_months / 12.0

        # Deduplicate strikes: average IVs at the same strike (puts + calls)
        strike_iv = (
            valid.groupby("strike")["implied_vol"]
            .mean()
            .reset_index()
            .sort_values("strike")
        )

        try:
            # Calibrate SVI for this tenor slice
            svi_params = calibrate_svi(
                strikes=strike_iv["strike"].values,
                market_vols=strike_iv["implied_vol"].values,
                forward=forward,
                tenor=tenor_years,
            )
            svi_slices.append(svi_params)

            # Extract standard delta-bucket vols from the SVI fit
            delta_vols = extract_delta_vols(svi_params)

            logger.debug(
                "%s %dM SVI: ATM=%.2f%% 25RR=%.2f%% RMSE=%.4f",
                symbol, tenor_months,
                delta_vols["ATM"] * 100,
                (delta_vols["25D_PUT"] - delta_vols["25D_CALL"]) * 100,
                svi_params.residual,
            )

        except Exception as e:
            logger.warning(
                "%s tenor %dM: SVI calibration failed (%s), falling back to interpolation",
                symbol, tenor_months, e,
            )
            delta_vols = _fallback_interpolation(valid, forward, tenor_years)

        for delta_label, iv in delta_vols.items():
            rows.append({
                "name": symbol,
                "strike_delta": delta_label,
                "tenor_months": tenor_months,
                "implied_vol": iv,
                "forward_price": forward,
                "spot_price": spot,
                "index_weight": symbol_weight(symbol),
                "sector": symbol_sector(symbol),
            })

    # Fit SSVI across tenors if we have enough slices
    if len(svi_slices) >= 2:
        try:
            ssvi = calibrate_ssvi(svi_slices)
            logger.info(
                "%s SSVI fit: rho=%.3f eta=%.3f gamma=%.3f",
                symbol, ssvi.rho, ssvi.eta, ssvi.gamma,
            )
        except Exception as e:
            logger.warning("%s SSVI fit failed: %s", symbol, e)

    return pd.DataFrame(rows)


def _estimate_forward(
    chain_data: pd.DataFrame, spot: float | None, tenor_months: int
) -> float:
    """
    Estimate forward price from put-call parity or risk-free approximation.
    """
    # Try put-call parity: find ATM pair
    if "bid" in chain_data.columns and "ask" in chain_data.columns:
        calls = chain_data[chain_data["right"] == "C"].copy()
        puts = chain_data[chain_data["right"] == "P"].copy()

        if not calls.empty and not puts.empty:
            # Find the strike where call - put is closest to 0 (ATM forward)
            merged = calls.merge(
                puts, on="strike", suffixes=("_c", "_p")
            )
            if not merged.empty:
                mid_c = (merged["bid_c"].fillna(0) + merged["ask_c"].fillna(0)) / 2
                mid_p = (merged["bid_p"].fillna(0) + merged["ask_p"].fillna(0)) / 2
                # F = K + e^rT * (C - P)
                synth_fwd = merged["strike"] + (mid_c - mid_p)
                valid_fwd = synth_fwd[synth_fwd > 0]
                if not valid_fwd.empty:
                    return float(valid_fwd.median())

    # Fallback: risk-free approximation
    if spot and spot > 0:
        r = 0.045  # approximate risk-free rate
        return spot * np.exp(r * tenor_months / 12)

    return spot or 100.0


def _fallback_interpolation(
    chain_data: pd.DataFrame, forward: float, tenor_years: float
) -> dict[str, float]:
    """
    Fallback: simple linear interpolation when SVI fails.
    Returns delta-bucket vols from nearest available data.
    """
    from scipy.interpolate import interp1d

    k = np.log(chain_data["strike"].values / forward)
    iv = chain_data["implied_vol"].values

    # Sort by moneyness
    order = np.argsort(k)
    k_sorted = k[order]
    iv_sorted = iv[order]

    interp = interp1d(k_sorted, iv_sorted, kind="linear", fill_value="extrapolate")

    # Approximate k values for standard deltas
    atm_vol = float(interp(0.0))
    results = {
        "ATM": atm_vol,
        "25D_PUT": float(interp(-0.25)),
        "10D_PUT": float(interp(-0.45)),
        "25D_CALL": float(interp(0.20)),
        "10D_CALL": float(interp(0.40)),
    }
    return results


def build_full_surface(client, symbols: list[str] | None = None) -> pd.DataFrame:
    """
    Fetch option chains and build the complete SVI vol surface for all symbols.

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
        logger.info("Building SVI vol surface for %s", sym)
        chain = client.fetch_option_chain(sym)
        if chain.empty:
            logger.warning("Empty chain for %s, skipping", sym)
            continue
        surface = build_surface(chain, sym)
        surfaces.append(surface)

    if not surfaces:
        return pd.DataFrame()

    full = pd.concat(surfaces, ignore_index=True)
    logger.info("Built SVI vol surface: %d rows across %d symbols", len(full), len(surfaces))
    return full
