"""
Data adapter — bridges IBKR live data into the same dict format
that main.py and all strategies expect.

This is the KEY integration layer. It produces a `data` dict identical
in structure to what synthetic_data.py generates, so strategies require
ZERO code changes.

Usage:
    adapter = LiveDataAdapter(ibkr_client)
    data = adapter.fetch_all()
    # data is the same dict you'd pass to strategy.generate_signals(data)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .ibkr.client import IBKRClient
from .ibkr.contracts import constituent_symbols, symbol_sector, symbol_weight, UNIVERSE
from .ibkr.vol_surface import build_full_surface
from .ibkr.regime import build_regime_signal, compute_realized_vol
from .db.connection import (
    insert_dataframe,
    query_df,
    latest_iv_surface,
    latest_realized_vol,
    latest_correlation_matrix,
    latest_regime,
)

logger = logging.getLogger(__name__)

# Path to static specs that don't change
SPECS_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"


class LiveDataAdapter:
    """
    Fetches live market data from IBKR and transforms it into the
    exact dict structure that strategies consume.

    The output dict has these keys (matching synthetic data):
        - implied_vol_surface: pd.DataFrame
        - realized_vol_history: pd.DataFrame
        - correlation_matrix: pd.DataFrame
        - vol_regime_signals: pd.DataFrame
        - portfolio_constraints: dict
        - option_overlay_specs: dict
    """

    def __init__(self, client: IBKRClient, use_cache: bool = True):
        """
        Args:
            client: Connected IBKRClient instance.
            use_cache: If True, check TimescaleDB for recent data before fetching.
        """
        self.client = client
        self.use_cache = use_cache

    def fetch_all(self) -> dict:
        """
        Fetch all data needed by the strategies.
        Returns the canonical data dict.
        """
        logger.info("Fetching all live data for strategy consumption")

        data = {}

        # 1. Historical prices (needed for realized vol + correlations)
        prices = self._fetch_prices()

        # 2. Implied vol surface
        data["implied_vol_surface"] = self._fetch_iv_surface()

        # 3. Realized vol history
        data["realized_vol_history"] = self._build_realized_vol(prices)

        # 4. Correlation matrix
        data["correlation_matrix"] = self._build_correlation_matrix(prices)

        # 5. Vol regime signals
        spx_atm_iv = self._extract_spx_atm_iv(data["implied_vol_surface"])
        data["vol_regime_signals"] = self._fetch_regime_signals(prices, spx_atm_iv)

        # 6. Static config (unchanged from reference data)
        data["portfolio_constraints"] = self._load_json("portfolio_constraints.json")
        data["option_overlay_specs"] = self._load_json("option_overlay_specs.json")

        logger.info("Live data fetch complete — all strategy inputs ready")
        return data

    # ------------------------------------------------------------------
    # Private fetch methods
    # ------------------------------------------------------------------

    def _fetch_prices(self) -> pd.DataFrame:
        """Fetch historical daily prices for the full universe."""
        logger.info("Fetching historical prices (252 days)")
        prices = self.client.fetch_historical_prices(days=300)  # extra buffer

        # Persist to DB
        if not prices.empty:
            self._persist_prices(prices)

        return prices

    def _fetch_iv_surface(self) -> pd.DataFrame:
        """Fetch live implied vol surface from IBKR option chains."""
        # Check cache first
        if self.use_cache:
            cached = latest_iv_surface()
            if not cached.empty:
                age_hours = (
                    datetime.now(timezone.utc) - pd.Timestamp(cached["ts"].max(), tz="UTC")
                ).total_seconds() / 3600
                if age_hours < 1:
                    logger.info("Using cached IV surface (%.1f hours old)", age_hours)
                    return cached.drop(columns=["ts"], errors="ignore")

        logger.info("Building live implied vol surface from IBKR chains")
        surface = build_full_surface(self.client)

        # Persist to DB
        if not surface.empty:
            surface_db = surface.copy()
            surface_db["ts"] = datetime.now(timezone.utc)
            insert_dataframe("implied_vol_surface", surface_db)

        return surface

    def _build_realized_vol(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute realized vol history from daily prices.
        Output matches realized_vol_history.csv format:
            date, {name}_realized_vol, {name}_return for each asset
        """
        if prices.empty:
            return pd.DataFrame()

        returns = prices.pct_change().dropna()
        rv_20d = returns.rolling(window=20).std() * np.sqrt(252)

        # Build wide-format DataFrame matching synthetic structure
        result = pd.DataFrame(index=returns.index)
        result.index.name = "date"

        for col in prices.columns:
            result[f"{col}_return"] = returns[col] if col in returns.columns else np.nan
            result[f"{col}_realized_vol"] = rv_20d[col] if col in rv_20d.columns else np.nan

        # Persist to DB (long format)
        self._persist_realized_vol(returns, rv_20d)

        return result.dropna(how="all")

    def _build_correlation_matrix(
        self, prices: pd.DataFrame, window: int = 90
    ) -> pd.DataFrame:
        """
        Compute rolling 90-day pairwise correlation matrix.
        Returns a square DataFrame matching correlation_matrix.csv.
        """
        if prices.empty or len(prices) < window:
            return pd.DataFrame()

        returns = prices.pct_change().dropna()
        corr = returns.tail(window).corr()

        # Persist to DB
        self._persist_correlation(corr)

        return corr

    def _fetch_regime_signals(
        self, prices: pd.DataFrame, spx_atm_iv: float | None
    ) -> pd.DataFrame:
        """Fetch VIX data and build regime signal."""
        vix_data = self.client.fetch_vix_data()
        signal = build_regime_signal(vix_data, prices, spx_atm_iv)

        # Persist to DB
        signal_db = signal.copy()
        insert_dataframe("vol_regime_signals", pd.DataFrame([signal_db]))

        # Return as DataFrame (strategies expect a DataFrame)
        signal_row = {k: v for k, v in signal.items() if k != "ts"}
        signal_row["date"] = datetime.now().strftime("%Y-%m-%d")
        return pd.DataFrame([signal_row])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_spx_atm_iv(self, surface: pd.DataFrame) -> float | None:
        """Pull SPX ATM 3-month IV from the surface."""
        if surface.empty:
            return None
        mask = (
            (surface["name"] == "SPX_INDEX")
            & (surface["strike_delta"] == "ATM")
            & (surface["tenor_months"] == 3)
        )
        match = surface[mask]
        if match.empty:
            return None
        return match["implied_vol"].iloc[0]

    def _load_json(self, filename: str) -> dict:
        """Load a static JSON config from the reference data directory."""
        path = SPECS_DIR / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        logger.warning("Config file not found: %s", path)
        return {}

    def _persist_prices(self, prices: pd.DataFrame):
        """Store price data in realized_vol_history table (long format)."""
        returns = prices.pct_change().dropna()
        rv = returns.rolling(20).std() * np.sqrt(252)

        rows = []
        for date in returns.index:
            for sym in returns.columns:
                rows.append({
                    "ts": pd.Timestamp(date, tz="UTC"),
                    "name": sym,
                    "daily_return": returns.loc[date, sym] if not np.isnan(returns.loc[date, sym]) else None,
                    "realized_vol": rv.loc[date, sym] if date in rv.index and not np.isnan(rv.loc[date, sym]) else None,
                })

        if rows:
            insert_dataframe("realized_vol_history", pd.DataFrame(rows))

    def _persist_realized_vol(self, returns: pd.DataFrame, rv: pd.DataFrame):
        """Store realized vol in long format to DB."""
        rows = []
        latest_date = returns.index[-1]
        for sym in returns.columns:
            rows.append({
                "ts": pd.Timestamp(latest_date, tz="UTC"),
                "name": sym,
                "daily_return": returns.loc[latest_date, sym],
                "realized_vol": rv.loc[latest_date, sym] if latest_date in rv.index else None,
            })
        if rows:
            insert_dataframe("realized_vol_history", pd.DataFrame(rows))

    def _persist_correlation(self, corr: pd.DataFrame):
        """Store correlation matrix in long format to DB."""
        now = datetime.now(timezone.utc)
        rows = []
        for a1 in corr.index:
            for a2 in corr.columns:
                rows.append({
                    "ts": now,
                    "asset_1": a1,
                    "asset_2": a2,
                    "correlation": corr.loc[a1, a2],
                })
        if rows:
            insert_dataframe("correlation_snapshots", pd.DataFrame(rows))
