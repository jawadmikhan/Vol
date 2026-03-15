"""
IBKR contract definitions for the volatility portfolio universe.

Defines the SPX index, ~45 constituents, VIX futures, and helper
functions for building option contract objects on the fly.
"""

from ib_insync import (
    Index,
    Stock,
    Future,
    Option,
    Contract,
)

# ---------------------------------------------------------------------------
# Universe — SPX index + top constituents by sector
# ---------------------------------------------------------------------------

# Sector-weighted constituent universe (mirrors synthetic_data.py structure)
# These are the actual liquid names you'd trade dispersion on.
UNIVERSE = {
    "SPX_INDEX": {"contract": Index("SPX", "CBOE"), "sector": "Index", "weight": 1.0},
    # Technology (~30%)
    "AAPL": {"sector": "Technology", "weight": 0.07},
    "MSFT": {"sector": "Technology", "weight": 0.065},
    "NVDA": {"sector": "Technology", "weight": 0.06},
    "AMZN": {"sector": "Technology", "weight": 0.035},
    "META": {"sector": "Technology", "weight": 0.025},
    "GOOGL": {"sector": "Technology", "weight": 0.025},
    "AVGO": {"sector": "Technology", "weight": 0.02},
    # Financials (~13%)
    "JPM": {"sector": "Financials", "weight": 0.025},
    "BRK B": {"sector": "Financials", "weight": 0.02},
    "V": {"sector": "Financials", "weight": 0.02},
    "MA": {"sector": "Financials", "weight": 0.018},
    "BAC": {"sector": "Financials", "weight": 0.015},
    "GS": {"sector": "Financials", "weight": 0.012},
    # Healthcare (~13%)
    "UNH": {"sector": "Healthcare", "weight": 0.025},
    "JNJ": {"sector": "Healthcare", "weight": 0.02},
    "LLY": {"sector": "Healthcare", "weight": 0.025},
    "PFE": {"sector": "Healthcare", "weight": 0.015},
    "ABBV": {"sector": "Healthcare", "weight": 0.018},
    "MRK": {"sector": "Healthcare", "weight": 0.017},
    # Consumer (~11%)
    "TSLA": {"sector": "Consumer", "weight": 0.02},
    "WMT": {"sector": "Consumer", "weight": 0.018},
    "PG": {"sector": "Consumer", "weight": 0.017},
    "COST": {"sector": "Consumer", "weight": 0.015},
    "HD": {"sector": "Consumer", "weight": 0.015},
    "KO": {"sector": "Consumer", "weight": 0.015},
    # Industrials (~9%)
    "CAT": {"sector": "Industrials", "weight": 0.02},
    "GE": {"sector": "Industrials", "weight": 0.018},
    "UNP": {"sector": "Industrials", "weight": 0.017},
    "RTX": {"sector": "Industrials", "weight": 0.018},
    "HON": {"sector": "Industrials", "weight": 0.017},
    # Energy (~6%)
    "XOM": {"sector": "Energy", "weight": 0.018},
    "CVX": {"sector": "Energy", "weight": 0.015},
    "COP": {"sector": "Energy", "weight": 0.014},
    "SLB": {"sector": "Energy", "weight": 0.013},
    # Communication (~8%)
    "GOOG": {"sector": "Communication", "weight": 0.025},
    "NFLX": {"sector": "Communication", "weight": 0.018},
    "DIS": {"sector": "Communication", "weight": 0.017},
    "CMCSA": {"sector": "Communication", "weight": 0.02},
    # Materials (~4%)
    "LIN": {"sector": "Materials", "weight": 0.015},
    "APD": {"sector": "Materials", "weight": 0.013},
    "FCX": {"sector": "Materials", "weight": 0.012},
    # Utilities (~3%)
    "NEE": {"sector": "Utilities", "weight": 0.016},
    "DUK": {"sector": "Utilities", "weight": 0.014},
    # Real Estate (~3%)
    "PLD": {"sector": "Real Estate", "weight": 0.016},
    "AMT": {"sector": "Real Estate", "weight": 0.014},
}

# Build stock contracts for constituents
for symbol, meta in UNIVERSE.items():
    if symbol == "SPX_INDEX":
        continue
    meta["contract"] = Stock(symbol, "SMART", "USD")


# ---------------------------------------------------------------------------
# VIX futures contracts (front and second month)
# ---------------------------------------------------------------------------

def vix_future(expiry: str) -> Future:
    """
    Create a VIX future contract.
    expiry format: 'YYYYMMDD' (e.g., '20260415')
    """
    return Future("VIX", expiry, "CFE")


# ---------------------------------------------------------------------------
# Option contract builders
# ---------------------------------------------------------------------------

def spx_option(expiry: str, strike: float, right: str) -> Option:
    """
    SPX index option (European, cash-settled, CBOE).
    right: 'C' or 'P'
    """
    return Option("SPX", expiry, strike, right, "SMART", tradingClass="SPX")


def equity_option(symbol: str, expiry: str, strike: float, right: str) -> Option:
    """
    Single-stock equity option (American, SMART routing).
    """
    return Option(symbol, expiry, strike, right, "SMART")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def constituent_symbols() -> list[str]:
    """Return list of constituent symbols (excluding SPX_INDEX)."""
    return [s for s in UNIVERSE if s != "SPX_INDEX"]


def symbol_sector(symbol: str) -> str:
    """Return the sector for a given symbol."""
    return UNIVERSE.get(symbol, {}).get("sector", "Unknown")


def symbol_weight(symbol: str) -> float:
    """Return the index weight for a given symbol."""
    return UNIVERSE.get(symbol, {}).get("weight", 0.0)
