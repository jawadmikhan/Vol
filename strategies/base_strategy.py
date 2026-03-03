"""
Base Strategy — Abstract interface for all sub-strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """Abstract base class that every sub-strategy must implement."""

    def __init__(self, name: str, capital: float):
        self.name = name
        self.capital = capital
        self.positions = []
        self.greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        self.pnl_history = []
        self.notional_deployed = 0.0

    @abstractmethod
    def generate_signals(self, data: dict) -> pd.DataFrame:
        """Generate entry/exit signals from reference data."""
        pass

    @abstractmethod
    def construct_positions(self, signals: pd.DataFrame) -> list:
        """Convert signals into concrete positions with sizing."""
        pass

    @abstractmethod
    def compute_greeks(self) -> dict:
        """Compute aggregate Delta, Gamma, Vega, Theta in dollar terms."""
        pass

    @abstractmethod
    def run_scenario(self, vix_level: float, correlation_shock: float = 0.0) -> float:
        """Return estimated Profit and Loss under a VIX scenario."""
        pass

    def validate_constraints(self, constraints: dict) -> dict:
        """Validate positions against Investment Committee mandate. Returns violations."""
        violations = {}
        if self.notional_deployed > constraints.get("notional_cap", float("inf")):
            violations["notional_breach"] = {
                "deployed": self.notional_deployed,
                "limit": constraints["notional_cap"],
            }
        return violations

    def summary(self) -> dict:
        """Return strategy summary for portfolio aggregation."""
        return {
            "strategy": self.name,
            "capital_allocated": self.capital,
            "notional_deployed": self.notional_deployed,
            "num_positions": len(self.positions),
            "greeks": self.greeks.copy(),
            "pnl_cumulative": sum(self.pnl_history) if self.pnl_history else 0.0,
        }
