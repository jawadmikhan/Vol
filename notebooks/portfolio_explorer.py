"""
Portfolio Explorer
===================
Standalone script to explore the generated reference data
and portfolio outputs. Run after main.py.

Usage:
    python notebooks/portfolio_explorer.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path


def explore():
    data_dir = Path(__file__).parent.parent / "data" / "reference"

    print("=" * 60)
    print("PORTFOLIO DATA EXPLORER")
    print("=" * 60)

    # 1. Implied Vol Surface
    print("\n--- Implied Volatility Surface ---")
    iv = pd.read_csv(data_dir / "implied_vol_surface.csv")
    print(f"  Shape: {iv.shape}")
    print(f"  Names: {iv['name'].nunique()} ({iv['name'].unique()[:5]}...)")
    print(f"  Strikes: {iv['strike_delta'].unique().tolist()}")
    print(f"  Tenors: {iv['tenor_months'].unique().tolist()}")
    print(f"  Implied Vol Range: {iv['implied_vol'].min():.4f} to {iv['implied_vol'].max():.4f}")

    # Index ATM term structure
    idx_atm = iv[(iv["name"] == "SPX_INDEX") & (iv["strike_delta"] == "ATM")]
    print(f"\n  SPX Index At-The-Money Term Structure:")
    for _, row in idx_atm.iterrows():
        print(f"    {row['tenor_months']:>2}M: {row['implied_vol']:.4f}")

    # 2. Realized Vol History
    print("\n--- Realized Volatility History ---")
    rv = pd.read_csv(data_dir / "realized_vol_history.csv")
    print(f"  Shape: {rv.shape}")
    print(f"  Date Range: {rv['date'].iloc[0]} to {rv['date'].iloc[-1]}")
    rv_cols = [c for c in rv.columns if "realized_vol" in c]
    print(f"  Realized Vol Columns: {len(rv_cols)}")

    # 3. Correlation Matrix
    print("\n--- Correlation Matrix ---")
    corr = pd.read_csv(data_dir / "correlation_matrix.csv", index_col=0)
    print(f"  Shape: {corr.shape}")
    upper_tri = corr.values[np.triu_indices(len(corr), k=1)]
    print(f"  Average Pairwise Correlation: {np.mean(upper_tri):.4f}")
    print(f"  Minimum Correlation:          {np.min(upper_tri):.4f}")
    print(f"  Maximum Correlation:          {np.max(upper_tri):.4f}")

    # 4. Regime Signals
    print("\n--- Volatility Regime Signals ---")
    reg = pd.read_csv(data_dir / "vol_regime_signals.csv")
    print(f"  Shape: {reg.shape}")
    print(f"  Date Range: {reg['date'].iloc[0]} to {reg['date'].iloc[-1]}")
    print(f"  VIX Range: {reg['vix_front_month'].min():.1f} to {reg['vix_front_month'].max():.1f}")
    print(f"  Regime Distribution:")
    for regime, count in reg["regime"].value_counts().items():
        print(f"    {regime}: {count} days ({count/len(reg)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Explorer complete. Run main.py for full portfolio construction.")
    print("=" * 60)


if __name__ == "__main__":
    explore()
