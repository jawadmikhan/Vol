"""
Synthetic Reference Data Generator
===================================
Generates all 6 reference data files specified in the Delivery Plan:
  1. implied_vol_surface.csv   — 5 strikes x 6 tenors x 45 names (OptionMetrics-style)
  2. realized_vol_history.csv  — 5-year daily realized vol, GARCH-filtered, index + 45 names
  3. correlation_matrix.csv    — 90-day pairwise realized correlations, 46 x 46
  4. vol_regime_signals.csv    — 24-month daily VIX term structure, VVIX, realized-vs-implied spread
  5. portfolio_constraints.json
  6. option_overlay_specs.json

All data is fabricated — no proprietary or live market data used.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Reproducibility
np.random.seed(42)

# Output directory
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "reference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Universe Definition — 45 constituent names (anonymized sector-based tickers)
# ============================================================================

SECTORS = {
    "Technology":    ["TECH_A", "TECH_B", "TECH_C", "TECH_D", "TECH_E", "TECH_F", "TECH_G"],
    "Financials":    ["FIN_A",  "FIN_B",  "FIN_C",  "FIN_D",  "FIN_E",  "FIN_F"],
    "Healthcare":    ["HLTH_A", "HLTH_B", "HLTH_C", "HLTH_D", "HLTH_E", "HLTH_F"],
    "Consumer":      ["CONS_A", "CONS_B", "CONS_C", "CONS_D", "CONS_E", "CONS_F"],
    "Industrials":   ["IND_A",  "IND_B",  "IND_C",  "IND_D",  "IND_E"],
    "Energy":        ["ENRG_A", "ENRG_B", "ENRG_C", "ENRG_D"],
    "Communication": ["COMM_A", "COMM_B", "COMM_C", "COMM_D"],
    "Materials":     ["MAT_A",  "MAT_B",  "MAT_C"],
    "Utilities":     ["UTIL_A", "UTIL_B"],
    "RealEstate":    ["REIT_A", "REIT_B"],
}

ALL_CONSTITUENTS = []
CONSTITUENT_SECTORS = {}
INDEX_WEIGHTS = {}

for sector, names in SECTORS.items():
    for name in names:
        ALL_CONSTITUENTS.append(name)
        CONSTITUENT_SECTORS[name] = sector

assert len(ALL_CONSTITUENTS) == 45, f"Expected 45 constituents, got {len(ALL_CONSTITUENTS)}"

# Assign realistic index weights (market-cap weighted, Technology heavy)
_sector_weight_map = {
    "Technology": 0.30, "Financials": 0.13, "Healthcare": 0.13, "Consumer": 0.11,
    "Industrials": 0.09, "Energy": 0.06, "Communication": 0.08, "Materials": 0.04,
    "Utilities": 0.03, "RealEstate": 0.03,
}

for sector, names in SECTORS.items():
    sector_total = _sector_weight_map[sector]
    n = len(names)
    # Distribute within sector with slight dispersion
    base = sector_total / n
    jitter = np.random.dirichlet(np.ones(n) * 5) * sector_total
    for i, name in enumerate(names):
        INDEX_WEIGHTS[name] = round(jitter[i], 6)

# Normalize to sum to 1
_total = sum(INDEX_WEIGHTS.values())
INDEX_WEIGHTS = {k: round(v / _total, 6) for k, v in INDEX_WEIGHTS.items()}

INDEX_NAME = "SPX_INDEX"


def generate_implied_vol_surface():
    """
    Generate implied_vol_surface.csv
    Structure: name, strike_delta, tenor_months, implied_vol, forward_price, spot_price
    5 strikes (10-delta put, 25-delta put, at-the-money, 25-delta call, 10-delta call)
    6 tenors (1, 2, 3, 6, 9, 12 months)
    """
    strikes = ["10D_PUT", "25D_PUT", "ATM", "25D_CALL", "10D_CALL"]
    tenors = [1, 2, 3, 6, 9, 12]

    # Base at-the-money implied vol per name (realistic ranges)
    base_atm_vol = {}
    for name in ALL_CONSTITUENTS:
        sector = CONSTITUENT_SECTORS[name]
        sector_base = {
            "Technology": 0.28, "Financials": 0.22, "Healthcare": 0.30,
            "Consumer": 0.20, "Industrials": 0.24, "Energy": 0.35,
            "Communication": 0.26, "Materials": 0.27, "Utilities": 0.16,
            "RealEstate": 0.21,
        }
        base_atm_vol[name] = sector_base[sector] + np.random.normal(0, 0.03)

    # Index vol is lower than constituent average (diversification benefit)
    base_atm_vol[INDEX_NAME] = 0.18 + np.random.normal(0, 0.01)

    rows = []
    for name in [INDEX_NAME] + ALL_CONSTITUENTS:
        atm = base_atm_vol[name]
        spot = 100.0 if name != INDEX_NAME else 4500.0

        for tenor in tenors:
            # Term structure: slight contango in normal regime
            tenor_adj = 1.0 + 0.005 * (tenor - 1)

            for strike in strikes:
                # Skew: puts trade at higher vol, calls at lower vol
                skew_map = {
                    "10D_PUT": 0.08, "25D_PUT": 0.04, "ATM": 0.0,
                    "25D_CALL": -0.02, "10D_CALL": -0.04,
                }
                skew = skew_map[strike]

                # Constituent skew is steeper than index skew
                if name != INDEX_NAME:
                    skew *= 1.3

                iv = max(0.05, atm * tenor_adj + skew + np.random.normal(0, 0.005))

                # Forward price (simplified: no dividend adjustment)
                risk_free = 0.045
                fwd = spot * np.exp(risk_free * tenor / 12)

                rows.append({
                    "name": name,
                    "strike_delta": strike,
                    "tenor_months": tenor,
                    "implied_vol": round(iv, 6),
                    "forward_price": round(fwd, 2),
                    "spot_price": round(spot, 2),
                    "index_weight": INDEX_WEIGHTS.get(name, 1.0),
                    "sector": CONSTITUENT_SECTORS.get(name, "Index"),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "implied_vol_surface.csv", index=False)
    print(f"  Generated implied_vol_surface.csv: {len(df)} rows "
          f"({len(df['name'].unique())} names x {len(strikes)} strikes x {len(tenors)} tenors)")
    return df


def generate_realized_vol_history():
    """
    Generate realized_vol_history.csv
    5-year daily realized volatility for index + 45 constituents.
    Uses a simplified GARCH(1,1) simulation for each name.
    """
    n_days = 252 * 5  # 5 years of trading days
    dates = pd.bdate_range(end="2025-12-31", periods=n_days)

    result = {"date": dates}

    # GARCH(1,1) parameters per name
    for name in [INDEX_NAME] + ALL_CONSTITUENTS:
        if name == INDEX_NAME:
            omega, alpha, beta = 0.000002, 0.08, 0.90
            long_run_vol = 0.18
        else:
            sector = CONSTITUENT_SECTORS[name]
            sector_vol = {
                "Technology": 0.28, "Financials": 0.22, "Healthcare": 0.30,
                "Consumer": 0.20, "Industrials": 0.24, "Energy": 0.35,
                "Communication": 0.26, "Materials": 0.27, "Utilities": 0.16,
                "RealEstate": 0.21,
            }
            long_run_vol = sector_vol[sector] + np.random.normal(0, 0.02)
            omega = long_run_vol**2 * (1 - 0.08 - 0.90) / 252
            alpha = 0.08 + np.random.uniform(-0.02, 0.02)
            beta = 0.90 + np.random.uniform(-0.02, 0.02)

        # Simulate GARCH(1,1) variance path
        sigma2 = np.zeros(n_days)
        returns = np.zeros(n_days)
        sigma2[0] = (long_run_vol / np.sqrt(252)) ** 2

        for t in range(1, n_days):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            sigma2[t] = max(sigma2[t], 1e-8)
            returns[t] = np.random.normal(0, np.sqrt(sigma2[t]))

        # Annualized realized vol (20-day rolling)
        daily_vol = np.sqrt(sigma2)
        rolling_rv = pd.Series(daily_vol).rolling(20).std() * np.sqrt(252)
        result[f"{name}_realized_vol"] = rolling_rv.values
        result[f"{name}_return"] = returns

    df = pd.DataFrame(result)
    df.to_csv(OUTPUT_DIR / "realized_vol_history.csv", index=False)
    print(f"  Generated realized_vol_history.csv: {len(df)} days x "
          f"{len([INDEX_NAME] + ALL_CONSTITUENTS)} names")
    return df


def generate_correlation_matrix():
    """
    Generate correlation_matrix.csv
    90-day pairwise realized correlations for 45 constituents + index.
    Structured as a realistic block-diagonal correlation matrix with sector clustering.
    """
    names = [INDEX_NAME] + ALL_CONSTITUENTS
    n = len(names)

    # Build block-diagonal structure
    corr = np.eye(n)

    # Assign sector indices
    name_to_idx = {name: i for i, name in enumerate(names)}

    # Base correlations
    for i in range(n):
        for j in range(i + 1, n):
            name_i, name_j = names[i], names[j]
            sector_i = CONSTITUENT_SECTORS.get(name_i, "Index")
            sector_j = CONSTITUENT_SECTORS.get(name_j, "Index")

            if name_i == INDEX_NAME or name_j == INDEX_NAME:
                # Constituent-to-index correlation: 0.55–0.85
                weight = INDEX_WEIGHTS.get(
                    name_j if name_i == INDEX_NAME else name_i, 0.02
                )
                base_corr = 0.55 + 0.30 * min(weight / 0.05, 1.0)
            elif sector_i == sector_j:
                # Intra-sector correlation: 0.45–0.75
                base_corr = 0.60 + np.random.normal(0, 0.08)
            else:
                # Inter-sector correlation: 0.20–0.50
                base_corr = 0.35 + np.random.normal(0, 0.08)

            base_corr = np.clip(base_corr, 0.10, 0.95)
            corr[i, j] = base_corr
            corr[j, i] = base_corr

    # Ensure positive semi-definiteness via nearest correlation matrix
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Normalize diagonal back to 1
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)

    df = pd.DataFrame(corr, index=names, columns=names)
    df.index.name = "name"
    df.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    print(f"  Generated correlation_matrix.csv: {n}x{n} matrix")

    # Also compute and save implied correlation
    weights = np.array([INDEX_WEIGHTS.get(name, 0.0) for name in ALL_CONSTITUENTS])
    weights = weights / weights.sum()
    constituent_corr = corr[1:, 1:]  # Exclude index
    implied_corr = float(weights @ constituent_corr @ weights)
    print(f"  Implied correlation (weighted average): {implied_corr:.4f}")

    return df


def generate_vol_regime_signals():
    """
    Generate vol_regime_signals.csv
    24-month daily signals: VIX level, VIX term structure (front/second month),
    VVIX, realized-vs-implied spread, and regime classification.
    """
    n_days = 252 * 2  # 24 months
    dates = pd.bdate_range(end="2025-12-31", periods=n_days)

    # Simulate VIX path using mean-reverting process
    vix = np.zeros(n_days)
    vix[0] = 18.0
    kappa = 0.05       # Mean reversion speed
    theta = 18.0       # Long-run mean
    sigma_vix = 4.0    # VIX volatility

    for t in range(1, n_days):
        dt = 1 / 252
        dW = np.random.normal(0, np.sqrt(dt))
        vix[t] = vix[t-1] + kappa * (theta - vix[t-1]) * dt + sigma_vix * dW
        vix[t] = max(vix[t], 9.0)  # Floor

        # Inject a vol spike regime around month 14-16
        if 290 < t < 340:
            vix[t] += np.random.uniform(5, 12)

    # VIX term structure (second month)
    vix_2m = vix * (1.0 + np.random.normal(0.03, 0.02, n_days))  # Normally contango
    # In spike regime, flip to backwardation
    for t in range(n_days):
        if vix[t] > 25:
            vix_2m[t] = vix[t] * (1.0 - np.random.uniform(0.02, 0.08))

    term_structure_slope = (vix_2m - vix) / vix  # Percentage slope

    # VVIX (vol of VIX) — correlated with VIX level
    vvix = 80 + 1.5 * (vix - 18) + np.random.normal(0, 5, n_days)
    vvix = np.clip(vvix, 60, 180)

    # Realized vol (20-day rolling of simulated SPX returns)
    spx_returns = np.random.normal(0.0003, 0.01, n_days)
    # Increase vol during spike
    for t in range(290, 340):
        spx_returns[t] = np.random.normal(-0.005, 0.025)

    realized_vol_20d = pd.Series(spx_returns).rolling(20).std() * np.sqrt(252) * 100

    # Realized-vs-implied spread
    rv_iv_spread = realized_vol_20d.values - vix

    # Regime classification
    regime = []
    for t in range(n_days):
        v = vix[t]
        slope = term_structure_slope[t]
        if v < 16 and slope > 0.02:
            regime.append("LOW_VOL_HARVESTING")
        elif v >= 25 or slope < -0.02:
            regime.append("CRISIS")
        else:
            regime.append("TRANSITIONAL")

    df = pd.DataFrame({
        "date": dates,
        "vix_front_month": np.round(vix, 2),
        "vix_second_month": np.round(vix_2m, 2),
        "term_structure_slope": np.round(term_structure_slope, 4),
        "vvix": np.round(vvix, 2),
        "realized_vol_20d": np.round(realized_vol_20d.values, 2),
        "rv_iv_spread": np.round(rv_iv_spread, 2),
        "regime": regime,
    })

    df.to_csv(OUTPUT_DIR / "vol_regime_signals.csv", index=False)

    regime_counts = df["regime"].value_counts()
    print(f"  Generated vol_regime_signals.csv: {len(df)} days")
    for r, c in regime_counts.items():
        print(f"    {r}: {c} days ({c/len(df)*100:.1f}%)")

    return df


def generate_portfolio_constraints():
    """
    Generate portfolio_constraints.json — Investment Committee mandate document.
    """
    constraints = {
        "document_title": "Investment Committee Portfolio Mandate — Systematic Volatility Book",
        "effective_date": "2025-01-01",
        "review_cycle": "12-month annual allocation",
        "total_capital_usd": 250_000_000,
        "gross_notional_cap_usd": 500_000_000,
        "net_vega_bounds": {
            "floor_usd_per_vix_point": -2_000_000,
            "ceiling_usd_per_vix_point": 4_000_000,
        },
        "strategy_limits": {
            "dispersion": {
                "max_capital_pct": 0.40,
                "max_capital_usd": 100_000_000,
                "correlation_regime_assessment_required": True,
            },
            "volatility_harvesting": {
                "short_variance_notional_cap_usd": 80_000_000,
                "realized_vs_implied_edge_documentation_required": True,
                "min_edge_basis_points": 200,
            },
            "dynamic_volatility_targeting": {
                "annualized_vol_target": 0.12,
                "rebalance_trigger_deviation_pp": 0.02,
                "lookback_window_days": 20,
            },
            "option_overlay": {
                "max_annual_drag_basis_points": 75,
                "put_tenor_months": 3,
                "put_otm_percent": 0.05,
                "financing_call_delta": 0.10,
                "drawdown_survival_pct": 0.30,
            },
        },
        "risk_limits": {
            "max_portfolio_drawdown": 0.15,
            "total_margin_cap_usd": 180_000_000,
            "stop_loss_long_multiplier": 1.5,
            "stop_loss_short_multiplier": 2.0,
        },
        "implementation": {
            "build_up_days": 60,
            "settlement_currency": "USD",
            "position_types": ["Over-the-Counter", "Listed"],
        },
    }

    with open(OUTPUT_DIR / "portfolio_constraints.json", "w") as f:
        json.dump(constraints, f, indent=2)
    print(f"  Generated portfolio_constraints.json")
    return constraints


def generate_option_overlay_specs():
    """
    Generate option_overlay_specs.json — Protective put and collar program specifications.
    """
    specs = {
        "document_title": "Option Overlay Program Specifications",
        "protective_put_program": {
            "structure": "Rolling 3-month 5% out-of-the-money puts on SPX",
            "tenor_months": 3,
            "moneyness": "5% out-of-the-money",
            "roll_schedule": "Monthly roll, staggered across 3 tranches",
            "notional_per_tranche_usd": 83_333_333,
            "estimated_annual_cost_bps": 120,
            "delta_at_inception": -0.25,
        },
        "collar_financing": {
            "structure": "Sell 10-delta calls against equity exposure to finance puts",
            "call_delta": 0.10,
            "call_tenor_months": 3,
            "estimated_annual_premium_received_bps": 55,
            "upside_cap_pct": "Approximately 8-12% above spot depending on volatility",
        },
        "net_premium_budget": {
            "put_cost_bps": 120,
            "call_premium_bps": 55,
            "net_cost_bps": 65,
            "budget_ceiling_bps": 75,
            "within_budget": True,
        },
        "tail_risk_hedge": {
            "structure": "Far out-of-the-money put spread (15-25% out-of-the-money)",
            "notional_coverage": "100% of portfolio notional",
            "drawdown_scenario_survival": "30% index decline over 30 days",
            "estimated_payout_at_30pct_drawdown_usd": 45_000_000,
            "annual_cost_bps": 10,
        },
        "convexity_profile": {
            "at_minus_5pct": {"portfolio_delta_change": 0.15, "gamma_pickup_usd": 800_000},
            "at_minus_10pct": {"portfolio_delta_change": 0.35, "gamma_pickup_usd": 2_200_000},
            "at_minus_20pct": {"portfolio_delta_change": 0.65, "gamma_pickup_usd": 5_500_000},
            "at_minus_30pct": {"portfolio_delta_change": 0.85, "gamma_pickup_usd": 9_000_000},
        },
    }

    with open(OUTPUT_DIR / "option_overlay_specs.json", "w") as f:
        json.dump(specs, f, indent=2)
    print(f"  Generated option_overlay_specs.json")
    return specs


def generate_all():
    """Generate all reference data files."""
    print("=" * 70)
    print("Generating Synthetic Reference Data")
    print("=" * 70)
    print()

    print("[1/6] Implied Volatility Surface...")
    generate_implied_vol_surface()
    print()

    print("[2/6] Realized Volatility History (5 years, GARCH-filtered)...")
    generate_realized_vol_history()
    print()

    print("[3/6] Correlation Matrix (90-day pairwise, 46x46)...")
    generate_correlation_matrix()
    print()

    print("[4/6] Volatility Regime Signals (24 months)...")
    generate_vol_regime_signals()
    print()

    print("[5/6] Portfolio Constraints (Investment Committee mandate)...")
    generate_portfolio_constraints()
    print()

    print("[6/6] Option Overlay Specifications...")
    generate_option_overlay_specs()
    print()

    print("=" * 70)
    print(f"All reference files saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    generate_all()
