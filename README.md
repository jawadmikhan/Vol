# Systematic Volatility Portfolio Process

Interactive workflow dashboard for a five-leg systematic volatility portfolio — from signal construction through execution, monitoring, and rebalancing.

**Live site:** [jawadmikhan.github.io/Vol](https://jawadmikhan.github.io/Vol)

| Page | Description | Link |
|------|-------------|------|
| Interactive Dashboard | 48-step workflow with Mermaid flowcharts, stage tables, task streams, blocker analysis | [index.html](https://jawadmikhan.github.io/Vol) |
| Workflow Map | Full-page Mermaid process map — all 7 stages, 5 gates, 3 feedback loops | [workflow.html](https://jawadmikhan.github.io/Vol/workflow.html) |
| Mock Portfolio | $250M portfolio implementation — 5 sub-strategies, Greeks, scenarios, 60-day timeline | [portfolio.html](https://jawadmikhan.github.io/Vol/portfolio.html) |

---

## Overview

A 48-step, 7-stage portfolio management workflow spanning five strategy legs:

| Leg | Strategy | Description |
|-----|----------|-------------|
| 1 | Dispersion (Core) | Index vs constituent volatility mispricing |
| 2 | Volatility Harvesting | Systematic short volatility premium capture |
| 3 | Directional Long/Short | Macro-regime driven volatility views |
| 4 | Dynamic Volatility Targeting | Inverse-volatility leverage scaling |
| 5 | Option Overlays | Protective puts, collars, tail-risk hedges |

## Mock Portfolio ($250M)

The `portfolio.html` page documents a complete mock portfolio implementation with:

| Parameter | Value |
|-----------|-------|
| Total Capital | $250,000,000 |
| Gross Notional Cap | $500,000,000 |
| Net Vega Bounds | −$2M to +$4M per VIX point |
| Volatility Target | 12% annualized |
| Short Variance Notional Cap | $80,000,000 |
| Option Overlay Drag Ceiling | 75 basis points |
| Implementation Horizon | 60 days (5 phases) |

### Capital Allocation

| Strategy | Weight | Capital |
|----------|--------|---------|
| Dispersion Trading | 40% | $100,000,000 |
| Volatility Harvesting | 20% | $50,000,000 |
| Directional Long/Short | 15% | $37,500,000 |
| Dynamic Volatility Targeting | 15% | $37,500,000 |
| Option Overlays | 10% | $25,000,000 |

## Python Codebase

The repository includes a full Python implementation that generates synthetic reference data, constructs all five strategies, aggregates portfolio-level Greeks, runs scenario analysis, and produces attribution reports.

### Quick Start

```bash
git clone https://github.com/jawadmikhan/Vol.git
cd Vol

pip install -r requirements.txt

# Generate synthetic reference data
python -m data.generators.synthetic_data

# Run the full portfolio (all 7 stages)
python main.py

# Run constraint validation tests
python tests/test_constraints.py
```

### Repository Structure

```
Vol/
├── index.html                          # Interactive dashboard (live site)
├── workflow.html                       # Full-page Mermaid workflow map
├── portfolio.html                      # Mock portfolio page
├── README.md
├── requirements.txt                    # numpy, pandas, scipy, matplotlib, seaborn
├── main.py                             # 7-stage orchestrator
├── config/
│   └── portfolio_constraints.py        # Investment Committee mandate parameters
├── data/
│   ├── generators/
│   │   └── synthetic_data.py           # Generates all 6 reference files
│   └── reference/
│       ├── implied_vol_surface.csv     # 5 strikes × 6 tenors × 46 names
│       ├── realized_vol_history.csv    # 5-year daily GARCH-filtered realized vol
│       ├── correlation_matrix.csv      # 46×46 pairwise 90-day correlations
│       ├── vol_regime_signals.csv      # 24-month VIX term structure + regime
│       ├── portfolio_constraints.json  # Investment Committee mandate (JSON)
│       └── option_overlay_specs.json   # Put/collar program specs
├── strategies/
│   ├── base_strategy.py                # Abstract base class
│   ├── dispersion.py                   # Implied correlation decomposition
│   ├── volatility_harvesting.py        # Short variance with convexity adjustment
│   ├── directional_long_short.py       # 3-regime classifier
│   ├── dynamic_vol_targeting.py        # Inverse-vol leverage scaling
│   └── option_overlay.py              # 3-tranche puts + collar financing
├── risk/
│   ├── greeks_engine.py                # Portfolio-level Greeks aggregation
│   ├── scenario_analysis.py            # VIX 45 spike / VIX 11 collapse
│   └── attribution.py                  # Daily Profit & Loss decomposition
├── tests/
│   └── test_constraints.py             # 6 constraint validation tests
└── notebooks/
    └── portfolio_explorer.py           # Reference data inspection
```

### Workflow Stages

| Stage | Focus | Steps |
|-------|-------|-------|
| 1. Idea Generation & Sourcing | Signal construction | 7 |
| 2. Research Aggregation | Model calibration | 6 |
| 3. Collaboration & Peer Review | Stress testing & review | 5 |
| 4. Thesis Development | Portfolio design | 11 |
| 5. Decision, Approval, & Portfolio Construction | Investment Committee vote & execution | 7 |
| 6. Monitoring & Review | Greeks & attribution tracking | 6 |
| 7. Rebalancing & Liquidity | Option rolls & reporting | 6 |

### Blocker Distribution

| Type | Count | Percentage |
|------|-------|------------|
| No Blocker (Green) | 34 | 71% |
| Soft Blocker (Yellow) | 10 | 21% |
| Hard Blocker (Red) | 4 | 8% |

### Reference Data

All synthetic — no proprietary or live market data. Formatted to match OptionMetrics and FactSet conventions.

| File | Description |
|------|-------------|
| `implied_vol_surface.csv` | 1,380 rows — 5 strikes × 6 tenors × 46 names |
| `realized_vol_history.csv` | 1,260 days × 46 names — GARCH(1,1) simulated |
| `correlation_matrix.csv` | 46×46 pairwise 90-day realized correlations |
| `vol_regime_signals.csv` | 504 days — VIX term structure, VVIX, regime classification |
| `portfolio_constraints.json` | Investment Committee mandate in JSON format |
| `option_overlay_specs.json` | Put/collar program specifications |

---

Systematic Volatility Portfolio Process · Cross-Asset Trading Strategist / Systematic Quant / Macro Trader
