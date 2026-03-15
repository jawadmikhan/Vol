# Systematic Volatility Portfolio

Interactive workflow dashboard and backtesting framework for a five-leg systematic volatility portfolio — from signal construction through execution, monitoring, rebalancing, and performance analysis.

**Live site:** [jawadmikhan.github.io/Vol](https://jawadmikhan.github.io/Vol)

| Page | Description | Link |
|------|-------------|------|
| Interactive Dashboard | 48-step workflow with Mermaid flowcharts, stage tables, task streams, blocker analysis | [index.html](https://jawadmikhan.github.io/Vol) |
| Workflow Map | Full-page Mermaid process map — all 7 stages, 5 gates, 3 feedback loops | [workflow.html](https://jawadmikhan.github.io/Vol/workflow.html) |
| Mock Portfolio | $250M portfolio implementation — 5 sub-strategies, Greeks, scenarios, 60-day timeline | [portfolio.html](https://jawadmikhan.github.io/Vol/portfolio.html) |
| Backtest Results | Walk-forward backtest — PnL attribution, regime analysis, strategy contribution | [backtest.html](https://jawadmikhan.github.io/Vol/backtest.html) |

---

## Overview

A 48-step, 7-stage portfolio management workflow spanning five strategy legs:

| Leg | Strategy | Weight | Capital | Description |
|-----|----------|--------|---------|-------------|
| 1 | Dispersion Trading | 40% | $100M | Index vs constituent volatility mispricing |
| 2 | Volatility Harvesting | 20% | $50M | Systematic short volatility premium capture |
| 3 | Directional Long/Short | 15% | $37.5M | Macro-regime driven volatility views |
| 4 | Dynamic Volatility Targeting | 15% | $37.5M | Inverse-volatility leverage scaling |
| 5 | Option Overlays | 10% | $25M | Protective puts, collars, tail-risk hedges |

## Mock Portfolio ($250M)

| Parameter | Value |
|-----------|-------|
| Total Capital | $250,000,000 |
| Gross Notional Cap | $500,000,000 |
| Net Vega Bounds | -$2M to +$4M per VIX point |
| Volatility Target | 12% annualized |
| Short Variance Notional Cap | $80,000,000 |
| Option Overlay Drag Ceiling | 75 basis points |
| Implementation Horizon | 60 days (5 phases) |

## Backtest Results

Walk-forward simulation over 504 trading days of synthetic regime data with weekly rebalancing.

| Metric | Value |
|--------|-------|
| Total PnL | $35,518,370 |
| Total Return | 14.2% |
| Annualized Return | 7.1% |
| Sharpe Ratio | 4.20 |
| Sortino Ratio | 9.94 |
| Max Drawdown | 0.99% |
| Calmar Ratio | 7.15 |
| Profit Factor | 3.55 |
| Win Rate | 50.4% |

### Performance by Regime

| Regime | Days | Total PnL | Avg Daily | Sharpe |
|--------|------|-----------|-----------|--------|
| Low Vol Harvesting | 86 (17%) | -$1,728,279 | -$20,096 | -3.03 |
| Transitional | 205 (41%) | $1,170,060 | $5,708 | 0.50 |
| Crisis | 213 (42%) | $36,076,589 | $169,374 | 7.94 |

### Strategy Contribution

| Strategy | Total PnL | Contribution | Sharpe |
|----------|-----------|-------------|--------|
| Option Overlay | $27,932,659 | 78.6% | 4.94 |
| Directional Long/Short | $9,011,660 | 25.4% | 4.39 |
| Dispersion Trading | -$26,068 | -0.1% | -0.10 |
| Dynamic Vol Targeting | -$160,076 | -0.5% | -2.63 |
| Volatility Harvesting | -$1,239,805 | -3.5% | -0.31 |

### PnL Attribution by Greek

| Component | Total PnL | % of Total |
|-----------|-----------|------------|
| Vega | $34,598,443 | 97.4% |
| Gamma | $1,838,271 | 5.2% |
| Theta | -$892,277 | -2.5% |
| Correlation | -$26,068 | -0.1% |

## Quick Start

```bash
git clone https://github.com/jawadmikhan/Vol.git
cd Vol

pip install -r requirements.txt

# Generate synthetic reference data
python -m data.generators.synthetic_data

# Run the full portfolio (all 7 stages)
python main.py

# Run the backtest
python -m backtest.run_backtest

# Run constraint validation tests
python tests/test_constraints.py
```

### Backtest Options

```bash
# Weekly rebalance (default)
python -m backtest.run_backtest

# Daily rebalance
python -m backtest.run_backtest --rebalance-days 1

# Skip chart generation
python -m backtest.run_backtest --no-plot

# Custom output path
python -m backtest.run_backtest --output results.csv
```

## Infrastructure (IBKR + TimescaleDB)

The repository includes a production-ready data infrastructure layer for running the strategies on live market data via Interactive Brokers.

### Architecture

```
IBKR Gateway --> IBKRClient --> LiveDataAdapter --> data dict
                                     |                  |
                                     v                  v
                                TimescaleDB      Strategies (unchanged)
                                (11 tables)            |
                                     ^                  v
                                     +---- Risk Engine --> Greeks / Scenarios / PnL
```

### Setup

```bash
cp .env.example .env            # Add your IBKR credentials
docker compose up -d            # Start TimescaleDB + IB Gateway + Vol Engine
docker compose logs -f vol-engine
```

### Components

| Component | Description |
|-----------|-------------|
| `infrastructure/ibkr/client.py` | IBKR API client — prices, option chains, VIX, streaming |
| `infrastructure/ibkr/contracts.py` | Universe: SPX + 45 constituents with sector weights |
| `infrastructure/ibkr/vol_surface.py` | Builds delta-parameterized IV surface from live chains |
| `infrastructure/ibkr/regime.py` | Real-time regime classifier (Crisis / Transitional / Low Vol) |
| `infrastructure/data_adapter.py` | Transforms IBKR data into the same dict strategies expect |
| `infrastructure/db/schema.sql` | TimescaleDB schema — 11 hypertables, compression, continuous aggregates |
| `infrastructure/db/connection.py` | Connection pool and read/write helpers |
| `infrastructure/run.py` | Live orchestrator — one-shot or scheduled during market hours |

## Repository Structure

```
Vol/
├── index.html                          # Interactive dashboard (live site)
├── workflow.html                       # Full-page Mermaid workflow map
├── portfolio.html                      # Mock portfolio page
├── backtest.html                       # Backtest results dashboard
├── main.py                             # 7-stage orchestrator
├── docker-compose.yml                  # TimescaleDB + IB Gateway + Vol Engine
├── Dockerfile                          # Python 3.12 container
├── .env.example                        # IBKR credentials template
├── requirements.txt                    # Dependencies
├── config/
│   └── portfolio_constraints.py        # Investment Committee mandate parameters
├── data/
│   ├── generators/
│   │   └── synthetic_data.py           # Generates all 6 reference files
│   └── reference/                      # Generated CSV/JSON reference data
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
│   └── attribution.py                  # Daily PnL decomposition
├── backtest/
│   ├── engine.py                       # Walk-forward backtest engine
│   ├── analytics.py                    # Performance metrics + visualization
│   └── run_backtest.py                 # Entry point with CLI options
├── infrastructure/
│   ├── ibkr/                           # IBKR API integration
│   ├── db/                             # TimescaleDB schema + connection
│   ├── data_adapter.py                 # Live data → strategy interface bridge
│   └── run.py                          # Live orchestrator
├── tests/
│   └── test_constraints.py             # Constraint validation tests
└── notebooks/
    └── portfolio_explorer.py           # Reference data inspection
```

## Workflow Stages

| Stage | Focus | Steps |
|-------|-------|-------|
| 1. Idea Generation & Sourcing | Signal construction | 7 |
| 2. Research Aggregation | Model calibration | 6 |
| 3. Collaboration & Peer Review | Stress testing & review | 5 |
| 4. Thesis Development | Portfolio design | 11 |
| 5. Decision, Approval, & Portfolio Construction | IC vote & execution | 7 |
| 6. Monitoring & Review | Greeks & attribution tracking | 6 |
| 7. Rebalancing & Liquidity | Option rolls & reporting | 6 |

## Reference Data

All synthetic — no proprietary or live market data. Formatted to match OptionMetrics and FactSet conventions.

| File | Description |
|------|-------------|
| `implied_vol_surface.csv` | 1,380 rows — 5 strikes x 6 tenors x 46 names |
| `realized_vol_history.csv` | 1,260 days x 46 names — GARCH(1,1) simulated |
| `correlation_matrix.csv` | 46x46 pairwise 90-day realized correlations |
| `vol_regime_signals.csv` | 504 days — VIX term structure, VVIX, regime classification |
| `portfolio_constraints.json` | Investment Committee mandate in JSON format |
| `option_overlay_specs.json` | Put/collar program specifications |

---

Systematic Volatility Portfolio · Cross-Asset Trading Strategist / Systematic Quant / Macro Trader
