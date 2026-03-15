"""
Backtest Analytics — Performance Metrics and Visualization
============================================================
Computes institutional-grade performance metrics from backtest results
and generates the charts a PM needs to evaluate the strategy.

Metrics:
  - Sharpe ratio (annualized)
  - Max drawdown ($ and %)
  - Calmar ratio
  - Win rate
  - Regime-conditional performance
  - Strategy contribution
  - Monthly returns table
  - Greeks utilization over time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "backtest" / "output"


def compute_metrics(results: pd.DataFrame, capital: float = 250_000_000) -> dict:
    """
    Compute key performance metrics from backtest results.

    Args:
        results: DataFrame from BacktestEngine.run()
        capital: Initial capital for return calculations.

    Returns:
        Dict of performance metrics.
    """
    daily_pnl = results["total_pnl"]
    cum_pnl = results["cumulative_pnl"]
    n_days = len(results)
    n_years = n_days / 252

    # Returns
    daily_returns = daily_pnl / capital
    total_return = cum_pnl.iloc[-1] / capital
    annualized_return = total_return / n_years if n_years > 0 else 0.0

    # Sharpe ratio (annualized, assuming 0% risk-free for simplicity)
    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()
    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

    # Sortino ratio (downside deviation only)
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-10
    sortino = (daily_mean / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

    # Max drawdown
    max_dd_dollars = results["drawdown"].max()
    max_dd_pct = results["drawdown_pct"].max()

    # Calmar ratio (annualized return / max drawdown)
    calmar = annualized_return / max_dd_pct if max_dd_pct > 0 else 0.0

    # Win rate
    win_days = (daily_pnl > 0).sum()
    loss_days = (daily_pnl < 0).sum()
    flat_days = (daily_pnl == 0).sum()
    win_rate = win_days / (win_days + loss_days) if (win_days + loss_days) > 0 else 0.0

    # Profit factor
    gross_profit = daily_pnl[daily_pnl > 0].sum()
    gross_loss = abs(daily_pnl[daily_pnl < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average win / average loss
    avg_win = daily_pnl[daily_pnl > 0].mean() if win_days > 0 else 0.0
    avg_loss = daily_pnl[daily_pnl < 0].mean() if loss_days > 0 else 0.0

    # Best / worst day
    best_day = daily_pnl.max()
    worst_day = daily_pnl.min()

    # Average daily PnL
    avg_daily_pnl = daily_pnl.mean()

    # Vega bound breaches
    vega_breaches = (~results["vega_within_bounds"]).sum()
    notional_breaches = (~results["notional_within_cap"]).sum()

    return {
        "total_pnl": cum_pnl.iloc[-1],
        "total_return_pct": total_return * 100,
        "annualized_return_pct": annualized_return * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_usd": max_dd_dollars,
        "max_drawdown_pct": max_dd_pct * 100,
        "calmar_ratio": calmar,
        "win_rate_pct": win_rate * 100,
        "profit_factor": profit_factor,
        "avg_daily_pnl": avg_daily_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_day": best_day,
        "worst_day": worst_day,
        "win_days": win_days,
        "loss_days": loss_days,
        "flat_days": flat_days,
        "trading_days": n_days,
        "vega_breaches": vega_breaches,
        "notional_breaches": notional_breaches,
    }


def regime_performance(results: pd.DataFrame, capital: float = 250_000_000) -> pd.DataFrame:
    """Break down performance by volatility regime."""
    rows = []
    for regime in ["LOW_VOL_HARVESTING", "TRANSITIONAL", "CRISIS"]:
        mask = results["regime"] == regime
        subset = results[mask]
        if subset.empty:
            continue

        daily_pnl = subset["total_pnl"]
        daily_ret = daily_pnl / capital
        n = len(subset)

        rows.append({
            "regime": regime,
            "days": n,
            "pct_of_total": n / len(results) * 100,
            "total_pnl": daily_pnl.sum(),
            "avg_daily_pnl": daily_pnl.mean(),
            "sharpe": (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0,
            "win_rate_pct": (daily_pnl > 0).sum() / max(1, ((daily_pnl > 0).sum() + (daily_pnl < 0).sum())) * 100,
            "worst_day": daily_pnl.min(),
            "best_day": daily_pnl.max(),
        })

    return pd.DataFrame(rows)


def strategy_contribution(results: pd.DataFrame) -> pd.DataFrame:
    """Decompose total PnL by strategy."""
    # strategy_pnl column contains dicts
    strat_pnls = pd.DataFrame(results["strategy_pnl"].tolist())

    if strat_pnls.empty:
        return pd.DataFrame()

    rows = []
    total = strat_pnls.sum().sum()

    for col in strat_pnls.columns:
        pnl = strat_pnls[col].sum()
        rows.append({
            "strategy": col,
            "total_pnl": pnl,
            "contribution_pct": (pnl / total * 100) if total != 0 else 0.0,
            "avg_daily_pnl": strat_pnls[col].mean(),
            "daily_std": strat_pnls[col].std(),
            "sharpe": (strat_pnls[col].mean() / strat_pnls[col].std() * np.sqrt(252))
            if strat_pnls[col].std() > 0 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)


def monthly_returns(results: pd.DataFrame, capital: float = 250_000_000) -> pd.DataFrame:
    """Compute monthly returns table (rows = years, columns = months)."""
    df = results[["date", "total_pnl"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly = df.groupby(["year", "month"])["total_pnl"].sum() / capital * 100
    table = monthly.unstack(level="month")
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(table.columns)]

    # Add annual total
    table["Year Total"] = table.sum(axis=1)

    return table


def greek_attribution(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize PnL attribution by Greek component."""
    return pd.DataFrame({
        "component": ["Vega", "Gamma", "Theta", "Correlation", "Total"],
        "total_pnl": [
            results["vega_pnl"].sum(),
            results["gamma_pnl"].sum(),
            results["theta_pnl"].sum(),
            results["correlation_pnl"].sum(),
            results["total_pnl"].sum(),
        ],
        "avg_daily": [
            results["vega_pnl"].mean(),
            results["gamma_pnl"].mean(),
            results["theta_pnl"].mean(),
            results["correlation_pnl"].mean(),
            results["total_pnl"].mean(),
        ],
        "std_daily": [
            results["vega_pnl"].std(),
            results["gamma_pnl"].std(),
            results["theta_pnl"].std(),
            results["correlation_pnl"].std(),
            results["total_pnl"].std(),
        ],
    })


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_backtest(results: pd.DataFrame, capital: float = 250_000_000, save: bool = True):
    """Generate the full backtest report as a multi-panel chart."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(16, 20), gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    fig.suptitle("Systematic Volatility Portfolio — Backtest Results", fontsize=16, fontweight="bold")

    dates = pd.to_datetime(results["date"])

    # ------------------------------------------------------------------
    # Panel 1: Cumulative PnL with regime shading
    # ------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(dates, results["cumulative_pnl"] / 1e6, color="#1a5276", linewidth=1.5, label="Portfolio PnL")
    ax1.fill_between(dates, 0, results["cumulative_pnl"] / 1e6,
                     where=results["cumulative_pnl"] > 0, alpha=0.15, color="green")
    ax1.fill_between(dates, 0, results["cumulative_pnl"] / 1e6,
                     where=results["cumulative_pnl"] < 0, alpha=0.15, color="red")

    # Shade regime periods
    crisis_mask = results["regime"] == "CRISIS"
    for start, end in _contiguous_ranges(dates, crisis_mask):
        ax1.axvspan(start, end, alpha=0.08, color="red", label="_nolegend_")

    ax1.set_ylabel("Cumulative PnL ($M)")
    ax1.set_title("Cumulative PnL with Crisis Regime Shading")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.1f}M"))
    ax1.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 2: Daily PnL bar chart
    # ------------------------------------------------------------------
    ax2 = axes[1]
    colors = ["green" if x > 0 else "red" for x in results["total_pnl"]]
    ax2.bar(dates, results["total_pnl"] / 1e6, color=colors, alpha=0.6, width=1)
    ax2.set_ylabel("Daily PnL ($M)")
    ax2.set_title("Daily PnL")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.1f}M"))
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 3: Drawdown
    # ------------------------------------------------------------------
    ax3 = axes[2]
    ax3.fill_between(dates, -results["drawdown_pct"] * 100, 0, color="red", alpha=0.3)
    ax3.plot(dates, -results["drawdown_pct"] * 100, color="darkred", linewidth=0.8)
    ax3.set_ylabel("Drawdown (%)")
    ax3.set_title("Drawdown from Peak")
    ax3.axhline(y=-MAX_PORTFOLIO_DRAWDOWN * 100, color="red", linewidth=1, linestyle="--",
                label=f"Stop Loss ({MAX_PORTFOLIO_DRAWDOWN*100:.0f}%)")
    ax3.legend(loc="lower left")
    ax3.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 4: VIX level with regime coloring + net vega
    # ------------------------------------------------------------------
    ax4 = axes[3]
    regime_colors = {
        "LOW_VOL_HARVESTING": "green",
        "TRANSITIONAL": "orange",
        "CRISIS": "red",
    }
    for regime, color in regime_colors.items():
        mask = results["regime"] == regime
        if mask.any():
            ax4.scatter(dates[mask], results.loc[mask, "vix"],
                       c=color, s=3, alpha=0.7, label=regime)

    ax4.set_ylabel("VIX Level")
    ax4.set_title("VIX Level by Regime")
    ax4.legend(loc="upper right", markerscale=5)
    ax4.grid(True, alpha=0.3)

    # Vega on secondary axis
    ax4b = ax4.twinx()
    ax4b.plot(dates, results["net_vega"] / 1e6, color="purple", linewidth=0.8, alpha=0.5, label="Net Vega ($M)")
    ax4b.set_ylabel("Net Vega ($M)")
    ax4b.legend(loc="upper left")

    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "backtest_report.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved: {path}")

    return fig


def _contiguous_ranges(dates, mask):
    """Yield (start, end) pairs for contiguous True values in mask."""
    in_range = False
    start = None
    for i, val in enumerate(mask):
        if val and not in_range:
            start = dates.iloc[i]
            in_range = True
        elif not val and in_range:
            yield start, dates.iloc[i - 1]
            in_range = False
    if in_range:
        yield start, dates.iloc[-1]


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(results: pd.DataFrame, capital: float = 250_000_000):
    """Print the full backtest performance report to console."""
    metrics = compute_metrics(results, capital)
    regime_perf = regime_performance(results, capital)
    strat_contrib = strategy_contribution(results)
    greek_attr = greek_attribution(results)

    print()
    print("=" * 80)
    print("  BACKTEST PERFORMANCE REPORT")
    print("=" * 80)

    print(f"\n  Period:              {results['date'].iloc[0]} to {results['date'].iloc[-1]}")
    print(f"  Trading Days:        {metrics['trading_days']}")
    print(f"  Initial Capital:     ${capital:,.0f}")

    print(f"\n  {'-' * 50}")
    print(f"  RETURNS")
    print(f"  {'-' * 50}")
    print(f"  Total PnL:           ${metrics['total_pnl']:>14,.0f}")
    print(f"  Total Return:        {metrics['total_return_pct']:>13.2f}%")
    print(f"  Annualized Return:   {metrics['annualized_return_pct']:>13.2f}%")

    print(f"\n  {'-' * 50}")
    print(f"  RISK-ADJUSTED")
    print(f"  {'-' * 50}")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>13.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>13.2f}")
    print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>13.2f}")
    print(f"  Max Drawdown:        ${metrics['max_drawdown_usd']:>14,.0f} ({metrics['max_drawdown_pct']:.2f}%)")
    print(f"  Profit Factor:       {metrics['profit_factor']:>13.2f}")

    print(f"\n  {'-' * 50}")
    print(f"  DAILY STATISTICS")
    print(f"  {'-' * 50}")
    print(f"  Avg Daily PnL:       ${metrics['avg_daily_pnl']:>14,.0f}")
    print(f"  Win Rate:            {metrics['win_rate_pct']:>13.1f}%")
    print(f"  Win / Loss Days:     {metrics['win_days']} / {metrics['loss_days']}")
    print(f"  Avg Win:             ${metrics['avg_win']:>14,.0f}")
    print(f"  Avg Loss:            ${metrics['avg_loss']:>14,.0f}")
    print(f"  Best Day:            ${metrics['best_day']:>14,.0f}")
    print(f"  Worst Day:           ${metrics['worst_day']:>14,.0f}")

    print(f"\n  {'-' * 50}")
    print(f"  RISK COMPLIANCE")
    print(f"  {'-' * 50}")
    print(f"  Vega Bound Breaches: {metrics['vega_breaches']}")
    print(f"  Notional Cap Breaches: {metrics['notional_breaches']}")

    # Greek attribution
    print(f"\n  {'-' * 50}")
    print(f"  PNL ATTRIBUTION BY GREEK")
    print(f"  {'-' * 50}")
    print(f"  {'Component':<20} {'Total PnL':>14} {'Avg Daily':>14} {'Std Daily':>14}")
    for _, row in greek_attr.iterrows():
        print(f"  {row['component']:<20} ${row['total_pnl']:>13,.0f} ${row['avg_daily']:>13,.0f} ${row['std_daily']:>13,.0f}")

    # Regime performance
    if not regime_perf.empty:
        print(f"\n  {'-' * 50}")
        print(f"  PERFORMANCE BY REGIME")
        print(f"  {'-' * 50}")
        print(f"  {'Regime':<25} {'Days':>6} {'Total PnL':>14} {'Avg Daily':>14} {'Sharpe':>8} {'Win%':>7}")
        for _, row in regime_perf.iterrows():
            print(f"  {row['regime']:<25} {row['days']:>6.0f} ${row['total_pnl']:>13,.0f} "
                  f"${row['avg_daily_pnl']:>13,.0f} {row['sharpe']:>8.2f} {row['win_rate_pct']:>6.1f}%")

    # Strategy contribution
    if not strat_contrib.empty:
        print(f"\n  {'-' * 50}")
        print(f"  STRATEGY CONTRIBUTION")
        print(f"  {'-' * 50}")
        print(f"  {'Strategy':<35} {'Total PnL':>14} {'Contrib%':>9} {'Sharpe':>8}")
        for _, row in strat_contrib.iterrows():
            print(f"  {row['strategy']:<35} ${row['total_pnl']:>13,.0f} {row['contribution_pct']:>8.1f}% "
                  f"{row['sharpe']:>8.2f}")

    print(f"\n{'=' * 80}")

    return metrics
