#!/usr/bin/env python
"""
Portfolio Analytics for Backtest Results

Calculates advanced metrics:
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Calmar Ratio
- Profit Factor
- And more...

Usage:
    python analyze_results.py                    # Analyze all symbols
    python analyze_results.py --symbol SPY       # Analyze specific symbol
    python analyze_results.py --combined         # Analyze combined portfolio
"""
import argparse
import json
import math
from datetime import datetime
from typing import Optional
from collections import defaultdict

from config import UNDERLYINGS
from backtest_multi import run_backtest


def calculate_metrics(trades: list[dict], risk_free_rate: float = 0.05) -> dict:
    """
    Calculate comprehensive portfolio metrics from trade list.

    Args:
        trades: List of trade dicts with 'date', 'pnl', etc.
        risk_free_rate: Annual risk-free rate (default 5%)

    Returns:
        Dict of calculated metrics
    """
    if not trades:
        return {"error": "No trades"}

    # Extract P&L values (field is 'pnl_if_sold' in backtest results)
    pnls = [t.get("pnl_if_sold", t.get("pnl", 0)) for t in trades]

    # Group by date for daily P&L
    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t.get("date", "unknown")] += t.get("pnl_if_sold", t.get("pnl", 0))

    daily_returns = list(daily_pnl.values())
    trading_days = len(daily_pnl)

    # Basic stats
    total_pnl = sum(pnls)
    total_trades = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    # Profit Factor = Gross Profit / Gross Loss
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max Win / Max Loss
    max_win = max(pnls) if pnls else 0
    max_loss = min(pnls) if pnls else 0

    # Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # === Cumulative P&L and Drawdown ===
    cumulative = []
    running_total = 0
    for pnl in pnls:
        running_total += pnl
        cumulative.append(running_total)

    # Max Drawdown
    peak = cumulative[0]
    max_drawdown = 0
    max_drawdown_pct = 0

    for value in cumulative:
        if value > peak:
            peak = value
        drawdown = peak - value
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            # Drawdown as % of peak (if peak > 0)
            max_drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0

    # === Risk-Adjusted Returns ===

    # Daily metrics
    if len(daily_returns) > 1:
        mean_daily = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_daily) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_daily = math.sqrt(variance)

        # Downside deviation (for Sortino)
        negative_returns = [r for r in daily_returns if r < 0]
        if negative_returns:
            downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
            downside_std = math.sqrt(downside_variance)
        else:
            downside_std = 0.001  # Avoid division by zero
    else:
        mean_daily = daily_returns[0] if daily_returns else 0
        std_daily = 0.001
        downside_std = 0.001

    # Annualize (assuming 252 trading days)
    annualization_factor = math.sqrt(252)
    annual_return = mean_daily * 252
    annual_std = std_daily * annualization_factor
    annual_downside_std = downside_std * annualization_factor

    # Sharpe Ratio = (Return - Risk Free) / Std Dev
    # Using daily returns annualized
    sharpe_ratio = (annual_return - risk_free_rate) / annual_std if annual_std > 0 else 0

    # Sortino Ratio = (Return - Risk Free) / Downside Std Dev
    sortino_ratio = (annual_return - risk_free_rate) / annual_downside_std if annual_downside_std > 0 else 0

    # Calmar Ratio = Annual Return / Max Drawdown
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else float('inf')

    # === Additional Metrics ===

    # Recovery Factor = Total P&L / Max Drawdown
    recovery_factor = total_pnl / max_drawdown if max_drawdown > 0 else float('inf')

    # Avg Trade
    avg_trade = total_pnl / total_trades if total_trades > 0 else 0

    # Payoff Ratio = Avg Win / |Avg Loss|
    payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')

    return {
        # Basic
        "total_trades": total_trades,
        "trading_days": trading_days,
        "total_pnl": round(total_pnl, 2),
        "avg_trade": round(avg_trade, 2),

        # Win/Loss
        "win_rate": round(win_rate * 100, 1),
        "wins": len(wins),
        "losses": len(losses),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_win": round(max_win, 2),
        "max_loss": round(max_loss, 2),

        # Risk Metrics
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 1),
        "profit_factor": round(profit_factor, 2),
        "payoff_ratio": round(payoff_ratio, 2),
        "expectancy": round(expectancy, 2),
        "recovery_factor": round(recovery_factor, 2),

        # Risk-Adjusted Returns
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "calmar_ratio": round(calmar_ratio, 2),

        # Annualized
        "annual_return": round(annual_return, 2),
        "annual_volatility": round(annual_std, 2),
    }


def print_metrics(symbol: str, metrics: dict):
    """Pretty print metrics for a symbol."""
    print(f"\n{'='*60}")
    print(f"  {symbol} PERFORMANCE METRICS")
    print(f"{'='*60}")

    if "error" in metrics:
        print(f"  Error: {metrics['error']}")
        return

    print(f"\n  --- BASIC STATS ---")
    print(f"  Total Trades:      {metrics['total_trades']:,}")
    print(f"  Trading Days:      {metrics['trading_days']:,}")
    print(f"  Total P&L:         ${metrics['total_pnl']:,.2f}")
    print(f"  Avg Trade:         ${metrics['avg_trade']:.2f}")

    print(f"\n  --- WIN/LOSS ---")
    print(f"  Win Rate:          {metrics['win_rate']:.1f}%")
    print(f"  Wins/Losses:       {metrics['wins']} / {metrics['losses']}")
    print(f"  Avg Win:           ${metrics['avg_win']:.2f}")
    print(f"  Avg Loss:          ${metrics['avg_loss']:.2f}")
    print(f"  Max Win:           ${metrics['max_win']:.2f}")
    print(f"  Max Loss:          ${metrics['max_loss']:.2f}")

    print(f"\n  --- RISK METRICS ---")
    print(f"  Max Drawdown:      ${metrics['max_drawdown']:,.2f}")
    print(f"  Profit Factor:     {metrics['profit_factor']:.2f}")
    print(f"  Payoff Ratio:      {metrics['payoff_ratio']:.2f}")
    print(f"  Expectancy:        ${metrics['expectancy']:.2f}")
    print(f"  Recovery Factor:   {metrics['recovery_factor']:.2f}")

    print(f"\n  --- RISK-ADJUSTED RETURNS ---")
    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:     {metrics['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio:      {metrics['calmar_ratio']:.2f}")

    print(f"\n  --- ANNUALIZED ---")
    print(f"  Annual Return:     ${metrics['annual_return']:,.2f}")
    print(f"  Annual Volatility: ${metrics['annual_volatility']:,.2f}")


def analyze_symbol(symbol: str, start_date: str, end_date: str,
                   day_filter: list = None, iv_threshold: float = None) -> dict:
    """Run backtest and return metrics for a single symbol."""
    print(f"  Analyzing {symbol}...", end=" ", flush=True)

    result = run_backtest(
        underlying=symbol,
        start_date=start_date,
        end_date=end_date,
        iv_threshold=iv_threshold,
        day_filter=day_filter,
        verbose=False
    )

    metrics = calculate_metrics(result.trades)
    print(f"done ({result.total_trades} trades)")

    return metrics, result.trades


def main():
    parser = argparse.ArgumentParser(description="Analyze backtest results")
    parser.add_argument("--symbol", "-s", help="Analyze specific symbol")
    parser.add_argument("--combined", "-c", action="store_true",
                        help="Analyze combined portfolio")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default="2025-12-31", help="End date")
    parser.add_argument("--quick", action="store_true", help="Quick 3-month test")

    args = parser.parse_args()

    if args.quick:
        args.start = "2025-10-01"
        args.end = "2025-12-31"

    print(f"\nAnalyzing period: {args.start} to {args.end}")

    if args.symbol:
        # Single symbol
        symbol = args.symbol.upper()
        config = UNDERLYINGS.get(symbol, {"days": [4], "iv_threshold": None})

        metrics, _ = analyze_symbol(
            symbol, args.start, args.end,
            day_filter=config.get("days"),
            iv_threshold=config.get("iv_threshold")
        )
        print_metrics(symbol, metrics)

    elif args.combined:
        # Combined portfolio - all symbols together
        print("\nRunning backtests for combined portfolio analysis...")
        all_trades = []

        for symbol, config in UNDERLYINGS.items():
            if not config.get("enabled", True):
                continue

            try:
                _, trades = analyze_symbol(
                    symbol, args.start, args.end,
                    day_filter=config.get("days"),
                    iv_threshold=config.get("iv_threshold")
                )
                all_trades.extend(trades)
            except Exception as e:
                print(f"  Error with {symbol}: {e}")

        # Sort by date for proper drawdown calculation
        all_trades.sort(key=lambda t: (t.get("date", ""), t.get("entry_time", "")))

        metrics = calculate_metrics(all_trades)
        print_metrics("COMBINED PORTFOLIO", metrics)

    else:
        # All symbols individually
        print("\nRunning backtests for all symbols...")
        all_metrics = {}

        for symbol, config in UNDERLYINGS.items():
            if not config.get("enabled", True):
                continue

            try:
                metrics, _ = analyze_symbol(
                    symbol, args.start, args.end,
                    day_filter=config.get("days"),
                    iv_threshold=config.get("iv_threshold")
                )
                all_metrics[symbol] = metrics
            except Exception as e:
                print(f"  Error with {symbol}: {e}")
                all_metrics[symbol] = {"error": str(e)}

        # Print summary table
        print(f"\n{'='*100}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*100}")
        print(f"{'Symbol':<8} {'Trades':>7} {'P&L':>12} {'Win%':>7} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>10} {'PF':>6}")
        print("-" * 100)

        total_pnl = 0
        for symbol in ["SPY", "QQQ"] + [s for s in all_metrics if s not in ["SPY", "QQQ"]]:
            if symbol not in all_metrics:
                continue
            m = all_metrics[symbol]
            if "error" in m:
                print(f"{symbol:<8} {'ERROR':>7}")
                continue

            print(f"{symbol:<8} {m['total_trades']:>7} ${m['total_pnl']:>10,.2f} "
                  f"{m['win_rate']:>6.1f}% {m['sharpe_ratio']:>8.2f} {m['sortino_ratio']:>8.2f} "
                  f"${m['max_drawdown']:>8,.2f} {m['profit_factor']:>6.2f}")
            total_pnl += m['total_pnl']

        print("-" * 100)
        print(f"{'TOTAL':<8} {'':<7} ${total_pnl:>10,.2f}")

        # Now calculate and show combined metrics
        print("\n\nCalculating combined portfolio metrics...")
        all_trades = []
        for symbol, config in UNDERLYINGS.items():
            if not config.get("enabled", True):
                continue
            try:
                result = run_backtest(
                    underlying=symbol,
                    start_date=args.start,
                    end_date=args.end,
                    iv_threshold=config.get("iv_threshold"),
                    day_filter=config.get("days"),
                    verbose=False
                )
                all_trades.extend(result.trades)
            except:
                pass

        all_trades.sort(key=lambda t: (t.get("date", ""), t.get("entry_time", "")))
        combined_metrics = calculate_metrics(all_trades)
        print_metrics("COMBINED PORTFOLIO", combined_metrics)


if __name__ == "__main__":
    main()
