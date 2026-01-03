#!/usr/bin/env python
"""
Full Multi-Underlying Backtest Runner

Runs comprehensive backtests for all configured underlyings:
- SPY: Daily 0DTE (Mon-Fri) with 20% IV filter
- QQQ: Mon/Wed/Fri 0DTE with 20% IV filter
- 10 Tech Stocks: Friday-only 0DTE, no IV filter

Usage:
    python run_full_backtest.py              # Run all backtests
    python run_full_backtest.py --symbol SPY # Run specific symbol only
    python run_full_backtest.py --quick      # Quick test (last 30 days)

Estimated runtime: ~1.5-2 hours for full 18-month backtest of all symbols.
"""
import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional

from config import UNDERLYINGS, get_iv_threshold
from backtest_multi import run_backtest, BacktestResult


# Date ranges for backtesting
# SPY/QQQ have daily/frequent 0DTE, stocks only have Friday 0DTE
START_DATE = "2024-01-01"  # Full 2024-2025 period
END_DATE = "2025-12-31"

RESULTS_FILE = "backtest_results.json"


def run_all_backtests(
    symbols: Optional[list[str]] = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    verbose: bool = True
) -> dict[str, BacktestResult]:
    """
    Run backtests for all configured underlyings.

    Args:
        symbols: List of symbols to test, or None for all
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        verbose: Print progress

    Returns:
        Dict mapping symbol to BacktestResult
    """
    results = {}

    # Get symbols to test
    if symbols:
        test_symbols = [s.upper() for s in symbols]
    else:
        test_symbols = list(UNDERLYINGS.keys())

    total_start = time.time()

    if verbose:
        print("=" * 70)
        print("FULL MULTI-UNDERLYING BACKTEST")
        print("=" * 70)
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Symbols: {', '.join(test_symbols)}")
        print("=" * 70)

    for i, symbol in enumerate(test_symbols, 1):
        config = UNDERLYINGS.get(symbol, {"days": [4], "iv_threshold": None})
        iv_threshold = config.get("iv_threshold")
        day_filter = config.get("days")

        if verbose:
            print(f"\n[{i}/{len(test_symbols)}] Testing {symbol}...")
            print(f"  Days: {day_filter}, IV threshold: {iv_threshold or 'None'}")

        symbol_start = time.time()

        try:
            result = run_backtest(
                underlying=symbol,
                start_date=start_date,
                end_date=end_date,
                iv_threshold=iv_threshold,
                day_filter=day_filter,
                verbose=False  # Suppress per-trade output
            )
            results[symbol] = result
            elapsed = time.time() - symbol_start

            if verbose:
                print(f"  Completed in {elapsed:.1f}s")
                print(f"  Trades: {result.total_trades}, P&L: ${result.total_pnl:.2f}")
                print(f"  OTM Rate: {result.otm_rate:.1%}, Win Rate: {result.win_rate:.1%}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[symbol] = None

    total_elapsed = time.time() - total_start

    if verbose:
        print("\n" + "=" * 70)
        print(f"BACKTEST COMPLETE in {total_elapsed/60:.1f} minutes")
        print("=" * 70)

    return results


def save_results(results: dict[str, BacktestResult], filename: str = RESULTS_FILE):
    """Save backtest results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "date_range": {"start": START_DATE, "end": END_DATE},
        "results": {}
    }

    for symbol, result in results.items():
        if result:
            data["results"][symbol] = {
                "total_trades": result.total_trades,
                "total_pnl": round(result.total_pnl, 2),
                "win_rate": round(result.win_rate, 4),
                "otm_rate": round(result.otm_rate, 4),
                "avg_iv": round(result.avg_iv, 4),
                "avg_premium": round(result.avg_premium, 2),
                "max_loss": round(result.max_loss, 2),
                "max_win": round(result.max_win, 2),
                "call_trades": result.call_trades,
                "put_trades": result.put_trades,
                "call_pnl": round(result.call_pnl, 2),
                "put_pnl": round(result.put_pnl, 2),
            }
        else:
            data["results"][symbol] = {"error": "Backtest failed"}

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {filename}")


def print_summary(results: dict[str, BacktestResult]):
    """Print summary table of all results."""
    print("\n" + "=" * 90)
    print("BACKTEST SUMMARY")
    print("=" * 90)
    print(f"{'Symbol':<8} {'Trades':>7} {'P&L':>12} {'Win%':>8} {'OTM%':>8} {'Avg IV':>8} {'Max Loss':>10}")
    print("-" * 90)

    total_trades = 0
    total_pnl = 0

    # Sort by P&L descending
    sorted_results = sorted(
        [(s, r) for s, r in results.items() if r],
        key=lambda x: x[1].total_pnl,
        reverse=True
    )

    for symbol, result in sorted_results:
        print(f"{symbol:<8} {result.total_trades:>7} ${result.total_pnl:>10,.2f} "
              f"{result.win_rate:>7.1%} {result.otm_rate:>7.1%} "
              f"{result.avg_iv:>7.1%} ${result.max_loss:>9,.2f}")
        total_trades += result.total_trades
        total_pnl += result.total_pnl

    print("-" * 90)
    print(f"{'TOTAL':<8} {total_trades:>7} ${total_pnl:>10,.2f}")
    print("=" * 90)

    # Category breakdown
    etf_symbols = ["SPY", "QQQ"]
    stock_symbols = [s for s in results.keys() if s not in etf_symbols]

    etf_pnl = sum(r.total_pnl for s, r in results.items() if r and s in etf_symbols)
    stock_pnl = sum(r.total_pnl for s, r in results.items() if r and s in stock_symbols)

    print(f"\nETFs (SPY, QQQ):     ${etf_pnl:>10,.2f}")
    print(f"Stocks (Fri only):   ${stock_pnl:>10,.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run full multi-underlying backtests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_backtest.py              # Run all 12 symbols
  python run_full_backtest.py --symbol SPY # Run SPY only
  python run_full_backtest.py --quick      # Quick 30-day test
  python run_full_backtest.py --symbol SPY --symbol QQQ  # Multiple symbols
        """
    )
    parser.add_argument(
        "--symbol", "-s",
        action="append",
        help="Specific symbol(s) to test. Can specify multiple times."
    )
    parser.add_argument(
        "--start",
        default=START_DATE,
        help=f"Start date (YYYY-MM-DD). Default: {START_DATE}"
    )
    parser.add_argument(
        "--end",
        default=END_DATE,
        help=f"End date (YYYY-MM-DD). Default: {END_DATE}"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (last 30 days only)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    # Adjust dates for quick mode
    start_date = args.start
    end_date = args.end
    if args.quick:
        from datetime import timedelta
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_dt = datetime.now() - timedelta(days=30)
        start_date = start_dt.strftime("%Y-%m-%d")
        print(f"Quick mode: {start_date} to {end_date}")

    # Run backtests
    results = run_all_backtests(
        symbols=args.symbol,
        start_date=start_date,
        end_date=end_date,
        verbose=not args.quiet
    )

    # Print summary
    if not args.quiet:
        print_summary(results)

    # Save results
    if not args.no_save:
        save_results(results)

    return 0


if __name__ == "__main__":
    exit(main())
