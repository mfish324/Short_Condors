"""
Multi-underlying backtest engine for 0DTE options strategy.
Supports SPY, QQQ, and individual stocks.
"""
import json
import sys
import time
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional
import requests
import os

from dotenv import load_dotenv

from black_scholes import (
    call_delta,
    put_delta,
    implied_volatility,
    find_strike_for_delta,
)

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io"
REQUEST_DELAY = 0.15


@dataclass
class TradeEntry:
    date: str
    underlying: str
    entry_time: str
    spot_price: float
    strike: float
    option_type: str
    delta: float
    iv: float
    option_price: float
    closing_spot: float
    expired_otm: bool
    pnl_if_sold: float


@dataclass
class BacktestResult:
    underlying: str
    start_date: str
    end_date: str
    total_trades: int
    otm_count: int
    itm_count: int
    otm_probability: float
    avg_premium_collected: float
    total_pnl: float
    avg_delta: float
    avg_iv: float
    trades: list


def get_minute_bars(date: str, underlying: str) -> list[dict]:
    """Get minute bars for any underlying."""
    url = f"{BASE_URL}/v2/aggs/ticker/{underlying}/range/1/minute/{date}/{date}"
    params = {"apiKey": API_KEY, "limit": 50000}

    try:
        resp = requests.get(url, params=params)
        time.sleep(REQUEST_DELAY)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("results", [])
    except Exception as e:
        print(f"Error fetching {underlying} bars: {e}")
    return []


def get_price_at_time(bars: list[dict], hour: int, minute: int) -> Optional[float]:
    """Get price at specific time from bars."""
    if not bars:
        return None

    target_minutes = hour * 60 + minute

    for bar in bars:
        bar_time = datetime.fromtimestamp(bar["t"] / 1000)
        bar_minutes = bar_time.hour * 60 + bar_time.minute

        if bar_minutes >= target_minutes - 4 and bar_minutes <= target_minutes + 1:
            return (bar["h"] + bar["l"]) / 2

    return None


def build_option_ticker(date: str, strike: float, option_type: str, underlying: str) -> str:
    """Build Polygon option ticker for any underlying."""
    dt = datetime.strptime(date, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    strike_str = f"{int(strike * 1000):08d}"
    return f"O:{underlying}{date_str}{option_type}{strike_str}"


def get_option_minute_bars(ticker: str, date: str) -> list[dict]:
    """Get minute bars for an option."""
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
    params = {"apiKey": API_KEY, "limit": 50000}

    try:
        resp = requests.get(url, params=params)
        time.sleep(REQUEST_DELAY)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("results", [])
    except Exception:
        pass
    return []


def get_option_price_at_time(bars: list[dict], hour: int, minute: int) -> Optional[float]:
    """Get option price at specific time."""
    if not bars:
        return None

    target_minutes = hour * 60 + minute

    for bar in bars:
        bar_time = datetime.fromtimestamp(bar["t"] / 1000)
        bar_minutes = bar_time.hour * 60 + bar_time.minute

        if bar_minutes >= target_minutes - 4 and bar_minutes <= target_minutes + 1:
            return (bar["h"] + bar["l"]) / 2

    return None


def estimate_iv_from_atm(spot: float, date: str, hour: int, minute: int,
                          T: float, underlying: str) -> float:
    """Estimate IV from ATM option prices."""
    r = 0.05
    atm_strike = round(spot)

    call_ticker = build_option_ticker(date, atm_strike, "C", underlying)
    put_ticker = build_option_ticker(date, atm_strike, "P", underlying)

    call_bars = get_option_minute_bars(call_ticker, date)
    put_bars = get_option_minute_bars(put_ticker, date)

    call_price = get_option_price_at_time(call_bars, hour, minute) if call_bars else None
    put_price = get_option_price_at_time(put_bars, hour, minute) if put_bars else None

    ivs = []
    if call_price and call_price > 0.05:
        iv = implied_volatility(call_price, spot, atm_strike, T, r, "call")
        if iv and 0.05 < iv < 2.0:
            ivs.append(iv)

    if put_price and put_price > 0.05:
        iv = implied_volatility(put_price, spot, atm_strike, T, r, "put")
        if iv and 0.05 < iv < 2.0:
            ivs.append(iv)

    return sum(ivs) / len(ivs) if ivs else 0.25


def run_backtest(
    underlying: str,
    start_date: str,
    end_date: str,
    entry_times: list[tuple[int, int]] = [(10, 0), (10, 30), (11, 0)],
    target_delta: float = 0.10,
    iv_threshold: Optional[float] = 0.20,
    day_filter: Optional[list[int]] = None,  # 0=Mon, 4=Fri, None=all weekdays
    verbose: bool = True
) -> BacktestResult:
    """
    Run backtest for any underlying.

    Args:
        underlying: Stock/ETF symbol (SPY, QQQ, AAPL, etc.)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        entry_times: List of (hour, minute) tuples
        target_delta: Target delta for options
        iv_threshold: Skip trades when IV > threshold (None to disable)
        day_filter: List of weekday numbers to trade (0=Mon, 4=Fri), None for all
        verbose: Print progress
    """
    trades = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    r = 0.05

    skipped_iv = 0

    if verbose:
        print(f"\nBacktesting {underlying}: {start_date} to {end_date}")
        if day_filter:
            days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
            print(f"Trading days: {[days[d] for d in day_filter]}")
        if iv_threshold:
            print(f"IV threshold: {iv_threshold:.0%}")
        print("-" * 50)

    while current <= end:
        # Skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        # Apply day filter (e.g., Friday only for stocks)
        if day_filter and current.weekday() not in day_filter:
            current += timedelta(days=1)
            continue

        date_str = current.strftime("%Y-%m-%d")
        bars = get_minute_bars(date_str, underlying)

        if len(bars) < 50:
            current += timedelta(days=1)
            continue

        closing_price = bars[-1]["c"]

        day_trades = []
        day_skipped = False

        for hour, minute in entry_times:
            spot = get_price_at_time(bars, hour, minute)
            if spot is None:
                continue

            hours_to_close = (16 - hour) - minute / 60
            T = hours_to_close / 24 / 365

            # Estimate IV
            iv = estimate_iv_from_atm(spot, date_str, hour, minute, T, underlying)

            # IV Filter
            if iv_threshold and iv > iv_threshold:
                skipped_iv += 2  # Would have done call + put
                day_skipped = True
                continue

            # CALL
            call_strike_theory = find_strike_for_delta(target_delta, spot, T, r, iv, "call")
            if call_strike_theory:
                call_strike = round(call_strike_theory)
                ticker = build_option_ticker(date_str, call_strike, "C", underlying)
                opt_bars = get_option_minute_bars(ticker, date_str)
                price = get_option_price_at_time(opt_bars, hour, minute)

                if price and price > 0.01:
                    actual_iv = implied_volatility(price, spot, call_strike, T, r, "call") or iv
                    delta = call_delta(spot, call_strike, T, r, actual_iv)

                    expired_otm = closing_price < call_strike
                    pnl = price * 100 if expired_otm else (price - (closing_price - call_strike)) * 100

                    day_trades.append(TradeEntry(
                        date=date_str,
                        underlying=underlying,
                        entry_time=f"{hour}:{minute:02d}",
                        spot_price=spot,
                        strike=call_strike,
                        option_type="call",
                        delta=delta,
                        iv=actual_iv,
                        option_price=price,
                        closing_spot=closing_price,
                        expired_otm=expired_otm,
                        pnl_if_sold=pnl
                    ))

            # PUT
            put_strike_theory = find_strike_for_delta(-target_delta, spot, T, r, iv, "put")
            if put_strike_theory:
                put_strike = round(put_strike_theory)
                ticker = build_option_ticker(date_str, put_strike, "P", underlying)
                opt_bars = get_option_minute_bars(ticker, date_str)
                price = get_option_price_at_time(opt_bars, hour, minute)

                if price and price > 0.01:
                    actual_iv = implied_volatility(price, spot, put_strike, T, r, "put") or iv
                    delta = put_delta(spot, put_strike, T, r, actual_iv)

                    expired_otm = closing_price > put_strike
                    pnl = price * 100 if expired_otm else (price - (put_strike - closing_price)) * 100

                    day_trades.append(TradeEntry(
                        date=date_str,
                        underlying=underlying,
                        entry_time=f"{hour}:{minute:02d}",
                        spot_price=spot,
                        strike=put_strike,
                        option_type="put",
                        delta=abs(delta),
                        iv=actual_iv,
                        option_price=price,
                        closing_spot=closing_price,
                        expired_otm=expired_otm,
                        pnl_if_sold=pnl
                    ))

        if verbose:
            if day_trades:
                otm = sum(1 for t in day_trades if t.expired_otm)
                pnl = sum(t.pnl_if_sold for t in day_trades)
                print(f"{date_str} | {underlying} ${closing_price:.2f} | {otm}/{len(day_trades)} OTM | ${pnl:+.0f}")
            elif day_skipped:
                print(f"{date_str} | {underlying} | SKIPPED (IV > {iv_threshold:.0%})")

        trades.extend(day_trades)
        current += timedelta(days=1)

    # Calculate stats
    total = len(trades)
    if total == 0:
        return BacktestResult(underlying, start_date, end_date, 0, 0, 0, 0, 0, 0, 0, 0, [])

    otm_count = sum(1 for t in trades if t.expired_otm)

    return BacktestResult(
        underlying=underlying,
        start_date=start_date,
        end_date=end_date,
        total_trades=total,
        otm_count=otm_count,
        itm_count=total - otm_count,
        otm_probability=otm_count / total if total > 0 else 0,
        avg_premium_collected=sum(t.option_price for t in trades) / total * 100,
        total_pnl=sum(t.pnl_if_sold for t in trades),
        avg_delta=sum(t.delta for t in trades) / total,
        avg_iv=sum(t.iv for t in trades) / total,
        trades=[asdict(t) for t in trades]
    )


def print_results(result: BacktestResult):
    """Print backtest results summary."""
    print(f"\n{'=' * 60}")
    print(f"BACKTEST RESULTS: {result.underlying}")
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"{'=' * 60}")

    if result.total_trades == 0:
        print("\nNo trades executed.")
        return

    print(f"\nTotal Trades: {result.total_trades}")
    print(f"Average Delta: {result.avg_delta:.3f}")
    print(f"Average IV: {result.avg_iv:.1%}")

    print(f"\n--- PROBABILITY ---")
    print(f"Expired OTM: {result.otm_count} ({result.otm_probability:.1%})")
    print(f"Expired ITM: {result.itm_count} ({1-result.otm_probability:.1%})")

    print(f"\n--- P&L ---")
    print(f"Avg Premium: ${result.avg_premium_collected:.2f}")
    print(f"Total P&L: ${result.total_pnl:,.2f}")

    # By type
    calls = [t for t in result.trades if t["option_type"] == "call"]
    puts = [t for t in result.trades if t["option_type"] == "put"]

    if calls:
        call_otm = sum(1 for t in calls if t["expired_otm"])
        call_pnl = sum(t["pnl_if_sold"] for t in calls)
        print(f"\nCALLS: {call_otm}/{len(calls)} OTM ({call_otm/len(calls):.1%}) | ${call_pnl:+,.0f}")

    if puts:
        put_otm = sum(1 for t in puts if t["expired_otm"])
        put_pnl = sum(t["pnl_if_sold"] for t in puts)
        print(f"PUTS:  {put_otm}/{len(puts)} OTM ({put_otm/len(puts):.1%}) | ${put_pnl:+,.0f}")


def run_all_backtests(start_date: str, end_date: str, verbose: bool = True):
    """Run backtests for all underlyings."""
    results = {}

    # SPY - all weekdays
    print("\n" + "=" * 60)
    print("BACKTESTING SPY (Mon-Fri)")
    print("=" * 60)
    results["SPY"] = run_backtest("SPY", start_date, end_date,
                                   iv_threshold=0.20, verbose=verbose)

    # QQQ - Mon/Wed/Fri only
    print("\n" + "=" * 60)
    print("BACKTESTING QQQ (Mon/Wed/Fri)")
    print("=" * 60)
    results["QQQ"] = run_backtest("QQQ", start_date, end_date,
                                   iv_threshold=0.20,
                                   day_filter=[0, 2, 4],  # Mon, Wed, Fri
                                   verbose=verbose)

    # Stocks - Friday only
    stocks = ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "GOOGL", "META", "TSLA", "NFLX", "CRM"]

    for stock in stocks:
        print(f"\n" + "=" * 60)
        print(f"BACKTESTING {stock} (Fridays only)")
        print("=" * 60)
        results[stock] = run_backtest(stock, start_date, end_date,
                                       iv_threshold=0.20,
                                       day_filter=[4],  # Friday only
                                       verbose=verbose)

    return results


def print_summary(results: dict):
    """Print summary of all backtest results."""
    print("\n" + "=" * 70)
    print("MULTI-UNDERLYING BACKTEST SUMMARY")
    print("=" * 70)

    print(f"\n{'Symbol':<8} {'Trades':>8} {'OTM %':>8} {'Avg IV':>8} {'Total P&L':>12} {'Per Trade':>10}")
    print("-" * 70)

    total_trades = 0
    total_pnl = 0

    for symbol, r in results.items():
        if r.total_trades > 0:
            per_trade = r.total_pnl / r.total_trades
            print(f"{symbol:<8} {r.total_trades:>8} {r.otm_probability:>7.1%} {r.avg_iv:>7.1%} ${r.total_pnl:>10,.0f} ${per_trade:>9.2f}")
            total_trades += r.total_trades
            total_pnl += r.total_pnl
        else:
            print(f"{symbol:<8} {'N/A':>8}")

    print("-" * 70)
    if total_trades > 0:
        print(f"{'TOTAL':<8} {total_trades:>8} {'':>8} {'':>8} ${total_pnl:>10,.0f} ${total_pnl/total_trades:>9.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-underlying 0DTE options backtest")
    parser.add_argument("--underlying", "-u", default=None, help="Single underlying to test")
    parser.add_argument("--start", "-s", default="2024-07-01", help="Start date")
    parser.add_argument("--end", "-e", default="2025-12-31", help="End date")
    parser.add_argument("--all", "-a", action="store_true", help="Run all underlyings")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if args.all:
        results = run_all_backtests(args.start, args.end, verbose=not args.quiet)
        print_summary(results)

        # Save results
        with open("backtest_multi_results.json", "w") as f:
            summary = {sym: {k: v for k, v in asdict(r).items() if k != "trades"}
                       for sym, r in results.items()}
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to backtest_multi_results.json")

    elif args.underlying:
        # Determine day filter based on underlying
        day_filter = None
        if args.underlying == "QQQ":
            day_filter = [0, 2, 4]  # Mon/Wed/Fri
        elif args.underlying not in ["SPY"]:
            day_filter = [4]  # Friday only for stocks

        result = run_backtest(args.underlying, args.start, args.end,
                              iv_threshold=0.20, day_filter=day_filter,
                              verbose=not args.quiet)
        print_results(result)
    else:
        parser.print_help()
