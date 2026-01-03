#!/usr/bin/env python
"""
0DTE Options Trade Scanner

Production-ready script for identifying 10-delta option trades.
Includes filters for event days and high IV conditions.

Usage:
    python trade_scanner.py              # Scan for today's trades
    python trade_scanner.py --date 2024-12-20  # Scan specific date
    python trade_scanner.py --backtest   # Run quick backtest
"""
import argparse
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

from dotenv import load_dotenv

from config import (
    TARGET_DELTA, ENTRY_TIMES, SKIP_EVENT_DAYS, SKIP_PUTS_HIGH_IV,
    SKIP_HALF_DAYS, RISK_FREE_RATE, MAX_IV_THRESHOLD,
    is_event_day, is_half_day, get_iv_threshold
)
from data_fetcher import (
    get_minute_bars, get_price_at_time,
    get_option_minute_bars, get_option_price_at_time,
    build_option_ticker
)
from black_scholes import (
    call_delta, put_delta, implied_volatility, find_strike_for_delta
)

load_dotenv()


@dataclass
class TradeSignal:
    """Represents a potential trade signal."""
    date: str
    time: str
    underlying: str
    option_type: str
    strike: float
    delta: float
    iv: float
    bid: float
    ask: float
    mid_price: float
    spot_price: float
    ticker: str
    action: str  # "SELL" or "SKIP"
    skip_reason: str = ""


def estimate_iv_from_atm(spot: float, date: str, hour: int, minute: int, T: float,
                          underlying: str = "SPY") -> float:
    """Estimate IV from ATM options for any underlying."""
    r = RISK_FREE_RATE
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

    return sum(ivs) / len(ivs) if ivs else 0.20


def scan_for_trades(date_str: str, underlying: str = "SPY",
                     verbose: bool = True) -> list[TradeSignal]:
    """
    Scan for trade opportunities on a given date for any underlying.

    Args:
        date_str: Date in YYYY-MM-DD format
        underlying: Stock/ETF symbol (default: "SPY")
        verbose: Print detailed output

    Returns list of TradeSignal objects with recommendations.
    """
    signals = []

    # Parse entry times
    entry_times = []
    for t in ENTRY_TIMES:
        h, m = map(int, t.split(":"))
        entry_times.append((h, m, t))

    # Check filters
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRADE SCANNER - {underlying} - {date_str}")
        print(f"{'='*60}")

    # Filter 1: Event day check
    is_event, event_type = is_event_day(date_str)
    if SKIP_EVENT_DAYS and is_event:
        if verbose:
            print(f"\n[SKIP DAY] {event_type} day - No trades recommended")
        return signals

    # Filter 2: Half day check
    if SKIP_HALF_DAYS and is_half_day(date_str):
        if verbose:
            print(f"\n[SKIP DAY] Half trading day - No trades recommended")
        return signals

    # Get underlying data
    bars = get_minute_bars(date_str, underlying)
    if len(bars) < 50:
        if verbose:
            print(f"\n[NO DATA] Insufficient {underlying} data for {date_str}")
        return signals

    # Get IV threshold for this underlying
    iv_threshold = get_iv_threshold(underlying)

    if verbose:
        if iv_threshold:
            print(f"\nFilter: Skip ALL trades when IV > {iv_threshold:.0%}")
        else:
            print(f"\nFilter: No IV filter for {underlying}")
        print(f"Target Delta: {TARGET_DELTA}")

    for hour, minute, time_str in entry_times:
        spot = get_price_at_time(bars, hour, minute)
        if spot is None:
            continue

        hours_to_close = (16 - hour) - minute / 60
        T = hours_to_close / 24 / 365

        # Estimate IV
        iv = estimate_iv_from_atm(spot, date_str, hour, minute, T, underlying)

        if verbose:
            print(f"\n--- {time_str} ET ---")
            print(f"{underlying}: ${spot:.2f} | IV: {iv:.1%} | Hours to close: {hours_to_close:.1f}")

        # PRIMARY FILTER: Skip ALL trades when IV > threshold (if threshold set)
        if iv_threshold and iv > iv_threshold:
            if verbose:
                print(f"  [SKIP] IV {iv:.1%} > {iv_threshold:.0%} threshold - No trades this window")
            continue

        # CALL signal
        call_strike_theory = find_strike_for_delta(TARGET_DELTA, spot, T, RISK_FREE_RATE, iv, "call")
        if call_strike_theory:
            call_strike = round(call_strike_theory)
            ticker = build_option_ticker(date_str, call_strike, "C", underlying)
            option_bars = get_option_minute_bars(ticker, date_str)
            price = get_option_price_at_time(option_bars, hour, minute)

            if price and price > 0.01:
                actual_iv = implied_volatility(price, spot, call_strike, T, RISK_FREE_RATE, "call") or iv
                delta = call_delta(spot, call_strike, T, RISK_FREE_RATE, actual_iv)

                signal = TradeSignal(
                    date=date_str,
                    time=time_str,
                    underlying=underlying,
                    option_type="CALL",
                    strike=call_strike,
                    delta=delta,
                    iv=actual_iv,
                    bid=price * 0.95,  # Estimate
                    ask=price * 1.05,
                    mid_price=price,
                    spot_price=spot,
                    ticker=ticker,
                    action="SELL"
                )
                signals.append(signal)

                if verbose:
                    otm_pct = (call_strike - spot) / spot * 100
                    print(f"  CALL {call_strike} ({otm_pct:+.1f}% OTM): "
                          f"delta={delta:.2f}, IV={actual_iv:.0%}, price=${price:.2f} -> SELL")

        # PUT signal
        put_strike_theory = find_strike_for_delta(-TARGET_DELTA, spot, T, RISK_FREE_RATE, iv, "put")
        if put_strike_theory:
            put_strike = round(put_strike_theory)
            ticker = build_option_ticker(date_str, put_strike, "P", underlying)
            option_bars = get_option_minute_bars(ticker, date_str)
            price = get_option_price_at_time(option_bars, hour, minute)

            if price and price > 0.01:
                actual_iv = implied_volatility(price, spot, put_strike, T, RISK_FREE_RATE, "put") or iv
                delta = put_delta(spot, put_strike, T, RISK_FREE_RATE, actual_iv)

                signal = TradeSignal(
                    date=date_str,
                    time=time_str,
                    underlying=underlying,
                    option_type="PUT",
                    strike=put_strike,
                    delta=abs(delta),
                    iv=actual_iv,
                    bid=price * 0.95,
                    ask=price * 1.05,
                    mid_price=price,
                    spot_price=spot,
                    ticker=ticker,
                    action="SELL"
                )
                signals.append(signal)

                if verbose:
                    otm_pct = (spot - put_strike) / spot * 100
                    print(f"  PUT  {put_strike} ({otm_pct:+.1f}% OTM): "
                          f"delta={abs(delta):.2f}, IV={actual_iv:.0%}, price=${price:.2f} -> SELL")

    return signals


def print_trade_summary(signals: list[TradeSignal]):
    """Print a summary of trade signals."""
    if not signals:
        print("\nNo trade signals generated.")
        return

    sell_signals = [s for s in signals if s.action == "SELL"]
    skip_signals = [s for s in signals if s.action == "SKIP"]

    print(f"\n{'='*60}")
    print("TRADE SUMMARY")
    print(f"{'='*60}")
    print(f"Total Signals: {len(signals)}")
    print(f"  SELL: {len(sell_signals)}")
    print(f"  SKIP: {len(skip_signals)}")

    if sell_signals:
        print(f"\n--- TRADES TO EXECUTE ---")
        total_premium = 0
        for s in sell_signals:
            print(f"  SELL {s.ticker} @ ${s.mid_price:.2f} (delta={s.delta:.2f})")
            total_premium += s.mid_price * 100
        print(f"\n  Total Premium (1 contract each): ${total_premium:.2f}")

    if skip_signals:
        print(f"\n--- SKIPPED (filtered) ---")
        for s in skip_signals:
            print(f"  {s.ticker}: {s.skip_reason}")


def save_signals_to_log(signals: list[TradeSignal], filename: str = "trade_log.json"):
    """Append signals to trade log file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "signals": [asdict(s) for s in signals]
    }

    # Load existing log or create new
    log_data = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = []

    log_data.append(log_entry)

    with open(filename, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"\nSignals saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="0DTE Options Trade Scanner with Risk Filters"
    )
    parser.add_argument(
        "--date", "-d",
        help="Date to scan (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--underlying", "-u",
        default="SPY",
        help="Underlying symbol to scan (e.g., SPY, QQQ, AAPL). Default: SPY"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save signals to trade log"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="Disable all filters (for comparison)"
    )

    args = parser.parse_args()

    # Determine date
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Temporarily disable filters if requested
    if args.no_filters:
        import config
        config.SKIP_EVENT_DAYS = False
        config.SKIP_PUTS_HIGH_IV = False
        config.SKIP_HALF_DAYS = False
        config.MAX_IV_THRESHOLD = None

    # Run scanner
    signals = scan_for_trades(date_str, underlying=args.underlying.upper(),
                               verbose=not args.quiet)

    # Print summary
    if not args.quiet:
        print_trade_summary(signals)

    # Save if requested
    if args.save:
        save_signals_to_log(signals)

    return 0


if __name__ == "__main__":
    exit(main())
