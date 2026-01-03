"""
Backtest with risk filters to improve 0DTE strategy performance.
"""
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Callable
import time

from data_fetcher import (
    get_spy_minute_bars,
    get_spy_price_at_time,
    get_option_minute_bars,
    get_option_price_at_time,
    build_option_ticker,
)
from black_scholes import (
    call_delta,
    put_delta,
    implied_volatility,
    find_strike_for_delta,
)


# Known event dates in 2024 (FOMC, CPI, Jobs Reports)
FOMC_DATES_2024 = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18"
]

CPI_DATES_2024 = [
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11"
]

JOBS_REPORT_DATES_2024 = [
    "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05",
    "2024-05-03", "2024-06-07", "2024-07-05", "2024-08-02",
    "2024-09-06", "2024-10-04", "2024-11-01", "2024-12-06"
]

# Half trading days (early close at 1pm ET)
HALF_DAYS_2024 = [
    "2024-07-03",  # Day before July 4th
    "2024-11-29",  # Day after Thanksgiving
    "2024-12-24",  # Christmas Eve
]

EVENT_DATES = set(FOMC_DATES_2024 + CPI_DATES_2024 + JOBS_REPORT_DATES_2024)


@dataclass
class TradeEntry:
    date: str
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
    filtered_reason: str = ""


@dataclass
class FilterResult:
    filter_name: str
    total_trades: int
    trades_filtered: int
    otm_count: int
    itm_count: int
    otm_probability: float
    total_pnl: float
    avg_premium: float
    trades: list


def estimate_iv_from_atm(spot: float, date: str, hour: int, minute: int, T: float) -> float:
    """Estimate IV from ATM options."""
    r = 0.05
    atm_strike = round(spot)

    call_ticker = build_option_ticker(date, atm_strike, "C")
    put_ticker = build_option_ticker(date, atm_strike, "P")

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


def run_filtered_backtest(
    start_date: str,
    end_date: str,
    entry_times: list[tuple[int, int]] = [(10, 0), (10, 30), (11, 0)],
    target_delta: float = 0.10,
    # Filters
    skip_event_days: bool = False,
    skip_half_days: bool = False,
    max_iv: float = None,  # Skip if IV > this
    iv_adjusted_delta: bool = False,  # Use 5 delta when IV > 20%
    skip_puts_high_iv: bool = False,  # Skip puts when IV > 20%
    verbose: bool = False
) -> FilterResult:
    """Run backtest with specified filters."""

    trades = []
    filtered_count = 0

    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    r = 0.05

    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        date_str = current.strftime("%Y-%m-%d")

        # Filter 1: Skip event days
        if skip_event_days and date_str in EVENT_DATES:
            filtered_count += 6  # 3 times x 2 options
            current += timedelta(days=1)
            continue

        # Filter 2: Skip half days
        if skip_half_days and date_str in HALF_DAYS_2024:
            filtered_count += 6
            current += timedelta(days=1)
            continue

        spy_bars = get_spy_minute_bars(date_str)
        if len(spy_bars) < 100:
            current += timedelta(days=1)
            continue

        closing_price = spy_bars[-1]["c"]

        for hour, minute in entry_times:
            spot = get_spy_price_at_time(spy_bars, hour, minute)
            if spot is None:
                continue

            hours_to_close = (16 - hour) - minute / 60
            T = hours_to_close / 24 / 365

            # Get IV estimate
            iv = estimate_iv_from_atm(spot, date_str, hour, minute, T)

            # Filter 3: Skip if IV too high
            if max_iv and iv > max_iv:
                filtered_count += 2
                continue

            # Determine delta to use
            use_delta = target_delta
            if iv_adjusted_delta and iv > 0.20:
                use_delta = 0.05  # Use 5 delta when IV elevated

            # Process CALL
            call_strike_theory = find_strike_for_delta(use_delta, spot, T, r, iv, "call")
            if call_strike_theory:
                call_strike = round(call_strike_theory)
                ticker = build_option_ticker(date_str, call_strike, "C")
                bars = get_option_minute_bars(ticker, date_str)
                price = get_option_price_at_time(bars, hour, minute)

                if price and price > 0.01:
                    actual_iv = implied_volatility(price, spot, call_strike, T, r, "call") or iv
                    delta = call_delta(spot, call_strike, T, r, actual_iv)

                    expired_otm = closing_price < call_strike
                    pnl = price * 100 if expired_otm else (price - (closing_price - call_strike)) * 100

                    trades.append(TradeEntry(
                        date=date_str,
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

            # Filter 5: Skip puts on high IV days
            if skip_puts_high_iv and iv > 0.20:
                filtered_count += 1
                continue

            # Process PUT
            put_strike_theory = find_strike_for_delta(-use_delta, spot, T, r, iv, "put")
            if put_strike_theory:
                put_strike = round(put_strike_theory)
                ticker = build_option_ticker(date_str, put_strike, "P")
                bars = get_option_minute_bars(ticker, date_str)
                price = get_option_price_at_time(bars, hour, minute)

                if price and price > 0.01:
                    actual_iv = implied_volatility(price, spot, put_strike, T, r, "put") or iv
                    delta = put_delta(spot, put_strike, T, r, actual_iv)

                    expired_otm = closing_price > put_strike
                    pnl = price * 100 if expired_otm else (price - (put_strike - closing_price)) * 100

                    trades.append(TradeEntry(
                        date=date_str,
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

        current += timedelta(days=1)
        time.sleep(0.2)

    # Calculate stats
    total = len(trades)
    if total == 0:
        return FilterResult("", 0, filtered_count, 0, 0, 0, 0, 0, [])

    otm_count = sum(1 for t in trades if t.expired_otm)

    return FilterResult(
        filter_name="",
        total_trades=total,
        trades_filtered=filtered_count,
        otm_count=otm_count,
        itm_count=total - otm_count,
        otm_probability=otm_count / total,
        total_pnl=sum(t.pnl_if_sold for t in trades),
        avg_premium=sum(t.option_price for t in trades) / total * 100,
        trades=[asdict(t) for t in trades]
    )


def print_comparison(results: list[FilterResult]):
    """Print comparison table of filter results."""
    print("\n" + "=" * 90)
    print("FILTER COMPARISON RESULTS")
    print("=" * 90)
    print(f"{'Filter':<35} {'Trades':<8} {'Filtered':<10} {'OTM %':<8} {'P&L':<12} {'Avg Prem':<10}")
    print("-" * 90)

    for r in results:
        print(f"{r.filter_name:<35} {r.total_trades:<8} {r.trades_filtered:<10} "
              f"{r.otm_probability*100:>5.1f}%   ${r.total_pnl:>9,.0f}  ${r.avg_premium:>7.2f}")


if __name__ == "__main__":
    print("Running Filter Backtests...")
    print("Date range: July - December 2024 (6 months)")
    print("=" * 70)

    START = "2024-07-01"
    END = "2024-12-31"

    results = []

    # Baseline (no filters)
    print("\n[1/7] Running BASELINE (no filters)...")
    r = run_filtered_backtest(START, END)
    r.filter_name = "BASELINE (no filters)"
    results.append(r)
    print(f"  Trades: {r.total_trades}, OTM: {r.otm_probability:.1%}, P&L: ${r.total_pnl:,.0f}")

    # Filter 1: Skip event days
    print("\n[2/7] Running SKIP EVENT DAYS (FOMC/CPI/Jobs)...")
    r = run_filtered_backtest(START, END, skip_event_days=True)
    r.filter_name = "Skip FOMC/CPI/Jobs days"
    results.append(r)
    print(f"  Trades: {r.total_trades}, OTM: {r.otm_probability:.1%}, P&L: ${r.total_pnl:,.0f}")

    # Filter 2: Skip half days
    print("\n[3/7] Running SKIP HALF DAYS...")
    r = run_filtered_backtest(START, END, skip_half_days=True)
    r.filter_name = "Skip half-days (holidays)"
    results.append(r)
    print(f"  Trades: {r.total_trades}, OTM: {r.otm_probability:.1%}, P&L: ${r.total_pnl:,.0f}")

    # Filter 3: Max IV 25%
    print("\n[4/7] Running MAX IV 25%...")
    r = run_filtered_backtest(START, END, max_iv=0.25)
    r.filter_name = "Skip when IV > 25%"
    results.append(r)
    print(f"  Trades: {r.total_trades}, OTM: {r.otm_probability:.1%}, P&L: ${r.total_pnl:,.0f}")

    # Filter 4: IV-adjusted delta
    print("\n[5/7] Running IV-ADJUSTED DELTA (5d when IV>20%)...")
    r = run_filtered_backtest(START, END, iv_adjusted_delta=True)
    r.filter_name = "Use 5-delta when IV > 20%"
    results.append(r)
    print(f"  Trades: {r.total_trades}, OTM: {r.otm_probability:.1%}, P&L: ${r.total_pnl:,.0f}")

    # Filter 5: Skip puts on high IV
    print("\n[6/7] Running SKIP PUTS WHEN IV > 20%...")
    r = run_filtered_backtest(START, END, skip_puts_high_iv=True)
    r.filter_name = "Skip puts when IV > 20%"
    results.append(r)
    print(f"  Trades: {r.total_trades}, OTM: {r.otm_probability:.1%}, P&L: ${r.total_pnl:,.0f}")

    # Combined best filters
    print("\n[7/7] Running COMBINED FILTERS...")
    r = run_filtered_backtest(
        START, END,
        skip_event_days=True,
        skip_half_days=True,
        skip_puts_high_iv=True
    )
    r.filter_name = "COMBINED: Events+Half+NoPutsHighIV"
    results.append(r)
    print(f"  Trades: {r.total_trades}, OTM: {r.otm_probability:.1%}, P&L: ${r.total_pnl:,.0f}")

    # Print comparison
    print_comparison(results)

    # Save results
    output = {
        'date_range': f'{START} to {END}',
        'filters': [
            {
                'name': r.filter_name,
                'trades': r.total_trades,
                'filtered': r.trades_filtered,
                'otm_pct': r.otm_probability,
                'pnl': r.total_pnl,
                'avg_premium': r.avg_premium
            }
            for r in results
        ]
    }

    with open('filter_comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to filter_comparison_results.json")
