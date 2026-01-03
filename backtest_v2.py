"""
Improved backtest engine for 0DTE options strategy.
Uses actual option prices to calculate IV and find true 10-delta options.
"""
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional
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
)


@dataclass
class TradeEntry:
    """Represents a single option entry."""
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


@dataclass
class BacktestResult:
    """Aggregate backtest results."""
    total_trades: int
    otm_count: int
    itm_count: int
    otm_probability: float
    avg_premium_collected: float
    total_pnl: float
    win_rate: float
    avg_delta: float
    trades: list


def get_closing_price(spy_bars: list[dict]) -> float | None:
    """Get SPY closing price."""
    if not spy_bars:
        return None
    return spy_bars[-1]["c"]


def find_option_with_target_delta(
    spot: float,
    time_to_expiry_hours: float,
    option_type: str,
    date: str,
    hour: int,
    minute: int,
    target_delta: float = 0.10,
) -> tuple[float, float, float, float] | None:
    """
    Find option closest to target delta using actual market prices.

    Scans through strikes, calculates IV from prices, then computes delta.

    Returns: (strike, delta, iv, option_price) or None
    """
    r = 0.05
    T = time_to_expiry_hours / 24 / 365

    # Generate strikes to check
    if option_type == "call":
        # For calls, search above spot
        strikes = [round(spot) + i for i in range(1, 25)]
        target = target_delta
        delta_func = call_delta
        opt_code = "C"
    else:
        # For puts, search below spot
        strikes = [round(spot) - i for i in range(1, 25)]
        target = -target_delta
        delta_func = put_delta
        opt_code = "P"

    candidates = []

    for strike in strikes:
        # Get option price
        ticker = build_option_ticker(date, strike, opt_code)
        bars = get_option_minute_bars(ticker, date)

        if not bars:
            continue

        price = get_option_price_at_time(bars, hour, minute)
        if price is None or price <= 0.01:  # Skip very cheap options
            continue

        # Calculate IV from this price
        iv = implied_volatility(price, spot, strike, T, r, option_type)
        if iv is None or iv < 0.05 or iv > 3.0:  # Sanity check IV
            continue

        # Calculate delta with this IV
        delta = delta_func(spot, strike, T, r, iv)

        candidates.append({
            'strike': strike,
            'delta': delta,
            'iv': iv,
            'price': price,
            'delta_diff': abs(delta - target)
        })

    if not candidates:
        return None

    # Find closest to target delta
    best = min(candidates, key=lambda x: x['delta_diff'])

    return (best['strike'], best['delta'], best['iv'], best['price'])


def run_backtest(
    start_date: str,
    end_date: str,
    entry_times: list[tuple[int, int]] = [(10, 0), (10, 30), (11, 0)],
    target_delta: float = 0.10,
    verbose: bool = True
) -> BacktestResult:
    """
    Run backtest using actual market IV to find 10-delta options.
    """
    trades = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        date_str = current.strftime("%Y-%m-%d")
        spy_bars = get_spy_minute_bars(date_str)

        if len(spy_bars) < 100:
            current += timedelta(days=1)
            continue

        closing_price = get_closing_price(spy_bars)
        if closing_price is None:
            current += timedelta(days=1)
            continue

        if verbose:
            print(f"\n{date_str} | SPY close: ${closing_price:.2f}")

        for hour, minute in entry_times:
            spot = get_spy_price_at_time(spy_bars, hour, minute)
            if spot is None:
                continue

            hours_to_close = (16 - hour) - minute / 60

            # Find 10-delta call
            call_result = find_option_with_target_delta(
                spot, hours_to_close, "call", date_str, hour, minute, target_delta
            )

            if call_result:
                strike, delta, iv, price = call_result
                expired_otm = closing_price < strike

                if expired_otm:
                    pnl = price * 100  # Keep full premium
                else:
                    intrinsic = closing_price - strike
                    pnl = (price - intrinsic) * 100

                trades.append(TradeEntry(
                    date=date_str,
                    entry_time=f"{hour}:{minute:02d}",
                    spot_price=spot,
                    strike=strike,
                    option_type="call",
                    delta=delta,
                    iv=iv,
                    option_price=price,
                    closing_spot=closing_price,
                    expired_otm=expired_otm,
                    pnl_if_sold=pnl
                ))

                if verbose:
                    otm_pct = (strike - spot) / spot * 100
                    result = "OTM" if expired_otm else "ITM"
                    print(f"  {hour}:{minute:02d} CALL {strike} ({otm_pct:+.1f}%) d={delta:.2f} IV={iv:.0%} -> {result}")

            # Find 10-delta put
            put_result = find_option_with_target_delta(
                spot, hours_to_close, "put", date_str, hour, minute, target_delta
            )

            if put_result:
                strike, delta, iv, price = put_result
                expired_otm = closing_price > strike

                if expired_otm:
                    pnl = price * 100
                else:
                    intrinsic = strike - closing_price
                    pnl = (price - intrinsic) * 100

                trades.append(TradeEntry(
                    date=date_str,
                    entry_time=f"{hour}:{minute:02d}",
                    spot_price=spot,
                    strike=strike,
                    option_type="put",
                    delta=abs(delta),
                    iv=iv,
                    option_price=price,
                    closing_spot=closing_price,
                    expired_otm=expired_otm,
                    pnl_if_sold=pnl
                ))

                if verbose:
                    otm_pct = (spot - strike) / spot * 100
                    result = "OTM" if expired_otm else "ITM"
                    print(f"  {hour}:{minute:02d} PUT  {strike} ({otm_pct:+.1f}%) d={delta:.2f} IV={iv:.0%} -> {result}")

        current += timedelta(days=1)
        time.sleep(0.3)

    # Calculate statistics
    total = len(trades)
    if total == 0:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, [])

    otm_count = sum(1 for t in trades if t.expired_otm)
    itm_count = total - otm_count
    avg_delta = sum(t.delta for t in trades) / total

    return BacktestResult(
        total_trades=total,
        otm_count=otm_count,
        itm_count=itm_count,
        otm_probability=otm_count / total,
        avg_premium_collected=sum(t.option_price for t in trades) / total * 100,
        total_pnl=sum(t.pnl_if_sold for t in trades),
        win_rate=otm_count / total,
        avg_delta=avg_delta,
        trades=[asdict(t) for t in trades]
    )


def print_results(result: BacktestResult):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS - 10 Delta 0DTE Options")
    print("=" * 60)

    print(f"\nTotal Trades: {result.total_trades}")
    print(f"Average Delta: {result.avg_delta:.2f}")

    print(f"\n--- PROBABILITY ANALYSIS ---")
    print(f"Expired OTM (Winners): {result.otm_count}")
    print(f"Expired ITM (Losers): {result.itm_count}")
    print(f"OTM PROBABILITY: {result.otm_probability:.1%}")

    print(f"\n--- P&L ANALYSIS ---")
    print(f"Avg Premium Collected: ${result.avg_premium_collected:.2f}")
    print(f"Total P&L: ${result.total_pnl:,.2f}")

    # Breakdown by type
    calls = [t for t in result.trades if t["option_type"] == "call"]
    puts = [t for t in result.trades if t["option_type"] == "put"]

    print(f"\n--- BY OPTION TYPE ---")
    if calls:
        call_otm = sum(1 for t in calls if t["expired_otm"])
        call_pnl = sum(t["pnl_if_sold"] for t in calls)
        print(f"CALLS: {call_otm}/{len(calls)} OTM ({call_otm/len(calls):.1%}) | P&L: ${call_pnl:,.2f}")

    if puts:
        put_otm = sum(1 for t in puts if t["expired_otm"])
        put_pnl = sum(t["pnl_if_sold"] for t in puts)
        print(f"PUTS:  {put_otm}/{len(puts)} OTM ({put_otm/len(puts):.1%}) | P&L: ${put_pnl:,.2f}")

    # Breakdown by entry time
    print(f"\n--- BY ENTRY TIME ---")
    for time_str in ["10:00", "10:30", "11:00"]:
        time_trades = [t for t in result.trades if t["entry_time"] == time_str]
        if time_trades:
            otm = sum(1 for t in time_trades if t["expired_otm"])
            pnl = sum(t["pnl_if_sold"] for t in time_trades)
            print(f"  {time_str}: {otm}/{len(time_trades)} OTM ({otm/len(time_trades):.1%}) | P&L: ${pnl:,.2f}")


if __name__ == "__main__":
    print("0DTE Options Backtest - Using Actual Market IV")
    print("=" * 60)
    print("Testing: 10-delta calls and puts")
    print("Entry times: 10:00, 10:30, 11:00 ET")
    print("Date range: 2024-12-16 to 2024-12-20")

    result = run_backtest(
        start_date="2024-12-16",
        end_date="2024-12-20",
        entry_times=[(10, 0), (10, 30), (11, 0)],
        target_delta=0.10,
        verbose=True
    )

    print_results(result)

    with open("backtest_v2_results.json", "w") as f:
        json.dump(asdict(result), f, indent=2)
    print("\nResults saved to backtest_v2_results.json")
