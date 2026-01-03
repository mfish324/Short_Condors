"""
Backtest engine for 0DTE options strategy.
Calculates probability of 10-delta options expiring OTM.
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
    find_strike_for_delta,
)


@dataclass
class TradeEntry:
    """Represents a single option entry."""
    date: str
    entry_time: str
    spot_price: float
    strike: float
    option_type: str  # "call" or "put"
    delta: float
    iv: float
    option_price: float
    closing_spot: float
    expired_otm: bool
    pnl_if_sold: float  # Premium collected if held to expiration


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
    trades: list


def get_closing_price(spy_bars: list[dict]) -> float | None:
    """Get SPY closing price (last bar of day around 4pm ET)."""
    if not spy_bars:
        return None
    # Find the last bar (should be around 4pm)
    last_bar = spy_bars[-1]
    return last_bar["c"]


def find_10_delta_strike(
    spot: float,
    time_to_expiry_hours: float,
    option_type: str,
    date: str,
    spy_bars: list[dict],
    target_delta: float = 0.10
) -> tuple[float, float, float] | None:
    """
    Find the strike closest to 10 delta for 0DTE option.

    Returns: (strike, actual_delta, iv) or None if not found
    """
    r = 0.05  # Risk-free rate
    T = time_to_expiry_hours / 24 / 365  # Convert to years

    # For 0DTE, we need to estimate IV first
    # Start with an initial IV estimate and refine
    # Typical 0DTE IV ranges from 10% to 50%

    # Search for the best strike in $1 increments
    best_strike = None
    best_delta_diff = float('inf')
    best_iv = 0.20  # Default IV

    # Determine search range based on option type
    if option_type == "call":
        # Calls: look above spot
        strikes = [spot + i for i in range(1, 20)]  # Up to $20 above
        target = target_delta
        delta_func = call_delta
    else:
        # Puts: look below spot
        strikes = [spot - i for i in range(1, 20)]  # Up to $20 below
        target = -target_delta
        delta_func = put_delta

    # For each strike, try to get option price and calculate IV
    for strike in strikes:
        strike = round(strike)  # SPY has $1 strikes

        # Build ticker and get price
        opt_type_code = "C" if option_type == "call" else "P"
        ticker = build_option_ticker(date, strike, opt_type_code)

        # Check if we have price data
        opt_bars = get_option_minute_bars(ticker, date)
        if not opt_bars:
            continue

        # Get option price at entry time
        # We'll use midpoint of the hour we're looking at
        # For simplicity, use first available price
        opt_price = None
        for bar in opt_bars:
            if bar.get("c", 0) > 0:
                opt_price = bar["c"]
                break

        if opt_price is None or opt_price <= 0:
            continue

        # Calculate IV from this option price
        iv = implied_volatility(opt_price, spot, strike, T, r, option_type)
        if iv is None:
            iv = 0.20  # Default

        # Calculate delta with this IV
        delta = delta_func(spot, strike, T, r, iv)

        delta_diff = abs(delta - target)
        if delta_diff < best_delta_diff:
            best_delta_diff = delta_diff
            best_strike = strike
            best_iv = iv

    if best_strike is None:
        return None

    # Get final delta
    final_delta = delta_func(spot, best_strike, T, r, best_iv)

    return (best_strike, final_delta, best_iv)


def find_strike_for_target_delta_fast(
    spot: float,
    time_to_expiry_hours: float,
    option_type: str,
    date: str,
    target_delta: float = 0.10,
    estimated_iv: float = 0.20
) -> tuple[float, float, float]:
    """
    Fast method: Calculate theoretical 10-delta strike using estimated IV.
    Then verify with actual option prices.

    Returns: (strike, estimated_delta, iv)
    """
    r = 0.05
    T = time_to_expiry_hours / 24 / 365

    if option_type == "call":
        target = target_delta
        theoretical_strike = find_strike_for_delta(target, spot, T, r, estimated_iv, "call")
    else:
        target = -target_delta
        theoretical_strike = find_strike_for_delta(target, spot, T, r, estimated_iv, "put")

    if theoretical_strike is None:
        # Fallback: use percentage OTM
        if option_type == "call":
            theoretical_strike = spot * 1.005  # 0.5% OTM
        else:
            theoretical_strike = spot * 0.995  # 0.5% OTM

    # Round to nearest $1 (SPY strikes)
    strike = round(theoretical_strike)

    # Recalculate delta at rounded strike
    if option_type == "call":
        delta = call_delta(spot, strike, T, r, estimated_iv)
    else:
        delta = put_delta(spot, strike, T, r, estimated_iv)

    return (strike, delta, estimated_iv)


def run_backtest(
    start_date: str,
    end_date: str,
    entry_times: list[tuple[int, int]] = [(10, 0), (10, 30), (11, 0)],
    target_delta: float = 0.10,
    estimated_iv: float = 0.20,
    verbose: bool = True
) -> BacktestResult:
    """
    Run backtest for 0DTE option selling strategy.

    Args:
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        entry_times: List of (hour, minute) tuples for entry times (ET)
        target_delta: Target delta for options (default 0.10)
        estimated_iv: Estimated IV for delta calculation
        verbose: Print progress

    Returns:
        BacktestResult with all trades and statistics
    """
    trades = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    days_processed = 0

    while current <= end:
        # Skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        date_str = current.strftime("%Y-%m-%d")

        # Get SPY data for the day
        spy_bars = get_spy_minute_bars(date_str)

        if len(spy_bars) < 100:  # Not a trading day or no data
            current += timedelta(days=1)
            continue

        closing_price = get_closing_price(spy_bars)
        if closing_price is None:
            current += timedelta(days=1)
            continue

        if verbose:
            print(f"\nProcessing {date_str}... SPY close: ${closing_price:.2f}")

        # Process each entry time
        for hour, minute in entry_times:
            spot = get_spy_price_at_time(spy_bars, hour, minute)
            if spot is None:
                continue

            # Calculate time to expiry (market closes at 4pm ET)
            hours_to_close = (16 - hour) + (0 - minute) / 60

            # Find 10-delta call
            call_result = find_strike_for_target_delta_fast(
                spot, hours_to_close, "call", date_str, target_delta, estimated_iv
            )
            call_strike, call_delta_val, call_iv = call_result

            # Get actual call option price
            call_ticker = build_option_ticker(date_str, call_strike, "C")
            call_bars = get_option_minute_bars(call_ticker, date_str)
            call_price = get_option_price_at_time(call_bars, hour, minute)

            if call_price and call_price > 0:
                call_expired_otm = closing_price < call_strike
                call_pnl = call_price * 100 if call_expired_otm else (call_price - (closing_price - call_strike)) * 100

                trades.append(TradeEntry(
                    date=date_str,
                    entry_time=f"{hour}:{minute:02d}",
                    spot_price=spot,
                    strike=call_strike,
                    option_type="call",
                    delta=call_delta_val,
                    iv=call_iv,
                    option_price=call_price,
                    closing_spot=closing_price,
                    expired_otm=call_expired_otm,
                    pnl_if_sold=call_pnl
                ))

                if verbose:
                    status = "OTM (win)" if call_expired_otm else "ITM (loss)"
                    print(f"  {hour}:{minute:02d} CALL @ {call_strike}: spot={spot:.2f}, close={closing_price:.2f} -> {status}")

            # Find 10-delta put
            put_result = find_strike_for_target_delta_fast(
                spot, hours_to_close, "put", date_str, target_delta, estimated_iv
            )
            put_strike, put_delta_val, put_iv = put_result

            # Get actual put option price
            put_ticker = build_option_ticker(date_str, put_strike, "P")
            put_bars = get_option_minute_bars(put_ticker, date_str)
            put_price = get_option_price_at_time(put_bars, hour, minute)

            if put_price and put_price > 0:
                put_expired_otm = closing_price > put_strike
                put_pnl = put_price * 100 if put_expired_otm else (put_price - (put_strike - closing_price)) * 100

                trades.append(TradeEntry(
                    date=date_str,
                    entry_time=f"{hour}:{minute:02d}",
                    spot_price=spot,
                    strike=put_strike,
                    option_type="put",
                    delta=put_delta_val,
                    iv=put_iv,
                    option_price=put_price,
                    closing_spot=closing_price,
                    expired_otm=put_expired_otm,
                    pnl_if_sold=put_pnl
                ))

                if verbose:
                    status = "OTM (win)" if put_expired_otm else "ITM (loss)"
                    print(f"  {hour}:{minute:02d} PUT @ {put_strike}: spot={spot:.2f}, close={closing_price:.2f} -> {status}")

        days_processed += 1
        current += timedelta(days=1)

        # Rate limiting
        time.sleep(0.5)

    # Calculate statistics
    total = len(trades)
    otm_count = sum(1 for t in trades if t.expired_otm)
    itm_count = total - otm_count

    result = BacktestResult(
        total_trades=total,
        otm_count=otm_count,
        itm_count=itm_count,
        otm_probability=otm_count / total if total > 0 else 0,
        avg_premium_collected=sum(t.option_price for t in trades) / total * 100 if total > 0 else 0,
        total_pnl=sum(t.pnl_if_sold for t in trades),
        win_rate=otm_count / total if total > 0 else 0,
        trades=[asdict(t) for t in trades]
    )

    return result


def print_results(result: BacktestResult):
    """Print formatted backtest results."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Trades: {result.total_trades}")
    print(f"Expired OTM (Winners): {result.otm_count}")
    print(f"Expired ITM (Losers): {result.itm_count}")
    print(f"\nOTM Probability: {result.otm_probability:.1%}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"\nAvg Premium Collected: ${result.avg_premium_collected:.2f}")
    print(f"Total P&L: ${result.total_pnl:.2f}")

    # Break down by option type
    calls = [t for t in result.trades if t["option_type"] == "call"]
    puts = [t for t in result.trades if t["option_type"] == "put"]

    if calls:
        call_otm = sum(1 for t in calls if t["expired_otm"])
        print(f"\nCALLS: {call_otm}/{len(calls)} OTM ({call_otm/len(calls):.1%})")

    if puts:
        put_otm = sum(1 for t in puts if t["expired_otm"])
        print(f"PUTS: {put_otm}/{len(puts)} OTM ({put_otm/len(puts):.1%})")

    # Break down by entry time
    print("\nBy Entry Time:")
    for time_str in ["10:00", "10:30", "11:00"]:
        time_trades = [t for t in result.trades if t["entry_time"] == time_str]
        if time_trades:
            otm = sum(1 for t in time_trades if t["expired_otm"])
            print(f"  {time_str}: {otm}/{len(time_trades)} OTM ({otm/len(time_trades):.1%})")


if __name__ == "__main__":
    # Run a small test backtest
    print("Running test backtest...")
    print("Date range: 2024-12-16 to 2024-12-20 (1 week)")

    result = run_backtest(
        start_date="2024-12-16",
        end_date="2024-12-20",
        entry_times=[(10, 0), (10, 30), (11, 0)],
        target_delta=0.10,
        estimated_iv=0.15,
        verbose=True
    )

    print_results(result)

    # Save results
    with open("backtest_results.json", "w") as f:
        json.dump(asdict(result), f, indent=2)
    print("\nResults saved to backtest_results.json")
