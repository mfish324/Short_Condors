"""
Fast backtest engine for 0DTE options strategy.
Optimized for fewer API calls by:
1. Estimating IV from ATM option prices
2. Calculating theoretical 10-delta strikes
3. Only fetching specific strike data
"""
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
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
    total_trades: int
    otm_count: int
    itm_count: int
    otm_probability: float
    avg_premium_collected: float
    total_pnl: float
    win_rate: float
    avg_delta: float
    trades: list


def estimate_iv_from_atm(spot: float, date: str, hour: int, minute: int, T: float) -> float:
    """
    Estimate IV by looking at ATM straddle price.
    This requires only 2 API calls instead of 20+.
    """
    r = 0.05
    atm_strike = round(spot)

    # Get ATM call and put prices
    call_ticker = build_option_ticker(date, atm_strike, "C")
    put_ticker = build_option_ticker(date, atm_strike, "P")

    call_bars = get_option_minute_bars(call_ticker, date)
    put_bars = get_option_minute_bars(put_ticker, date)

    call_price = get_option_price_at_time(call_bars, hour, minute) if call_bars else None
    put_price = get_option_price_at_time(put_bars, hour, minute) if put_bars else None

    # Calculate IV from whichever we have
    ivs = []
    if call_price and call_price > 0.05:
        iv = implied_volatility(call_price, spot, atm_strike, T, r, "call")
        if iv and 0.05 < iv < 2.0:
            ivs.append(iv)

    if put_price and put_price > 0.05:
        iv = implied_volatility(put_price, spot, atm_strike, T, r, "put")
        if iv and 0.05 < iv < 2.0:
            ivs.append(iv)

    if ivs:
        return sum(ivs) / len(ivs)

    # Fallback: typical 0DTE IV
    return 0.20


def run_backtest(
    start_date: str,
    end_date: str,
    entry_times: list[tuple[int, int]] = [(10, 0), (10, 30), (11, 0)],
    target_delta: float = 0.10,
    verbose: bool = True
) -> BacktestResult:
    """
    Fast backtest using estimated IV and theoretical delta strikes.
    """
    trades = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    r = 0.05

    days_processed = 0

    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        date_str = current.strftime("%Y-%m-%d")
        spy_bars = get_spy_minute_bars(date_str)

        if len(spy_bars) < 100:
            current += timedelta(days=1)
            continue

        closing_price = spy_bars[-1]["c"]
        if verbose:
            print(f"{date_str} | Close: ${closing_price:.2f}", end="")

        day_trades = []

        for hour, minute in entry_times:
            spot = get_spy_price_at_time(spy_bars, hour, minute)
            if spot is None:
                continue

            hours_to_close = (16 - hour) - minute / 60
            T = hours_to_close / 24 / 365

            # Estimate IV from ATM options (fast: only 2 API calls)
            iv = estimate_iv_from_atm(spot, date_str, hour, minute, T)

            # Calculate theoretical 10-delta strikes
            call_strike_theory = find_strike_for_delta(target_delta, spot, T, r, iv, "call")
            put_strike_theory = find_strike_for_delta(-target_delta, spot, T, r, iv, "put")

            if call_strike_theory:
                call_strike = round(call_strike_theory)

                # Get actual price at this strike
                ticker = build_option_ticker(date_str, call_strike, "C")
                bars = get_option_minute_bars(ticker, date_str)
                price = get_option_price_at_time(bars, hour, minute)

                if price and price > 0.01:
                    # Calculate actual delta
                    actual_iv = implied_volatility(price, spot, call_strike, T, r, "call")
                    if actual_iv is None:
                        actual_iv = iv
                    delta = call_delta(spot, call_strike, T, r, actual_iv)

                    expired_otm = closing_price < call_strike
                    if expired_otm:
                        pnl = price * 100
                    else:
                        pnl = (price - (closing_price - call_strike)) * 100

                    day_trades.append(TradeEntry(
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

            if put_strike_theory:
                put_strike = round(put_strike_theory)

                ticker = build_option_ticker(date_str, put_strike, "P")
                bars = get_option_minute_bars(ticker, date_str)
                price = get_option_price_at_time(bars, hour, minute)

                if price and price > 0.01:
                    actual_iv = implied_volatility(price, spot, put_strike, T, r, "put")
                    if actual_iv is None:
                        actual_iv = iv
                    delta = put_delta(spot, put_strike, T, r, actual_iv)

                    expired_otm = closing_price > put_strike
                    if expired_otm:
                        pnl = price * 100
                    else:
                        pnl = (price - (put_strike - closing_price)) * 100

                    day_trades.append(TradeEntry(
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

        # Daily summary
        if day_trades:
            otm = sum(1 for t in day_trades if t.expired_otm)
            if verbose:
                print(f" | {otm}/{len(day_trades)} OTM")
            trades.extend(day_trades)
        else:
            if verbose:
                print(" | No trades")

        days_processed += 1
        current += timedelta(days=1)
        time.sleep(0.2)  # Rate limiting

    # Calculate statistics
    total = len(trades)
    if total == 0:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, [])

    otm_count = sum(1 for t in trades if t.expired_otm)
    avg_delta = sum(t.delta for t in trades) / total

    return BacktestResult(
        total_trades=total,
        otm_count=otm_count,
        itm_count=total - otm_count,
        otm_probability=otm_count / total,
        avg_premium_collected=sum(t.option_price for t in trades) / total * 100,
        total_pnl=sum(t.pnl_if_sold for t in trades),
        win_rate=otm_count / total,
        avg_delta=avg_delta,
        trades=[asdict(t) for t in trades]
    )


def print_results(result: BacktestResult):
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

    print(f"\n--- BY ENTRY TIME ---")
    for time_str in ["10:00", "10:30", "11:00"]:
        time_trades = [t for t in result.trades if t["entry_time"] == time_str]
        if time_trades:
            otm = sum(1 for t in time_trades if t["expired_otm"])
            pnl = sum(t["pnl_if_sold"] for t in time_trades)
            print(f"  {time_str}: {otm}/{len(time_trades)} OTM ({otm/len(time_trades):.1%}) | P&L: ${pnl:,.2f}")


if __name__ == "__main__":
    print("Fast 0DTE Options Backtest")
    print("=" * 60)
    print("Running 1-month test: December 2024\n")

    result = run_backtest(
        start_date="2024-12-01",
        end_date="2024-12-31",
        entry_times=[(10, 0), (10, 30), (11, 0)],
        target_delta=0.10,
        verbose=True
    )

    print_results(result)

    with open("backtest_fast_results.json", "w") as f:
        json.dump(asdict(result), f, indent=2)
