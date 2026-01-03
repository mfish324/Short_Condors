#!/usr/bin/env python
"""
Trade Tracker - Log and track 0DTE option trades

Usage:
    python trade_tracker.py add --ticker O:SPY250102C00600000 --price 0.15
    python trade_tracker.py close --id 1 --price 0.00
    python trade_tracker.py status
    python trade_tracker.py history
"""
import argparse
import json
import os
import re
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional


TRADE_FILE = "trades.json"


@dataclass
class Trade:
    id: int
    date: str
    time: str
    ticker: str
    option_type: str  # CALL or PUT
    strike: float
    entry_price: float
    exit_price: Optional[float]
    status: str  # OPEN, CLOSED, EXPIRED
    pnl: Optional[float]
    notes: str = ""
    alpaca_order_id: str = ""  # Alpaca order ID for auto-trades
    source: str = "manual"  # "manual", "backtest", or "alpaca"
    underlying: str = "SPY"  # Underlying symbol (SPY, QQQ, AAPL, etc.)

    @classmethod
    def from_dict(cls, d):
        # Handle old trades without 'underlying' field
        if "underlying" not in d:
            d["underlying"] = "SPY"
        return cls(**d)


def load_trades() -> list[Trade]:
    """Load trades from file."""
    if not os.path.exists(TRADE_FILE):
        return []
    with open(TRADE_FILE, "r") as f:
        data = json.load(f)
    return [Trade.from_dict(t) for t in data]


def save_trades(trades: list[Trade]):
    """Save trades to file."""
    with open(TRADE_FILE, "w") as f:
        json.dump([asdict(t) for t in trades], f, indent=2)


def parse_ticker(ticker: str) -> tuple[str, str, float]:
    """Parse option ticker to extract underlying, type, and strike.

    Supports both formats:
    - Polygon: O:SPY250102C00600000, O:AAPL250102C00200000
    - Alpaca: SPY250102C00600000, AAPL250102C00200000

    Returns:
        Tuple of (underlying, option_type, strike)
        e.g., ("SPY", "CALL", 600.0) or ("AAPL", "PUT", 200.0)
    """
    # Remove O: prefix if present (Polygon format)
    if ticker.startswith("O:"):
        code = ticker[2:]
    else:
        code = ticker

    # Use regex to parse: SYMBOL(1-5 chars) + DATE(6 digits) + TYPE(C/P) + STRIKE(8 digits)
    # Examples: SPY241220C00600000, AAPL241220P00200000, GOOGL241220C00150000
    match = re.match(r'^([A-Z]{1,5})(\d{6})([CP])(\d{8})$', code)

    if match:
        underlying = match.group(1)
        option_char = match.group(3)
        strike_str = match.group(4)

        option_type = "CALL" if option_char == "C" else "PUT"
        strike = int(strike_str) / 1000

        return underlying, option_type, strike

    return "UNKNOWN", "UNKNOWN", 0.0


def add_trade(ticker: str, price: float, notes: str = "",
               alpaca_order_id: str = "", source: str = "manual") -> Trade:
    """Add a new trade.

    Args:
        ticker: Option ticker (Polygon or Alpaca format)
        price: Entry price
        notes: Optional notes
        alpaca_order_id: Alpaca order ID for auto-trades
        source: Trade source - "manual", "backtest", or "alpaca"
    """
    trades = load_trades()

    # Generate ID
    new_id = max([t.id for t in trades], default=0) + 1

    underlying, opt_type, strike = parse_ticker(ticker)

    trade = Trade(
        id=new_id,
        date=datetime.now().strftime("%Y-%m-%d"),
        time=datetime.now().strftime("%H:%M"),
        ticker=ticker,
        option_type=opt_type,
        strike=strike,
        entry_price=price,
        exit_price=None,
        status="OPEN",
        pnl=None,
        notes=notes,
        alpaca_order_id=alpaca_order_id,
        source=source,
        underlying=underlying
    )

    trades.append(trade)
    save_trades(trades)

    print(f"Trade #{new_id} added: SELL {underlying} {opt_type} {strike} @ ${price:.2f}")
    return trade


def close_trade(trade_id: int, exit_price: float, status: str = "CLOSED"):
    """Close an existing trade."""
    trades = load_trades()

    for t in trades:
        if t.id == trade_id:
            t.exit_price = exit_price
            t.status = status
            # P&L = (entry - exit) * 100 for short options
            t.pnl = (t.entry_price - exit_price) * 100

            save_trades(trades)

            result = "WIN" if t.pnl > 0 else "LOSS"
            print(f"Trade #{trade_id} closed: {result} ${t.pnl:.2f}")
            return t

    print(f"Trade #{trade_id} not found")
    return None


def expire_trade(trade_id: int):
    """Mark trade as expired worthless (full profit)."""
    return close_trade(trade_id, 0.0, "EXPIRED")


def show_status():
    """Show current open positions and daily P&L."""
    trades = load_trades()

    open_trades = [t for t in trades if t.status == "OPEN"]
    today = datetime.now().strftime("%Y-%m-%d")
    today_trades = [t for t in trades if t.date == today]
    closed_today = [t for t in today_trades if t.status != "OPEN"]

    print("\n" + "=" * 60)
    print("TRADE STATUS")
    print("=" * 60)

    print(f"\n--- OPEN POSITIONS ({len(open_trades)}) ---")
    if open_trades:
        total_premium = 0
        for t in open_trades:
            print(f"  #{t.id}: {t.ticker} @ ${t.entry_price:.2f}")
            total_premium += t.entry_price * 100
        print(f"\n  Total Premium at Risk: ${total_premium:.2f}")
    else:
        print("  No open positions")

    print(f"\n--- TODAY'S ACTIVITY ({today}) ---")
    print(f"  Trades opened: {len([t for t in today_trades if t.status == 'OPEN'])}")
    print(f"  Trades closed: {len(closed_today)}")

    if closed_today:
        today_pnl = sum(t.pnl or 0 for t in closed_today)
        wins = len([t for t in closed_today if (t.pnl or 0) > 0])
        print(f"  Today's P&L: ${today_pnl:.2f}")
        print(f"  Win Rate: {wins}/{len(closed_today)}")


def show_history(days: int = 30):
    """Show trade history and statistics."""
    trades = load_trades()
    closed = [t for t in trades if t.status != "OPEN"]

    print("\n" + "=" * 60)
    print("TRADE HISTORY")
    print("=" * 60)

    if not closed:
        print("\nNo closed trades yet.")
        return

    # Overall stats
    total_pnl = sum(t.pnl or 0 for t in closed)
    wins = len([t for t in closed if (t.pnl or 0) > 0])
    losses = len([t for t in closed if (t.pnl or 0) <= 0])

    print(f"\n--- OVERALL STATISTICS ---")
    print(f"  Total Trades: {len(closed)}")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {wins/len(closed)*100:.1f}%")
    print(f"  Total P&L: ${total_pnl:.2f}")
    print(f"  Avg P&L per Trade: ${total_pnl/len(closed):.2f}")

    # By option type
    calls = [t for t in closed if t.option_type == "CALL"]
    puts = [t for t in closed if t.option_type == "PUT"]

    if calls:
        call_pnl = sum(t.pnl or 0 for t in calls)
        call_wins = len([t for t in calls if (t.pnl or 0) > 0])
        print(f"\n  CALLS: {call_wins}/{len(calls)} wins, P&L: ${call_pnl:.2f}")

    if puts:
        put_pnl = sum(t.pnl or 0 for t in puts)
        put_wins = len([t for t in puts if (t.pnl or 0) > 0])
        print(f"  PUTS:  {put_wins}/{len(puts)} wins, P&L: ${put_pnl:.2f}")

    # Recent trades
    print(f"\n--- RECENT TRADES ---")
    for t in sorted(closed, key=lambda x: (x.date, x.time), reverse=True)[:10]:
        result = "WIN " if (t.pnl or 0) > 0 else "LOSS"
        print(f"  {t.date} {t.ticker}: {result} ${t.pnl:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Track 0DTE option trades")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add trade
    add_parser = subparsers.add_parser("add", help="Add a new trade")
    add_parser.add_argument("--ticker", "-t", required=True, help="Option ticker")
    add_parser.add_argument("--price", "-p", type=float, required=True, help="Entry price")
    add_parser.add_argument("--notes", "-n", default="", help="Trade notes")

    # Close trade
    close_parser = subparsers.add_parser("close", help="Close a trade")
    close_parser.add_argument("--id", type=int, required=True, help="Trade ID")
    close_parser.add_argument("--price", "-p", type=float, required=True, help="Exit price")

    # Expire trade
    expire_parser = subparsers.add_parser("expire", help="Mark trade as expired worthless")
    expire_parser.add_argument("--id", type=int, required=True, help="Trade ID")

    # Status
    subparsers.add_parser("status", help="Show current status")

    # History
    history_parser = subparsers.add_parser("history", help="Show trade history")
    history_parser.add_argument("--days", "-d", type=int, default=30, help="Days of history")

    args = parser.parse_args()

    if args.command == "add":
        add_trade(args.ticker, args.price, args.notes)
    elif args.command == "close":
        close_trade(args.id, args.price)
    elif args.command == "expire":
        expire_trade(args.id)
    elif args.command == "status":
        show_status()
    elif args.command == "history":
        show_history(args.days)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
