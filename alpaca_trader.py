#!/usr/bin/env python
"""
Alpaca Paper Trading Module for 0DTE Options Strategy

Automated daemon that:
1. Runs at scheduled times (10:00, 10:30, 11:00 ET)
2. Gets live SPY price and option quotes from Alpaca
3. Calculates IV from ATM options
4. Applies IV filter (skip if > 20%)
5. Finds 10-delta strikes using Black-Scholes
6. Submits limit sell orders at bid price
7. Logs all trades to trade_tracker

Usage:
    python alpaca_trader.py           # Run daemon
    python alpaca_trader.py --once    # Single execution (for testing)
    python alpaca_trader.py --status  # Check account and positions
"""
import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, time as dt_time
from dataclasses import dataclass
from typing import Optional, Tuple

import pytz
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, OptionLatestQuoteRequest

from config import (
    TARGET_DELTA, ENTRY_TIMES, MAX_IV_THRESHOLD,
    RISK_FREE_RATE, MAX_POSITION_SIZE, ORDER_TYPE, USE_BID_PRICE,
    UNDERLYINGS, get_underlyings_for_day, get_iv_threshold
)
from black_scholes import (
    implied_volatility, find_strike_for_delta,
    call_delta, put_delta
)
from trade_tracker import add_trade

load_dotenv()

# Eastern timezone
ET = pytz.timezone("America/New_York")


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str
    secret_key: str
    paper: bool = True


def load_alpaca_config() -> AlpacaConfig:
    """Load Alpaca credentials from environment."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"

    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env\n"
            "Get your keys at: https://app.alpaca.markets/paper/dashboard/overview"
        )

    if api_key == "your_paper_api_key_here":
        raise ValueError(
            "Please update .env with your actual Alpaca API credentials.\n"
            "Get your keys at: https://app.alpaca.markets/paper/dashboard/overview"
        )

    return AlpacaConfig(api_key=api_key, secret_key=secret_key, paper=paper)


class AlpacaClientManager:
    """Manages Alpaca API clients."""

    def __init__(self, config: AlpacaConfig):
        self.config = config
        self.trading_client: Optional[TradingClient] = None
        self.stock_client: Optional[StockHistoricalDataClient] = None
        self.option_client: Optional[OptionHistoricalDataClient] = None

    def connect(self):
        """Initialize all Alpaca clients."""
        self.trading_client = TradingClient(
            api_key=self.config.api_key,
            secret_key=self.config.secret_key,
            paper=self.config.paper
        )
        self.stock_client = StockHistoricalDataClient(
            api_key=self.config.api_key,
            secret_key=self.config.secret_key
        )
        self.option_client = OptionHistoricalDataClient(
            api_key=self.config.api_key,
            secret_key=self.config.secret_key
        )
        logging.info("Alpaca clients initialized (paper=%s)", self.config.paper)

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        clock = self.trading_client.get_clock()
        return clock.is_open

    def get_account_info(self) -> dict:
        """Get account information."""
        account = self.trading_client.get_account()
        return {
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "status": account.status
        }

    def get_price(self, underlying: str = "SPY") -> float:
        """Get current price for any underlying."""
        request = StockLatestQuoteRequest(symbol_or_symbols=[underlying])
        quotes = self.stock_client.get_stock_latest_quote(request)
        quote = quotes[underlying]
        # Use midpoint of bid/ask
        return (quote.bid_price + quote.ask_price) / 2

    # Backward compatibility
    def get_spy_price(self) -> float:
        """Get current SPY price (backward compatibility)."""
        return self.get_price("SPY")

    def get_positions(self) -> list:
        """Get all open positions."""
        return self.trading_client.get_all_positions()


def build_alpaca_option_symbol(expiration_date: str, strike: float, option_type: str,
                                underlying: str = "SPY") -> str:
    """
    Build Alpaca option symbol format for any underlying.

    Alpaca format: SPY241220C00600000 (no "O:" prefix)

    Args:
        expiration_date: Date in YYYY-MM-DD format
        strike: Strike price
        option_type: "C" for call, "P" for put
        underlying: Stock/ETF symbol (default: "SPY")

    Returns:
        Alpaca-formatted option symbol
    """
    dt = datetime.strptime(expiration_date, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")  # YYMMDD
    strike_str = f"{int(strike * 1000):08d}"  # Strike * 1000, zero-padded to 8 digits
    return f"{underlying}{date_str}{option_type.upper()}{strike_str}"


def estimate_iv_from_live_quotes(
    client_manager: AlpacaClientManager,
    spot_price: float,
    expiration_date: str,
    time_to_expiry_years: float,
    underlying: str = "SPY"
) -> Optional[float]:
    """
    Estimate IV from ATM option quotes.

    Uses the same methodology as the backtest (estimate_iv_from_atm)
    but with live Alpaca data instead of Polygon historical data.
    """
    r = RISK_FREE_RATE
    atm_strike = round(spot_price)

    call_symbol = build_alpaca_option_symbol(expiration_date, atm_strike, "C", underlying)
    put_symbol = build_alpaca_option_symbol(expiration_date, atm_strike, "P", underlying)

    try:
        request = OptionLatestQuoteRequest(symbol_or_symbols=[call_symbol, put_symbol])
        quotes = client_manager.option_client.get_option_latest_quote(request)
    except Exception as e:
        logging.error("Failed to get ATM quotes: %s", e)
        return None

    ivs = []

    # Calculate IV from call
    if call_symbol in quotes:
        call_quote = quotes[call_symbol]
        call_mid = (call_quote.bid_price + call_quote.ask_price) / 2
        if call_mid > 0.05:
            iv = implied_volatility(call_mid, spot_price, atm_strike,
                                    time_to_expiry_years, r, "call")
            if iv and 0.05 < iv < 2.0:
                ivs.append(iv)

    # Calculate IV from put
    if put_symbol in quotes:
        put_quote = quotes[put_symbol]
        put_mid = (put_quote.bid_price + put_quote.ask_price) / 2
        if put_mid > 0.05:
            iv = implied_volatility(put_mid, spot_price, atm_strike,
                                    time_to_expiry_years, r, "put")
            if iv and 0.05 < iv < 2.0:
                ivs.append(iv)

    if ivs:
        return sum(ivs) / len(ivs)

    # Fallback IV
    logging.warning("Could not calculate IV, using fallback 20%%")
    return 0.20


def submit_limit_sell_order(
    client_manager: AlpacaClientManager,
    symbol: str,
    quantity: int,
    limit_price: float
) -> Optional[str]:
    """
    Submit a limit sell order for an option.

    Returns order ID if successful, None otherwise.
    """
    try:
        request = LimitOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.SELL,
            limit_price=round(limit_price, 2),
            time_in_force=TimeInForce.DAY
        )
        order = client_manager.trading_client.submit_order(request)
        logging.info("Order submitted: SELL %d x %s @ $%.2f (ID: %s)",
                     quantity, symbol, limit_price, order.id)
        return str(order.id)
    except Exception as e:
        logging.error("Order submission failed for %s: %s", symbol, e)
        return None


def submit_market_sell_order(
    client_manager: AlpacaClientManager,
    symbol: str,
    quantity: int
) -> Optional[str]:
    """
    Submit a market sell order for an option.

    Returns order ID if successful, None otherwise.
    """
    try:
        request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        order = client_manager.trading_client.submit_order(request)
        logging.info("Order submitted: SELL %d x %s @ MARKET (ID: %s)",
                     quantity, symbol, order.id)
        return str(order.id)
    except Exception as e:
        logging.error("Order submission failed for %s: %s", symbol, e)
        return None


def execute_trades_for_underlying(client_manager: AlpacaClientManager,
                                   underlying: str = "SPY") -> Tuple[int, int]:
    """
    Execute trades for a single underlying.

    Args:
        client_manager: Alpaca client manager
        underlying: Stock/ETF symbol to trade

    Returns:
        Tuple of (orders_submitted, orders_skipped)
    """
    now = datetime.now(ET)
    today_str = now.strftime("%Y-%m-%d")

    logging.info("-" * 40)
    logging.info("Processing %s", underlying)

    # Get current price
    try:
        spot_price = client_manager.get_price(underlying)
        logging.info("%s price: $%.2f", underlying, spot_price)
    except Exception as e:
        logging.error("Failed to get %s price: %s", underlying, e)
        return 0, 0

    # Calculate time to expiration (market close is 4:00 PM ET)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    hours_to_close = (market_close - now).total_seconds() / 3600
    T = hours_to_close / 24 / 365  # Time in years

    # Estimate IV from ATM options
    iv = estimate_iv_from_live_quotes(client_manager, spot_price, today_str, T, underlying)
    if iv is None:
        logging.error("Could not estimate IV for %s, skipping", underlying)
        return 0, 0

    logging.info("%s IV: %.1f%%", underlying, iv * 100)

    # Get IV threshold for this underlying
    iv_threshold = get_iv_threshold(underlying)

    # IV Filter: Skip all trades if IV > threshold
    if iv_threshold and iv > iv_threshold:
        logging.info("%s IV %.1f%% > %.1f%% threshold - SKIPPING",
                     underlying, iv * 100, iv_threshold * 100)
        return 0, 2

    orders_submitted = 0
    orders_skipped = 0

    # Process CALL
    call_strike_theory = find_strike_for_delta(TARGET_DELTA, spot_price, T,
                                                RISK_FREE_RATE, iv, "call")
    if call_strike_theory:
        call_strike = round(call_strike_theory)
        call_symbol = build_alpaca_option_symbol(today_str, call_strike, "C", underlying)

        # Get quote for limit order price
        try:
            request = OptionLatestQuoteRequest(symbol_or_symbols=[call_symbol])
            quotes = client_manager.option_client.get_option_latest_quote(request)

            if call_symbol in quotes:
                quote = quotes[call_symbol]
                bid_price = quote.bid_price
                mid_price = (quote.bid_price + quote.ask_price) / 2

                if bid_price and bid_price > 0.01:
                    # Calculate actual delta for logging
                    actual_iv = implied_volatility(mid_price, spot_price, call_strike,
                                                   T, RISK_FREE_RATE, "call") or iv
                    actual_delta = call_delta(spot_price, call_strike, T, RISK_FREE_RATE, actual_iv)

                    otm_pct = (call_strike - spot_price) / spot_price * 100
                    logging.info("CALL %s: strike=%d (%+.1f%% OTM), bid=$%.2f, delta=%.3f",
                                 call_symbol, call_strike, otm_pct, bid_price, actual_delta)

                    # Submit order
                    order_price = bid_price if USE_BID_PRICE else mid_price
                    if ORDER_TYPE == "limit":
                        order_id = submit_limit_sell_order(
                            client_manager, call_symbol, MAX_POSITION_SIZE, order_price
                        )
                    else:
                        order_id = submit_market_sell_order(
                            client_manager, call_symbol, MAX_POSITION_SIZE
                        )

                    if order_id:
                        # Log to trade tracker (use Polygon format for compatibility)
                        polygon_ticker = f"O:{call_symbol}"
                        add_trade(polygon_ticker, order_price,
                                  f"Alpaca order {order_id}",
                                  alpaca_order_id=order_id, source="alpaca")
                        orders_submitted += 1
                    else:
                        orders_skipped += 1
                else:
                    logging.warning("CALL %s: bid price too low ($%.2f)", call_symbol, bid_price or 0)
                    orders_skipped += 1
            else:
                logging.warning("CALL %s: no quote available", call_symbol)
                orders_skipped += 1
        except Exception as e:
            logging.error("CALL processing error: %s", e)
            orders_skipped += 1

    # Process PUT
    put_strike_theory = find_strike_for_delta(-TARGET_DELTA, spot_price, T,
                                               RISK_FREE_RATE, iv, "put")
    if put_strike_theory:
        put_strike = round(put_strike_theory)
        put_symbol = build_alpaca_option_symbol(today_str, put_strike, "P", underlying)

        try:
            request = OptionLatestQuoteRequest(symbol_or_symbols=[put_symbol])
            quotes = client_manager.option_client.get_option_latest_quote(request)

            if put_symbol in quotes:
                quote = quotes[put_symbol]
                bid_price = quote.bid_price
                mid_price = (quote.bid_price + quote.ask_price) / 2

                if bid_price and bid_price > 0.01:
                    actual_iv = implied_volatility(mid_price, spot_price, put_strike,
                                                   T, RISK_FREE_RATE, "put") or iv
                    actual_delta = put_delta(spot_price, put_strike, T, RISK_FREE_RATE, actual_iv)

                    otm_pct = (spot_price - put_strike) / spot_price * 100
                    logging.info("PUT %s: strike=%d (%+.1f%% OTM), bid=$%.2f, delta=%.3f",
                                 put_symbol, put_strike, otm_pct, bid_price, abs(actual_delta))

                    order_price = bid_price if USE_BID_PRICE else mid_price
                    if ORDER_TYPE == "limit":
                        order_id = submit_limit_sell_order(
                            client_manager, put_symbol, MAX_POSITION_SIZE, order_price
                        )
                    else:
                        order_id = submit_market_sell_order(
                            client_manager, put_symbol, MAX_POSITION_SIZE
                        )

                    if order_id:
                        polygon_ticker = f"O:{put_symbol}"
                        add_trade(polygon_ticker, order_price,
                                  f"Alpaca order {order_id}",
                                  alpaca_order_id=order_id, source="alpaca")
                        orders_submitted += 1
                    else:
                        orders_skipped += 1
                else:
                    logging.warning("PUT %s: bid price too low ($%.2f)", put_symbol, bid_price or 0)
                    orders_skipped += 1
            else:
                logging.warning("PUT %s: no quote available", put_symbol)
                orders_skipped += 1
        except Exception as e:
            logging.error("PUT processing error: %s", e)
            orders_skipped += 1

    logging.info("%s complete: %d submitted, %d skipped",
                 underlying, orders_submitted, orders_skipped)
    return orders_submitted, orders_skipped


def execute_trades(client_manager: AlpacaClientManager) -> Tuple[int, int]:
    """
    Execute trades for all underlyings scheduled for today.

    This is the main function called by the scheduler.

    Returns:
        Tuple of (total_orders_submitted, total_orders_skipped)
    """
    now = datetime.now(ET)
    today_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    weekday = now.weekday()  # 0=Monday, 4=Friday

    logging.info("=" * 50)
    logging.info("Trade execution started at %s ET", time_str)

    # Check if market is open
    if not client_manager.is_market_open():
        logging.info("Market is closed, skipping")
        return 0, 0

    # Check 3:30 PM cutoff for 0DTE orders
    cutoff_time = dt_time(15, 30)
    if now.time() > cutoff_time:
        logging.warning("Past 3:30 PM ET cutoff for 0DTE orders")
        return 0, 0

    # Get underlyings to trade today
    underlyings = get_underlyings_for_day(weekday)
    logging.info("Underlyings for %s: %s", ["Mon","Tue","Wed","Thu","Fri"][weekday],
                 ", ".join(underlyings))

    total_submitted = 0
    total_skipped = 0

    for underlying in underlyings:
        try:
            submitted, skipped = execute_trades_for_underlying(client_manager, underlying)
            total_submitted += submitted
            total_skipped += skipped
        except Exception as e:
            logging.error("Error processing %s: %s", underlying, e)

    logging.info("=" * 50)
    logging.info("Total: %d submitted, %d skipped across %d underlyings",
                 total_submitted, total_skipped, len(underlyings))

    return total_submitted, total_skipped


def setup_scheduler(client_manager: AlpacaClientManager) -> BlockingScheduler:
    """
    Configure APScheduler for trade execution times.

    Schedules:
    - 10:00 AM ET
    - 10:30 AM ET
    - 11:00 AM ET

    Only runs Monday-Friday.
    """
    scheduler = BlockingScheduler(timezone=ET)

    # Parse entry times from config
    for time_str in ENTRY_TIMES:
        hour, minute = map(int, time_str.split(":"))

        trigger = CronTrigger(
            day_of_week="mon-fri",
            hour=hour,
            minute=minute,
            timezone=ET
        )

        scheduler.add_job(
            execute_trades,
            trigger=trigger,
            args=[client_manager],
            id=f"trade_{time_str}",
            name=f"Execute trades at {time_str} ET",
            misfire_grace_time=60  # Allow 60 seconds grace period
        )

        logging.info("Scheduled trade execution at %s ET (Mon-Fri)", time_str)

    return scheduler


def setup_signal_handlers(scheduler: BlockingScheduler):
    """Setup handlers for graceful shutdown."""

    def shutdown_handler(signum, frame):
        logging.info("Received shutdown signal, stopping scheduler...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)


def show_status(client_manager: AlpacaClientManager):
    """Show account status and positions."""
    print("\n" + "=" * 60)
    print("ALPACA ACCOUNT STATUS")
    print("=" * 60)

    # Account info
    account = client_manager.get_account_info()
    print(f"\nAccount Status: {account['status']}")
    print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
    print(f"Cash: ${account['cash']:,.2f}")
    print(f"Buying Power: ${account['buying_power']:,.2f}")

    # Market status
    clock = client_manager.trading_client.get_clock()
    print(f"\nMarket Open: {clock.is_open}")
    if not clock.is_open:
        print(f"Next Open: {clock.next_open}")

    # Positions
    positions = client_manager.get_positions()
    print(f"\n--- OPEN POSITIONS ({len(positions)}) ---")
    if positions:
        for pos in positions:
            print(f"  {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f}")
            print(f"    Current: ${float(pos.current_price):.2f}, P&L: ${float(pos.unrealized_pl):.2f}")
    else:
        print("  No open positions")


def main():
    """Main entry point for the trading daemon."""
    parser = argparse.ArgumentParser(
        description="0DTE Options Trading Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python alpaca_trader.py           # Run as daemon
  python alpaca_trader.py --once    # Execute once and exit
  python alpaca_trader.py --status  # Show account status
        """
    )
    parser.add_argument("--once", action="store_true",
                        help="Execute once immediately then exit")
    parser.add_argument("--status", action="store_true",
                        help="Show account status and positions")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Load configuration and connect
    try:
        config = load_alpaca_config()
        client_manager = AlpacaClientManager(config)
        client_manager.connect()
    except Exception as e:
        logging.error("Failed to initialize: %s", e)
        sys.exit(1)

    if args.status:
        show_status(client_manager)
        return

    if args.once:
        # Single execution mode
        logging.info("Running single execution...")
        submitted, skipped = execute_trades(client_manager)
        logging.info("Done: %d orders submitted, %d skipped", submitted, skipped)
        return

    # Daemon mode
    logging.info("Starting 0DTE trading daemon")
    logging.info("Strategy: Sell %.0f-delta calls and puts", TARGET_DELTA * 100)
    logging.info("Entry times: %s ET", ", ".join(ENTRY_TIMES))
    logging.info("Order type: %s at %s", ORDER_TYPE, "bid" if USE_BID_PRICE else "mid")

    # Show underlyings configuration
    logging.info("Underlyings configured:")
    for day_name, day_num in [("Mon", 0), ("Tue", 1), ("Wed", 2), ("Thu", 3), ("Fri", 4)]:
        symbols = get_underlyings_for_day(day_num)
        if symbols:
            logging.info("  %s: %s", day_name, ", ".join(symbols))

    scheduler = setup_scheduler(client_manager)
    setup_signal_handlers(scheduler)

    try:
        logging.info("Daemon started, waiting for scheduled times...")
        logging.info("Press Ctrl+C to stop")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Daemon stopped")


if __name__ == "__main__":
    main()
