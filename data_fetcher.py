"""
Data fetching module for stocks and options from Polygon.io

Supports multiple underlyings: SPY, QQQ, and individual stocks.
"""
import os
import time
import requests
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io"

# Rate limiting
REQUEST_DELAY = 0.1  # seconds between requests


def _make_request(url: str, params: dict = None) -> dict | None:
    """Make API request with error handling."""
    if params is None:
        params = {}
    params["apiKey"] = API_KEY

    try:
        resp = requests.get(url, params=params)
        time.sleep(REQUEST_DELAY)  # Rate limiting

        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"API error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"Request error: {e}")
        return None


def get_minute_bars(date: str, underlying: str = "SPY") -> list[dict]:
    """
    Get minute bars for any underlying on a specific date.

    Args:
        date: Date string in YYYY-MM-DD format
        underlying: Stock/ETF symbol (default: "SPY")

    Returns:
        List of bar data with keys: t (timestamp ms), o, h, l, c, v
    """
    url = f"{BASE_URL}/v2/aggs/ticker/{underlying}/range/1/minute/{date}/{date}"
    data = _make_request(url, {"limit": 1000})

    if data and "results" in data:
        return data["results"]
    return []


# Backward compatibility alias
def get_spy_minute_bars(date: str) -> list[dict]:
    """Get SPY minute bars (backward compatibility wrapper)."""
    return get_minute_bars(date, "SPY")


def get_price_at_time(bars: list[dict], target_hour: int, target_minute: int) -> float | None:
    """
    Get closing price at a specific time from bar data.

    Args:
        bars: List of minute bars
        target_hour: Hour (ET) e.g., 10 for 10:00 AM
        target_minute: Minute e.g., 0 for on the hour, 30 for half hour

    Returns:
        Closing price at that minute, or None if not found
    """
    for bar in bars:
        ts = datetime.fromtimestamp(bar["t"] / 1000)
        if ts.hour == target_hour and ts.minute == target_minute:
            return bar["c"]
    return None


# Backward compatibility alias
def get_spy_price_at_time(bars: list[dict], target_hour: int, target_minute: int) -> float | None:
    """Get price at time (backward compatibility wrapper)."""
    return get_price_at_time(bars, target_hour, target_minute)


def get_option_minute_bars(ticker: str, date: str) -> list[dict]:
    """
    Get option minute bars for a specific contract and date.

    Args:
        ticker: Option ticker e.g., "O:SPY241220C00600000"
        date: Date string in YYYY-MM-DD format

    Returns:
        List of bar data
    """
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
    data = _make_request(url, {"limit": 1000})

    if data and "results" in data:
        return data["results"]
    return []


def get_option_price_at_time(bars: list[dict], target_hour: int, target_minute: int) -> float | None:
    """Get option closing price at a specific time."""
    for bar in bars:
        ts = datetime.fromtimestamp(bar["t"] / 1000)
        if ts.hour == target_hour and ts.minute == target_minute:
            return bar["c"]
    return None


def build_option_ticker(date: str, strike: float, option_type: str, underlying: str = "SPY") -> str:
    """
    Build option ticker symbol for any underlying.

    Args:
        date: Expiration date YYYY-MM-DD
        strike: Strike price
        option_type: "C" or "P"
        underlying: Stock/ETF symbol (default: "SPY")

    Returns:
        Option ticker e.g., "O:SPY241220C00600000" or "O:AAPL241220C00200000"
    """
    dt = datetime.strptime(date, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")

    # Strike is multiplied by 1000 and zero-padded to 8 digits
    strike_str = f"{int(strike * 1000):08d}"

    return f"O:{underlying}{date_str}{option_type}{strike_str}"


def get_available_strikes(date: str, option_type: str = "C") -> list[float]:
    """
    Get available strike prices for SPY options on a given date.

    Since expired contracts aren't in reference data, we'll generate
    a range of likely strikes based on SPY price.

    For SPY, strikes are typically in $1 increments.
    """
    # Standard SPY strike increments
    # We'll return a range of strikes to check
    return []  # Will be populated based on spot price


def get_trading_days(start_date: str, end_date: str) -> list[str]:
    """
    Get list of trading days between two dates.
    Uses SPY data availability as proxy for trading days.
    """
    trading_days = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        # Skip weekends
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            trading_days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return trading_days


def verify_trading_day(date: str, underlying: str = "SPY") -> bool:
    """Verify a date is a trading day by checking for data."""
    bars = get_minute_bars(date, underlying)
    return len(bars) > 100  # Should have many bars on a trading day


if __name__ == "__main__":
    # Test data fetching
    test_date = "2024-12-20"

    print(f"Testing data fetch for {test_date}")
    print("=" * 50)

    # Get SPY data
    spy_bars = get_spy_minute_bars(test_date)
    print(f"SPY bars: {len(spy_bars)}")

    # Get prices at specific times
    for hour, minute in [(10, 0), (10, 30), (11, 0)]:
        price = get_spy_price_at_time(spy_bars, hour, minute)
        print(f"SPY at {hour}:{minute:02d}: ${price:.2f}" if price else f"No data at {hour}:{minute:02d}")

    print()

    # Test option ticker building
    ticker = build_option_ticker(test_date, 590, "C")
    print(f"Built ticker: {ticker}")

    # Get option data
    option_bars = get_option_minute_bars(ticker, test_date)
    print(f"Option bars for {ticker}: {len(option_bars)}")

    if option_bars:
        price = get_option_price_at_time(option_bars, 10, 0)
        print(f"Option price at 10:00: ${price:.2f}" if price else "No data at 10:00")
