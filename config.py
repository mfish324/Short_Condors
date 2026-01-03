"""
Configuration for 0DTE Options Trading Strategy
"""

# Strategy Parameters
TARGET_DELTA = 0.10  # Target delta for options
ENTRY_TIMES = ["10:00", "10:30", "11:00"]  # Eastern Time

# Filter Settings (Updated based on 2024-2025 backtest results)
# The IV filter is the only consistently profitable filter across both periods
SKIP_EVENT_DAYS = False  # Event filter was inconsistent (helped in 2024, hurt in 2025)
SKIP_PUTS_HIGH_IV = False  # Superseded by MAX_IV_THRESHOLD
SKIP_HALF_DAYS = False  # Negligible impact
MAX_IV_THRESHOLD = 0.20  # Skip ALL trades when IV > 20% (BEST FILTER: +$11,746 in 2025)

# Risk Parameters
MAX_POSITION_SIZE = 1  # Number of contracts per trade
RISK_FREE_RATE = 0.05  # For Black-Scholes calculations

# Alpaca Trading Settings
ALPACA_PAPER_MODE = True  # Always use paper trading
ORDER_TYPE = "limit"  # Order type: "limit" or "market"
USE_BID_PRICE = True  # Use bid price for limit orders (vs mid price)

# Multi-Underlying Configuration
# days: which days to trade (0=Mon, 4=Fri)
# iv_threshold: max IV to trade (None = no filter)
# Based on backtest: ETFs need IV filter, stocks profitable without filter
UNDERLYINGS = {
    # ETFs - daily/frequent 0DTE availability
    "SPY": {
        "days": [0, 1, 2, 3, 4],  # Mon-Fri (daily 0DTE)
        "iv_threshold": 0.20,     # 20% IV filter
        "enabled": True,
    },
    "QQQ": {
        "days": [0, 2, 4],        # Mon/Wed/Fri (0DTE on these days)
        "iv_threshold": 0.20,     # 20% IV filter
        "enabled": True,
    },
    # Individual stocks - Friday only (weekly 0DTE)
    # Higher IV but still profitable per backtest
    "AAPL": {"days": [4], "iv_threshold": None, "enabled": True},
    "MSFT": {"days": [4], "iv_threshold": None, "enabled": True},
    "NVDA": {"days": [4], "iv_threshold": None, "enabled": True},
    "AMD": {"days": [4], "iv_threshold": None, "enabled": True},
    "AMZN": {"days": [4], "iv_threshold": None, "enabled": True},
    "GOOGL": {"days": [4], "iv_threshold": None, "enabled": True},
    "META": {"days": [4], "iv_threshold": None, "enabled": True},
    "TSLA": {"days": [4], "iv_threshold": None, "enabled": True},
    "NFLX": {"days": [4], "iv_threshold": None, "enabled": True},
    "CRM": {"days": [4], "iv_threshold": None, "enabled": True},
}


def get_underlyings_for_day(weekday: int) -> list[str]:
    """Get list of underlyings to trade on a given weekday (0=Mon, 4=Fri)."""
    return [
        symbol for symbol, config in UNDERLYINGS.items()
        if config["enabled"] and weekday in config["days"]
    ]


def get_iv_threshold(symbol: str) -> float | None:
    """Get IV threshold for a symbol. Returns None if no filter."""
    if symbol in UNDERLYINGS:
        return UNDERLYINGS[symbol].get("iv_threshold")
    return MAX_IV_THRESHOLD  # Default for unknown symbols

# 2024 Event Calendar
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

HALF_DAYS_2024 = [
    "2024-07-03", "2024-11-29", "2024-12-24"
]

# 2025 Event Calendar
FOMC_DATES_2025 = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17"
]

CPI_DATES_2025 = [
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-12",
    "2025-09-11", "2025-10-10", "2025-11-13", "2025-12-10"
]

JOBS_REPORT_DATES_2025 = [
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05"
]

HALF_DAYS_2025 = [
    "2025-07-03", "2025-11-28", "2025-12-24"
]

# 2026 Event Calendar (projected)
FOMC_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16"
]

CPI_DATES_2026 = [
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-14",
    "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-11", "2026-10-13", "2026-11-12", "2026-12-10"
]

JOBS_REPORT_DATES_2026 = [
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-08", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04"
]

HALF_DAYS_2026 = [
    "2026-07-03", "2026-11-27", "2026-12-24"
]

# Combine all events
ALL_FOMC_DATES = set(FOMC_DATES_2024 + FOMC_DATES_2025 + FOMC_DATES_2026)
ALL_CPI_DATES = set(CPI_DATES_2024 + CPI_DATES_2025 + CPI_DATES_2026)
ALL_JOBS_DATES = set(JOBS_REPORT_DATES_2024 + JOBS_REPORT_DATES_2025 + JOBS_REPORT_DATES_2026)
ALL_HALF_DAYS = set(HALF_DAYS_2024 + HALF_DAYS_2025 + HALF_DAYS_2026)

ALL_EVENT_DATES = ALL_FOMC_DATES | ALL_CPI_DATES | ALL_JOBS_DATES


def is_event_day(date_str: str) -> tuple[bool, str]:
    """Check if date is an event day. Returns (is_event, event_type)."""
    if date_str in ALL_FOMC_DATES:
        return True, "FOMC"
    if date_str in ALL_CPI_DATES:
        return True, "CPI"
    if date_str in ALL_JOBS_DATES:
        return True, "Jobs Report"
    return False, ""


def is_half_day(date_str: str) -> bool:
    """Check if date is a half trading day."""
    return date_str in ALL_HALF_DAYS
