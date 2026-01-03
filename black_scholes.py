"""
Black-Scholes calculations for options pricing and Greeks.
"""
import math
from scipy.stats import norm
from scipy.optimize import brentq


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes call option price."""
    if T <= 0:
        return max(S - K, 0)
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    return S * norm.cdf(d1_val) - K * math.exp(-r * T) * norm.cdf(d2_val)


def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes put option price."""
    if T <= 0:
        return max(K - S, 0)
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)


def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate delta for a call option. Returns value between 0 and 1."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    return norm.cdf(d1(S, K, T, r, sigma))


def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate delta for a put option. Returns value between -1 and 0."""
    if T <= 0:
        return -1.0 if S < K else 0.0
    return norm.cdf(d1(S, K, T, r, sigma)) - 1


def implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    precision: float = 1e-6
) -> float | None:
    """
    Calculate implied volatility from option price using Brent's method.

    Returns None if IV cannot be calculated (e.g., price below intrinsic value).
    """
    if T <= 0:
        return None

    # Intrinsic value check
    if option_type == "call":
        intrinsic = max(S - K, 0)
        price_func = lambda sigma: call_price(S, K, T, r, sigma)
    else:
        intrinsic = max(K - S, 0)
        price_func = lambda sigma: put_price(S, K, T, r, sigma)

    if option_price < intrinsic:
        return None

    # Try to find IV in reasonable range (1% to 500%)
    try:
        iv = brentq(
            lambda sigma: price_func(sigma) - option_price,
            0.01,  # 1% volatility
            5.0,   # 500% volatility
            xtol=precision
        )
        return iv
    except ValueError:
        return None


def find_strike_for_delta(
    target_delta: float,
    S: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    precision: float = 0.01
) -> float:
    """
    Find the strike price that gives approximately the target delta.

    Args:
        target_delta: Target delta (positive for calls, negative for puts)
        S: Underlying price
        T: Time to expiration in years
        r: Risk-free rate
        sigma: Implied volatility
        option_type: "call" or "put"
        precision: Strike precision

    Returns:
        Strike price closest to target delta
    """
    if option_type == "call":
        # Calls: delta decreases as strike increases
        # For 10 delta call, strike is above spot
        low_strike = S * 0.8
        high_strike = S * 1.5
        delta_func = lambda K: call_delta(S, K, T, r, sigma) - target_delta
    else:
        # Puts: delta increases (less negative) as strike increases
        # For -10 delta put, strike is below spot
        low_strike = S * 0.5
        high_strike = S * 1.2
        delta_func = lambda K: put_delta(S, K, T, r, sigma) - target_delta

    try:
        strike = brentq(delta_func, low_strike, high_strike, xtol=precision)
        return strike
    except ValueError:
        return None


if __name__ == "__main__":
    # Test calculations
    S = 588  # SPY price
    K_call = 610  # Call strike
    K_put = 555  # Put strike
    T = 6.5 / 24 / 365  # 6.5 hours to expiration
    r = 0.05  # 5% risk-free rate
    sigma = 0.15  # 15% IV

    print(f"Test parameters: S={S}, T={T:.6f} years ({T*365*24:.1f} hours)")
    print(f"Risk-free rate: {r:.1%}, IV: {sigma:.1%}")
    print()

    # Calculate deltas
    print(f"Call at {K_call} strike: delta = {call_delta(S, K_call, T, r, sigma):.4f}")
    print(f"Put at {K_put} strike: delta = {put_delta(S, K_put, T, r, sigma):.4f}")
    print()

    # Find 10 delta strikes
    call_strike = find_strike_for_delta(0.10, S, T, r, sigma, "call")
    put_strike = find_strike_for_delta(-0.10, S, T, r, sigma, "put")
    print(f"10 delta call strike: {call_strike:.2f}")
    print(f"10 delta put strike: {put_strike:.2f}")
