"""Strike selection and expiry calendar for Nifty F&O.

Nifty options:
    - Strike interval: 50 points
    - Lot size: 25 (as of 2024, configurable)
    - Weekly expiry: every Thursday
    - Monthly expiry: last Thursday of month
"""

import datetime as dt

STRIKE_INTERVAL = 50
LOT_SIZE = 25


def get_atm_strike(spot_price: float, interval: int = STRIKE_INTERVAL) -> int:
    """Return the ATM strike (nearest strike to spot price)."""
    return int(round(spot_price / interval) * interval)


def get_otm_strikes(
    spot_price: float,
    option_type: str,
    num_strikes: int = 4,
    interval: int = STRIKE_INTERVAL,
) -> list[int]:
    """Return a list of OTM strikes away from ATM.

    For CE: OTM = strikes above spot (higher strike → cheaper premium).
    For PE: OTM = strikes below spot (lower strike → cheaper premium).

    Returns:
        List of strikes: [1-OTM, 2-OTM, ..., num_strikes-OTM]
    """
    atm = get_atm_strike(spot_price, interval)
    strikes = []
    for i in range(1, num_strikes + 1):
        if option_type == "CE":
            strikes.append(atm + i * interval)
        else:
            strikes.append(atm - i * interval)
    return strikes


def get_strike_range(
    spot_price: float,
    option_type: str,
    num_otm: int = 4,
    interval: int = STRIKE_INTERVAL,
) -> list[int]:
    """Return [ATM, 1-OTM, 2-OTM, ..., num_otm-OTM] for given option type.

    This is the full range we test: ATM plus N OTM strikes.
    """
    atm = get_atm_strike(spot_price, interval)
    return [atm] + get_otm_strikes(spot_price, option_type, num_otm, interval)


def next_thursdays(from_date: dt.date, count: int = 4) -> list[dt.date]:
    """Return the next *count* Thursdays from (but not including) from_date.

    These represent weekly expiry dates. Thursday = weekday 3.
    """
    thursdays = []
    d = from_date + dt.timedelta(days=1)
    while len(thursdays) < count:
        if d.weekday() == 3:  # Thursday
            thursdays.append(d)
        d += dt.timedelta(days=1)
    return thursdays


def days_to_expiry(current_date: dt.date, expiry_date: dt.date) -> int:
    """Calendar days remaining until expiry."""
    return max((expiry_date - current_date).days, 0)


def time_to_expiry_years(current_date: dt.date, expiry_date: dt.date) -> float:
    """Time to expiry as fraction of year (for Black-Scholes T)."""
    days = days_to_expiry(current_date, expiry_date)
    return days / 365.0


def describe_strike(
    spot_price: float, strike: int, option_type: str,
    interval: int = STRIKE_INTERVAL,
) -> str:
    """Human-readable label like 'ATM', '1-OTM', '2-OTM' etc."""
    atm = get_atm_strike(spot_price, interval)
    diff = abs(strike - atm) // interval
    if diff == 0:
        return "ATM"
    return f"{diff}-OTM"
