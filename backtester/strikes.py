"""Strike selection and expiry calendar for Nifty F&O.

Nifty options:
    - Strike interval: 50 points
    - Lot size: 25 (as of 2024, configurable)
    - Weekly expiry: every Tuesday (changed from Thursday mid-2024)
    - Monthly expiry: last Tuesday of month
    - If Tuesday is an NSE holiday, expiry moves to previous trading day
"""

import datetime as dt

STRIKE_INTERVAL = 50
LOT_SIZE = 25

# ---------------------------------------------------------------------------
# NSE holidays (dates when the exchange is closed)
# Covers 2024-2026.  Add future years as needed.
# Source: NSE circular / official holiday lists.
# ---------------------------------------------------------------------------

NSE_HOLIDAYS: set[dt.date] = {
    # 2024
    dt.date(2024, 1, 26),   # Republic Day
    dt.date(2024, 3, 8),    # Maha Shivaratri
    dt.date(2024, 3, 25),   # Holi
    dt.date(2024, 3, 29),   # Good Friday
    dt.date(2024, 4, 11),   # Id-Ul-Fitr (Eid)
    dt.date(2024, 4, 14),   # Dr. Ambedkar Jayanti
    dt.date(2024, 4, 17),   # Ram Navami
    dt.date(2024, 4, 21),   # Mahavir Jayanti
    dt.date(2024, 5, 1),    # Maharashtra Day
    dt.date(2024, 5, 23),   # Buddha Purnima
    dt.date(2024, 6, 17),   # Eid-Ul-Adha (Bakri Id)
    dt.date(2024, 7, 17),   # Muharram
    dt.date(2024, 8, 15),   # Independence Day
    dt.date(2024, 9, 16),   # Milad-un-Nabi
    dt.date(2024, 10, 2),   # Mahatma Gandhi Jayanti
    dt.date(2024, 10, 12),  # Dussehra
    dt.date(2024, 11, 1),   # Diwali (Laxmi Pujan)
    dt.date(2024, 11, 15),  # Guru Nanak Jayanti
    dt.date(2024, 12, 25),  # Christmas

    # 2025
    dt.date(2025, 1, 26),   # Republic Day
    dt.date(2025, 2, 26),   # Maha Shivaratri
    dt.date(2025, 3, 14),   # Holi
    dt.date(2025, 3, 31),   # Id-Ul-Fitr (Eid)
    dt.date(2025, 4, 10),   # Mahavir Jayanti
    dt.date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    dt.date(2025, 4, 18),   # Good Friday
    dt.date(2025, 5, 1),    # Maharashtra Day
    dt.date(2025, 5, 12),   # Buddha Purnima
    dt.date(2025, 6, 7),    # Eid-Ul-Adha (Bakri Id)
    dt.date(2025, 7, 6),    # Muharram
    dt.date(2025, 8, 15),   # Independence Day
    dt.date(2025, 8, 16),   # Parsi New Year
    dt.date(2025, 9, 5),    # Milad-un-Nabi
    dt.date(2025, 10, 2),   # Mahatma Gandhi Jayanti / Dussehra
    dt.date(2025, 10, 21),  # Diwali (Laxmi Pujan)
    dt.date(2025, 10, 22),  # Diwali (Balipratipada)
    dt.date(2025, 11, 5),   # Guru Nanak Jayanti
    dt.date(2025, 12, 25),  # Christmas

    # 2026
    dt.date(2026, 1, 26),   # Republic Day
    dt.date(2026, 2, 17),   # Maha Shivaratri
    dt.date(2026, 3, 4),    # Holi
    dt.date(2026, 3, 20),   # Id-Ul-Fitr (Eid)
    dt.date(2026, 3, 30),   # Ram Navami
    dt.date(2026, 4, 2),    # Mahavir Jayanti
    dt.date(2026, 4, 3),    # Good Friday
    dt.date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    dt.date(2026, 5, 1),    # Maharashtra Day / Buddha Purnima
    dt.date(2026, 5, 27),   # Eid-Ul-Adha (Bakri Id)
    dt.date(2026, 6, 26),   # Muharram
    dt.date(2026, 8, 15),   # Independence Day
    dt.date(2026, 8, 25),   # Milad-un-Nabi
    dt.date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    dt.date(2026, 10, 19),  # Dussehra
    dt.date(2026, 11, 9),   # Diwali (Laxmi Pujan)
    dt.date(2026, 11, 24),  # Guru Nanak Jayanti
    dt.date(2026, 12, 25),  # Christmas
}


def is_trading_day(d: dt.date) -> bool:
    """Check if a date is an NSE trading day (not weekend, not holiday)."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return d not in NSE_HOLIDAYS


def adjust_expiry(d: dt.date) -> dt.date:
    """If the expiry date is an NSE holiday, move to previous trading day.
    This matches real NSE behavior."""
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


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

    For CE: OTM = strikes above spot (higher strike -> cheaper premium).
    For PE: OTM = strikes below spot (lower strike -> cheaper premium).

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


def next_expiry_dates(from_date: dt.date, count: int = 4) -> list[dt.date]:
    """Return the next *count* expiry dates from (but not including) from_date.

    Expiry is normally Tuesday (weekday 1). If that Tuesday is an NSE
    holiday, the expiry moves to the previous trading day.
    """
    expiries = []
    d = from_date + dt.timedelta(days=1)
    while len(expiries) < count:
        if d.weekday() == 1:  # Tuesday
            actual_expiry = adjust_expiry(d)
            expiries.append(actual_expiry)
        d += dt.timedelta(days=1)
    return expiries


def next_thursdays(from_date: dt.date, count: int = 4) -> list[dt.date]:
    """Return the next *count* expiry dates (holiday-adjusted).

    Backwards-compatible alias for next_expiry_dates.
    """
    return next_expiry_dates(from_date, count)


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
