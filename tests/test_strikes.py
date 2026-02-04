"""Tests for strike selection and expiry calendar."""

import datetime as dt
import pytest

from backtester.strikes import (
    get_atm_strike, get_otm_strikes, get_strike_range,
    next_thursdays, next_expiry_dates, days_to_expiry, time_to_expiry_years,
    describe_strike, is_trading_day, adjust_expiry, NSE_HOLIDAYS,
    STRIKE_INTERVAL, LOT_SIZE,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_strike_interval(self):
        assert STRIKE_INTERVAL == 50

    def test_lot_size(self):
        assert LOT_SIZE == 25


# ---------------------------------------------------------------------------
# ATM strike
# ---------------------------------------------------------------------------

class TestGetATMStrike:
    def test_exact_multiple(self):
        assert get_atm_strike(22000) == 22000

    def test_rounds_up(self):
        assert get_atm_strike(22030) == 22050

    def test_rounds_down(self):
        assert get_atm_strike(22020) == 22000

    def test_midpoint_rounds(self):
        # Python uses banker's rounding: 22025/50=440.5 rounds to 440
        assert get_atm_strike(22025) == 22000
        # 22075/50=441.5 rounds to 442
        assert get_atm_strike(22075) == 22100

    def test_below_strike(self):
        assert get_atm_strike(21990) == 22000

    def test_custom_interval(self):
        assert get_atm_strike(22030, interval=100) == 22000
        assert get_atm_strike(22060, interval=100) == 22100

    def test_large_spot(self):
        assert get_atm_strike(24567) == 24550

    def test_small_spot(self):
        assert get_atm_strike(18123) == 18100


# ---------------------------------------------------------------------------
# OTM strikes
# ---------------------------------------------------------------------------

class TestGetOTMStrikes:
    def test_ce_otm_above_spot(self):
        """CE OTM strikes are above the ATM."""
        strikes = get_otm_strikes(22000, "CE", num_strikes=4)
        assert strikes == [22050, 22100, 22150, 22200]

    def test_pe_otm_below_spot(self):
        """PE OTM strikes are below the ATM."""
        strikes = get_otm_strikes(22000, "PE", num_strikes=4)
        assert strikes == [21950, 21900, 21850, 21800]

    def test_num_strikes(self):
        strikes = get_otm_strikes(22000, "CE", num_strikes=2)
        assert len(strikes) == 2

    def test_non_round_spot_ce(self):
        """ATM for 22030 = 22050, then OTM CEs go up."""
        strikes = get_otm_strikes(22030, "CE", num_strikes=2)
        assert strikes == [22100, 22150]

    def test_non_round_spot_pe(self):
        """ATM for 22030 = 22050, then OTM PEs go down."""
        strikes = get_otm_strikes(22030, "PE", num_strikes=2)
        assert strikes == [22000, 21950]

    def test_single_strike(self):
        strikes = get_otm_strikes(22000, "CE", num_strikes=1)
        assert strikes == [22050]


# ---------------------------------------------------------------------------
# Strike range (ATM + OTM)
# ---------------------------------------------------------------------------

class TestGetStrikeRange:
    def test_atm_plus_otm_ce(self):
        result = get_strike_range(22000, "CE", num_otm=3)
        assert result == [22000, 22050, 22100, 22150]

    def test_atm_plus_otm_pe(self):
        result = get_strike_range(22000, "PE", num_otm=3)
        assert result == [22000, 21950, 21900, 21850]

    def test_atm_only(self):
        result = get_strike_range(22000, "CE", num_otm=0)
        assert result == [22000]

    def test_length(self):
        result = get_strike_range(22000, "CE", num_otm=4)
        assert len(result) == 5  # ATM + 4 OTM


# ---------------------------------------------------------------------------
# Next expiry dates (Tuesday)
# ---------------------------------------------------------------------------

class TestNextExpiries:
    def test_from_monday(self):
        """Monday → next Tuesday is 1 day later."""
        mon = dt.date(2024, 1, 1)  # Monday
        result = next_expiry_dates(mon, count=1)
        assert len(result) == 1
        assert result[0] == dt.date(2024, 1, 2)  # Tuesday
        assert result[0].weekday() == 1

    def test_from_tuesday(self):
        """Tuesday → skip current day, return NEXT Tuesday."""
        tue = dt.date(2024, 1, 2)  # Tuesday
        result = next_expiry_dates(tue, count=1)
        assert result[0] == dt.date(2024, 1, 9)

    def test_four_expiries(self):
        result = next_expiry_dates(dt.date(2024, 1, 1), count=4)
        assert len(result) == 4

    def test_expiries_are_roughly_weekly(self):
        result = next_expiry_dates(dt.date(2024, 1, 1), count=4)
        for i in range(1, len(result)):
            gap = (result[i] - result[i - 1]).days
            assert 5 <= gap <= 8  # allow for holiday adjustment

    def test_from_wednesday(self):
        """Wednesday → next Tuesday is 6 days later."""
        wed = dt.date(2024, 1, 3)
        result = next_expiry_dates(wed, count=1)
        assert result[0] == dt.date(2024, 1, 9)

    def test_from_friday(self):
        """Friday → next Tuesday is 4 days later."""
        fri = dt.date(2024, 1, 5)
        result = next_expiry_dates(fri, count=1)
        assert result[0] == dt.date(2024, 1, 9)

    def test_backward_compat_alias(self):
        """next_thursdays still works as an alias."""
        result_new = next_expiry_dates(dt.date(2024, 1, 1), count=4)
        result_old = next_thursdays(dt.date(2024, 1, 1), count=4)
        assert result_new == result_old


# ---------------------------------------------------------------------------
# Days/time to expiry
# ---------------------------------------------------------------------------

class TestDaysToExpiry:
    def test_positive_days(self):
        assert days_to_expiry(dt.date(2024, 1, 1), dt.date(2024, 1, 8)) == 7

    def test_same_day(self):
        assert days_to_expiry(dt.date(2024, 1, 1), dt.date(2024, 1, 1)) == 0

    def test_past_expiry_returns_zero(self):
        assert days_to_expiry(dt.date(2024, 1, 10), dt.date(2024, 1, 1)) == 0


class TestTimeToExpiryYears:
    def test_seven_days(self):
        result = time_to_expiry_years(dt.date(2024, 1, 1), dt.date(2024, 1, 8))
        assert pytest.approx(result, abs=1e-6) == 7 / 365.0

    def test_one_year(self):
        result = time_to_expiry_years(dt.date(2024, 1, 1), dt.date(2025, 1, 1))
        assert pytest.approx(result, abs=0.01) == 1.0

    def test_expired_returns_zero(self):
        result = time_to_expiry_years(dt.date(2024, 2, 1), dt.date(2024, 1, 1))
        assert result == 0.0


# ---------------------------------------------------------------------------
# Describe strike
# ---------------------------------------------------------------------------

class TestDescribeStrike:
    def test_atm(self):
        assert describe_strike(22000, 22000, "CE") == "ATM"

    def test_1_otm_ce(self):
        assert describe_strike(22000, 22050, "CE") == "1-OTM"

    def test_2_otm_pe(self):
        assert describe_strike(22000, 21900, "PE") == "2-OTM"

    def test_non_round_spot(self):
        # spot=22030 → ATM=22050, strike=22050 → ATM
        assert describe_strike(22030, 22050, "CE") == "ATM"

    def test_4_otm(self):
        assert describe_strike(22000, 22200, "CE") == "4-OTM"


# ---------------------------------------------------------------------------
# NSE holidays & expiry adjustment
# ---------------------------------------------------------------------------

class TestNSEHolidays:
    def test_holidays_set_not_empty(self):
        assert len(NSE_HOLIDAYS) > 0

    def test_republic_day_2024_is_holiday(self):
        assert dt.date(2024, 1, 26) in NSE_HOLIDAYS

    def test_christmas_2024_is_holiday(self):
        assert dt.date(2024, 12, 25) in NSE_HOLIDAYS

    def test_regular_trading_day(self):
        # Jan 2, 2024 is a Tuesday — normal trading day
        assert is_trading_day(dt.date(2024, 1, 2))

    def test_saturday_not_trading_day(self):
        assert not is_trading_day(dt.date(2024, 1, 6))

    def test_sunday_not_trading_day(self):
        assert not is_trading_day(dt.date(2024, 1, 7))

    def test_holiday_not_trading_day(self):
        assert not is_trading_day(dt.date(2024, 1, 26))

    def test_adjust_expiry_normal_tuesday(self):
        """Non-holiday Tuesday stays as is."""
        tue = dt.date(2024, 1, 2)  # regular Tuesday
        assert adjust_expiry(tue) == tue

    def test_adjust_expiry_holiday_tuesday(self):
        """Holiday Tuesday moves to previous trading day (Monday)."""
        # Oct 21, 2025 is Diwali (Laxmi Pujan) — a Tuesday
        tue = dt.date(2025, 10, 21)
        assert tue.weekday() == 1  # confirm it's a Tuesday
        adjusted = adjust_expiry(tue)
        assert adjusted < tue
        assert is_trading_day(adjusted)

    def test_next_expiry_dates_adjusts_holidays(self):
        """next_expiry_dates should return adjusted dates."""
        # Test near Oct 21, 2025 (Diwali on a Tuesday)
        result = next_expiry_dates(dt.date(2025, 10, 14), count=2)
        assert len(result) == 2
        for d in result:
            assert is_trading_day(d)

    def test_all_expiry_dates_are_trading_days(self):
        """All returned expiry dates should be valid trading days."""
        result = next_expiry_dates(dt.date(2024, 1, 1), count=52)
        for d in result:
            assert is_trading_day(d), f"{d} is not a trading day"
