"""Tests for strike selection and expiry calendar."""

import datetime as dt
import pytest

from backtester.strikes import (
    get_atm_strike, get_otm_strikes, get_strike_range,
    next_thursdays, days_to_expiry, time_to_expiry_years,
    describe_strike,
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
# Next Thursdays
# ---------------------------------------------------------------------------

class TestNextThursdays:
    def test_from_monday(self):
        """Monday → next Thursday is 3 days later."""
        mon = dt.date(2024, 1, 1)  # Monday
        result = next_thursdays(mon, count=1)
        assert len(result) == 1
        assert result[0] == dt.date(2024, 1, 4)  # Thursday
        assert result[0].weekday() == 3

    def test_from_thursday(self):
        """Thursday → skip current day, return NEXT Thursday."""
        thu = dt.date(2024, 1, 4)  # Thursday
        result = next_thursdays(thu, count=1)
        assert result[0] == dt.date(2024, 1, 11)

    def test_four_thursdays(self):
        result = next_thursdays(dt.date(2024, 1, 1), count=4)
        assert len(result) == 4
        for d in result:
            assert d.weekday() == 3

    def test_thursdays_are_consecutive_weeks(self):
        result = next_thursdays(dt.date(2024, 1, 1), count=4)
        for i in range(1, len(result)):
            assert (result[i] - result[i - 1]).days == 7

    def test_from_wednesday(self):
        """Wednesday → next day is Thursday."""
        wed = dt.date(2024, 1, 3)
        result = next_thursdays(wed, count=1)
        assert result[0] == dt.date(2024, 1, 4)

    def test_from_friday(self):
        """Friday → skip weekend, next Thursday is 6 days later."""
        fri = dt.date(2024, 1, 5)
        result = next_thursdays(fri, count=1)
        assert result[0] == dt.date(2024, 1, 11)


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
