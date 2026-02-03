import numpy as np
import pandas as pd
import pytest

from backtester.indicators import vwap, obv, ad_line


class TestVWAP:
    def test_basic_calculation(self, sample_ohlcv):
        result = vwap(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
        assert not result.isna().all()

    def test_first_bar_equals_typical_price(self, sample_ohlcv):
        """First bar of the day: VWAP == typical price."""
        result = vwap(sample_ohlcv)
        row = sample_ohlcv.iloc[0]
        expected = (row["high"] + row["low"] + row["close"]) / 3
        assert pytest.approx(result.iloc[0], rel=1e-6) == expected

    def test_vwap_between_high_and_low(self, sample_ohlcv):
        """VWAP should stay within the day's price range over time."""
        result = vwap(sample_ohlcv)
        # VWAP is a cumulative average so it won't always be between bar
        # high/low, but should be within overall range.
        assert result.min() >= sample_ohlcv["low"].min() - 1
        assert result.max() <= sample_ohlcv["high"].max() + 1

    def test_single_bar(self):
        df = pd.DataFrame(
            {"high": [110], "low": [90], "close": [100], "volume": [500]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = vwap(df)
        assert pytest.approx(result.iloc[0]) == 100.0  # (110+90+100)/3

    def test_zero_volume_returns_nan(self):
        """Zero cumulative volume → NaN (division by zero)."""
        df = pd.DataFrame(
            {"high": [100], "low": [90], "close": [95], "volume": [0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = vwap(df)
        assert np.isnan(result.iloc[0])

    def test_non_datetime_index_uses_cumulative(self):
        """When index isn't DatetimeIndex, VWAP doesn't reset daily."""
        df = pd.DataFrame(
            {
                "high": [110, 120],
                "low": [90, 100],
                "close": [100, 110],
                "volume": [1000, 1000],
            },
            index=[0, 1],
        )
        result = vwap(df)
        # Bar 0: tp=100, cum_tp_vol=100000, cum_vol=1000 → 100
        # Bar 1: tp=110, cum_tp_vol=210000, cum_vol=2000 → 105
        assert pytest.approx(result.iloc[1]) == 105.0


class TestOBV:
    def test_first_bar_is_zero(self, sample_ohlcv):
        result = obv(sample_ohlcv)
        assert result.iloc[0] == 0

    def test_rising_prices_add_volume(self, trending_up_ohlcv):
        result = obv(trending_up_ohlcv)
        # Every close > prev close, so OBV should monotonically increase
        diffs = result.diff().iloc[1:]
        assert (diffs > 0).all()

    def test_falling_prices_subtract_volume(self, trending_down_ohlcv):
        result = obv(trending_down_ohlcv)
        diffs = result.diff().iloc[1:]
        assert (diffs < 0).all()

    def test_flat_close_no_change(self, flat_ohlcv):
        result = obv(flat_ohlcv)
        # All closes equal → direction = 0 → OBV stays 0
        assert (result == 0).all()

    def test_known_values(self):
        df = pd.DataFrame(
            {
                "close": [10, 12, 11, 13],
                "volume": [100, 200, 150, 300],
            }
        )
        result = obv(df)
        # bar0: 0
        # bar1: +200 (12>10)
        # bar2: 200-150=50 (11<12)
        # bar3: 50+300=350 (13>11)
        expected = [0, 200, 50, 350]
        np.testing.assert_array_equal(result.values, expected)


class TestADLine:
    def test_flat_prices_return_zero(self, flat_ohlcv):
        """When high==low==close, MFM is 0 → A/D stays 0."""
        result = ad_line(flat_ohlcv)
        np.testing.assert_array_almost_equal(result.values, 0)

    def test_close_at_high_gives_positive(self):
        """Close at high → MFM = +1."""
        df = pd.DataFrame(
            {"high": [110], "low": [100], "close": [110], "volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = ad_line(df)
        # MFM = ((110-100)-(110-110)) / (110-100) = 10/10 = 1
        # MFV = 1 * 1000 = 1000
        assert pytest.approx(result.iloc[0]) == 1000.0

    def test_close_at_low_gives_negative(self):
        """Close at low → MFM = -1."""
        df = pd.DataFrame(
            {"high": [110], "low": [100], "close": [100], "volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = ad_line(df)
        # MFM = ((100-100)-(110-100)) / (110-100) = -10/10 = -1
        assert pytest.approx(result.iloc[0]) == -1000.0

    def test_close_at_midpoint_gives_zero(self):
        df = pd.DataFrame(
            {"high": [110], "low": [100], "close": [105], "volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = ad_line(df)
        # MFM = ((105-100)-(110-105)) / 10 = 0/10 = 0
        assert pytest.approx(result.iloc[0]) == 0.0

    def test_cumulative_behavior(self):
        df = pd.DataFrame(
            {
                "high":   [110, 120],
                "low":    [100, 100],
                "close":  [110, 100],
                "volume": [1000, 2000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )
        result = ad_line(df)
        # Bar 0: MFM=1, MFV=1000
        # Bar 1: MFM=((100-100)-(120-100))/20 = -1, MFV=-2000
        # Cumulative: 1000, -1000
        assert pytest.approx(result.iloc[0]) == 1000.0
        assert pytest.approx(result.iloc[1]) == -1000.0
