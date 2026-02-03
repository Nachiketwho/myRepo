import numpy as np
import pandas as pd
import pytest

from backtester.indicators import vwap, vwap_bands, obv, ad_line


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
        result = vwap(sample_ohlcv)
        assert result.min() >= sample_ohlcv["low"].min() - 1
        assert result.max() <= sample_ohlcv["high"].max() + 1

    def test_single_bar(self):
        df = pd.DataFrame(
            {"high": [110], "low": [90], "close": [100], "volume": [500]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = vwap(df)
        assert pytest.approx(result.iloc[0]) == 100.0

    def test_zero_volume_returns_nan(self):
        df = pd.DataFrame(
            {"high": [100], "low": [90], "close": [95], "volume": [0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = vwap(df)
        assert np.isnan(result.iloc[0])

    def test_non_datetime_index_uses_cumulative(self):
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
        assert pytest.approx(result.iloc[1]) == 105.0


class TestVWAPBands:
    def test_returns_five_series(self, sample_ohlcv):
        vwap_line, ui, uo, li, lo = vwap_bands(sample_ohlcv)
        assert isinstance(vwap_line, pd.Series)
        assert isinstance(ui, pd.Series)
        assert isinstance(uo, pd.Series)
        assert isinstance(li, pd.Series)
        assert isinstance(lo, pd.Series)
        assert len(vwap_line) == len(sample_ohlcv)

    def test_outer_wider_than_inner(self, sample_ohlcv):
        vwap_line, ui, uo, li, lo = vwap_bands(sample_ohlcv)
        # Outer upper >= inner upper, outer lower <= inner lower
        assert (uo.iloc[1:] >= ui.iloc[1:] - 1e-9).all()
        assert (lo.iloc[1:] <= li.iloc[1:] + 1e-9).all()

    def test_upper_above_vwap(self, sample_ohlcv):
        vwap_line, ui, uo, li, lo = vwap_bands(sample_ohlcv)
        assert (ui.iloc[1:] >= vwap_line.iloc[1:] - 1e-9).all()
        assert (uo.iloc[1:] >= vwap_line.iloc[1:] - 1e-9).all()

    def test_lower_below_vwap(self, sample_ohlcv):
        vwap_line, ui, uo, li, lo = vwap_bands(sample_ohlcv)
        assert (li.iloc[1:] <= vwap_line.iloc[1:] + 1e-9).all()
        assert (lo.iloc[1:] <= vwap_line.iloc[1:] + 1e-9).all()

    def test_first_bar_bands_equal_vwap(self):
        """First bar: std=0, so all bands == vwap."""
        df = pd.DataFrame(
            {"high": [110], "low": [90], "close": [100], "volume": [500]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        vwap_line, ui, uo, li, lo = vwap_bands(df)
        assert pytest.approx(ui.iloc[0]) == vwap_line.iloc[0]
        assert pytest.approx(uo.iloc[0]) == vwap_line.iloc[0]
        assert pytest.approx(li.iloc[0]) == vwap_line.iloc[0]
        assert pytest.approx(lo.iloc[0]) == vwap_line.iloc[0]

    def test_multiplier_widens_bands(self, sample_ohlcv):
        _, ui1, _, li1, _ = vwap_bands(sample_ohlcv, multiplier_inner=1.0, multiplier_outer=2.0)
        _, ui2, _, li2, _ = vwap_bands(sample_ohlcv, multiplier_inner=2.0, multiplier_outer=3.0)
        assert (ui2.iloc[1:] >= ui1.iloc[1:] - 1e-9).all()
        assert (li2.iloc[1:] <= li1.iloc[1:] + 1e-9).all()

    def test_flat_prices_zero_std(self, flat_ohlcv):
        """Flat prices → std=0 → all bands collapse to vwap."""
        vwap_line, ui, uo, li, lo = vwap_bands(flat_ohlcv)
        np.testing.assert_array_almost_equal(ui.values, vwap_line.values)
        np.testing.assert_array_almost_equal(uo.values, vwap_line.values)
        np.testing.assert_array_almost_equal(li.values, vwap_line.values)
        np.testing.assert_array_almost_equal(lo.values, vwap_line.values)

    def test_non_datetime_index(self):
        df = pd.DataFrame(
            {
                "high": [110, 120, 115],
                "low": [90, 100, 95],
                "close": [100, 110, 105],
                "volume": [1000, 1000, 1000],
            },
            index=[0, 1, 2],
        )
        vwap_line, ui, uo, li, lo = vwap_bands(df, multiplier_inner=1.0, multiplier_outer=2.0)
        assert len(vwap_line) == 3
        assert uo.iloc[1] > lo.iloc[1]

    def test_inner_and_outer_different(self, sample_ohlcv):
        """Inner and outer bands should differ when multipliers differ."""
        vwap_line, ui, uo, li, lo = vwap_bands(
            sample_ohlcv, multiplier_inner=1.0, multiplier_outer=2.0
        )
        # After first bar (where std=0), they should differ
        assert not (ui.iloc[2:] == uo.iloc[2:]).all()
        assert not (li.iloc[2:] == lo.iloc[2:]).all()


class TestOBV:
    def test_first_bar_is_zero(self, sample_ohlcv):
        result = obv(sample_ohlcv)
        assert result.iloc[0] == 0

    def test_rising_prices_add_volume(self, trending_up_ohlcv):
        result = obv(trending_up_ohlcv)
        diffs = result.diff().iloc[1:]
        assert (diffs > 0).all()

    def test_falling_prices_subtract_volume(self, trending_down_ohlcv):
        result = obv(trending_down_ohlcv)
        diffs = result.diff().iloc[1:]
        assert (diffs < 0).all()

    def test_flat_close_no_change(self, flat_ohlcv):
        result = obv(flat_ohlcv)
        assert (result == 0).all()

    def test_known_values(self):
        df = pd.DataFrame(
            {
                "close": [10, 12, 11, 13],
                "volume": [100, 200, 150, 300],
            }
        )
        result = obv(df)
        expected = [0, 200, 50, 350]
        np.testing.assert_array_equal(result.values, expected)


class TestADLine:
    def test_flat_prices_return_zero(self, flat_ohlcv):
        result = ad_line(flat_ohlcv)
        np.testing.assert_array_almost_equal(result.values, 0)

    def test_close_at_high_gives_positive(self):
        df = pd.DataFrame(
            {"high": [110], "low": [100], "close": [110], "volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = ad_line(df)
        assert pytest.approx(result.iloc[0]) == 1000.0

    def test_close_at_low_gives_negative(self):
        df = pd.DataFrame(
            {"high": [110], "low": [100], "close": [100], "volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = ad_line(df)
        assert pytest.approx(result.iloc[0]) == -1000.0

    def test_close_at_midpoint_gives_zero(self):
        df = pd.DataFrame(
            {"high": [110], "low": [100], "close": [105], "volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = ad_line(df)
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
        assert pytest.approx(result.iloc[0]) == 1000.0
        assert pytest.approx(result.iloc[1]) == -1000.0
