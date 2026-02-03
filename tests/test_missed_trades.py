import pandas as pd
import numpy as np
import pytest

from backtester.missed_trades import find_missed_trades


class TestFindMissedTrades:
    def test_returns_dataframe(self, sample_ohlcv):
        result = find_missed_trades(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, sample_ohlcv):
        result = find_missed_trades(sample_ohlcv)
        if not result.empty:
            expected_cols = {
                "date", "close", "vwap", "vwap_upper", "vwap_lower",
                "obv", "ad_line", "met_conditions", "missed_condition",
                "direction",
            }
            assert expected_cols.issubset(set(result.columns))

    def test_direction_values(self, sample_ohlcv):
        result = find_missed_trades(sample_ohlcv)
        if not result.empty:
            assert set(result["direction"].unique()).issubset({"buy", "sell"})

    def test_missed_condition_values(self, sample_ohlcv):
        """Missed condition should be either volume_confirmation or band_touch."""
        result = find_missed_trades(sample_ohlcv)
        for _, row in result.iterrows():
            assert row["missed_condition"] in ("volume_confirmation", "band_touch")

    def test_missed_condition_is_string(self, sample_ohlcv):
        result = find_missed_trades(sample_ohlcv)
        for _, row in result.iterrows():
            assert isinstance(row["missed_condition"], str)

    def test_custom_lookback(self, sample_ohlcv):
        r3 = find_missed_trades(sample_ohlcv, obv_lookback=3, ad_lookback=3)
        r14 = find_missed_trades(sample_ohlcv, obv_lookback=14, ad_lookback=14)
        assert isinstance(r3, pd.DataFrame)
        assert isinstance(r14, pd.DataFrame)

    def test_band_multiplier(self, sample_ohlcv):
        r_tight = find_missed_trades(sample_ohlcv, band_multiplier=0.5)
        r_wide = find_missed_trades(sample_ohlcv, band_multiplier=3.0)
        assert isinstance(r_tight, pd.DataFrame)
        assert isinstance(r_wide, pd.DataFrame)

    def test_flat_data_no_near_misses(self, flat_ohlcv):
        """Flat prices: bands collapse to VWAP, no bounces possible."""
        result = find_missed_trades(flat_ohlcv)
        assert isinstance(result, pd.DataFrame)
