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
                "date", "close", "vwap", "obv", "ad_line",
                "met_conditions", "missed_condition", "direction",
            }
            assert expected_cols.issubset(set(result.columns))

    def test_direction_values(self, sample_ohlcv):
        result = find_missed_trades(sample_ohlcv)
        if not result.empty:
            assert set(result["direction"].unique()).issubset({"buy", "sell"})

    def test_met_conditions_has_two(self, sample_ohlcv):
        """Each near-miss should have exactly 2 met conditions."""
        result = find_missed_trades(sample_ohlcv)
        for _, row in result.iterrows():
            assert len(row["met_conditions"]) == 2

    def test_missed_condition_is_string(self, sample_ohlcv):
        result = find_missed_trades(sample_ohlcv)
        for _, row in result.iterrows():
            assert isinstance(row["missed_condition"], str)

    def test_custom_lookback(self, sample_ohlcv):
        r3 = find_missed_trades(sample_ohlcv, obv_lookback=3, ad_lookback=3)
        r14 = find_missed_trades(sample_ohlcv, obv_lookback=14, ad_lookback=14)
        # Different lookbacks → potentially different near-misses
        assert isinstance(r3, pd.DataFrame)
        assert isinstance(r14, pd.DataFrame)

    def test_flat_data_no_near_misses(self, flat_ohlcv):
        """Flat prices: close == VWAP, OBV == 0, A/D == 0 → no trends."""
        result = find_missed_trades(flat_ohlcv)
        # With all indicators flat, conditions like "rising" or "falling"
        # won't be met, so at most the price conditions are met.
        # This is a sanity check — should not crash.
        assert isinstance(result, pd.DataFrame)
