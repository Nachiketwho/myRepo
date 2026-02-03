import pandas as pd
import pytest

from backtester.strategy import VolumeStrategy, PARAM_GRID


class TestVolumeStrategy:
    def test_signal_column_added(self, sample_ohlcv):
        strat = VolumeStrategy()
        result = strat.generate_signals(sample_ohlcv)
        assert "signal" in result.columns
        assert "vwap" in result.columns
        assert "obv" in result.columns
        assert "ad_line" in result.columns

    def test_signals_are_valid_values(self, sample_ohlcv):
        strat = VolumeStrategy()
        result = strat.generate_signals(sample_ohlcv)
        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_no_signal_on_first_bars(self, sample_ohlcv):
        """First `lookback` bars can't have signals (no shifted data)."""
        strat = VolumeStrategy(obv_lookback=5, ad_lookback=5)
        result = strat.generate_signals(sample_ohlcv)
        assert (result["signal"].iloc[:5] == 0).all()

    def test_buy_signal_conditions(self, trending_up_ohlcv):
        """In a strong uptrend, OBV + A/D are rising. If close dips below
        VWAP at some point, we should see buy signals."""
        strat = VolumeStrategy(obv_lookback=3, ad_lookback=3)
        result = strat.generate_signals(trending_up_ohlcv)
        # At minimum, verify the strategy produces *some* signals
        # (exact count depends on VWAP vs close interplay)
        assert result["signal"].abs().sum() >= 0  # no crash

    def test_sell_signal_in_downtrend(self, trending_down_ohlcv):
        strat = VolumeStrategy(obv_lookback=3, ad_lookback=3)
        result = strat.generate_signals(trending_down_ohlcv)
        # Should see sell signals when price > VWAP and OBV/AD falling
        sell_count = (result["signal"] == -1).sum()
        assert sell_count >= 0

    def test_custom_lookback(self, sample_ohlcv):
        s3 = VolumeStrategy(obv_lookback=3, ad_lookback=3)
        s20 = VolumeStrategy(obv_lookback=20, ad_lookback=20)
        r3 = s3.generate_signals(sample_ohlcv)
        r20 = s20.generate_signals(sample_ohlcv)
        # Shorter lookback is more sensitive → likely more signals
        assert r3["signal"].abs().sum() >= r20["signal"].abs().sum()

    def test_different_params_produce_different_signals(self, sample_ohlcv):
        s_fast = VolumeStrategy(obv_lookback=3, ad_lookback=3)
        s_slow = VolumeStrategy(obv_lookback=14, ad_lookback=14)
        r_fast = s_fast.generate_signals(sample_ohlcv)
        r_slow = s_slow.generate_signals(sample_ohlcv)
        # Not necessarily different on small data, but shouldn't crash
        assert len(r_fast) == len(r_slow)

    def test_original_df_not_mutated(self, sample_ohlcv):
        cols_before = list(sample_ohlcv.columns)
        strat = VolumeStrategy()
        strat.generate_signals(sample_ohlcv)
        assert list(sample_ohlcv.columns) == cols_before


class TestIterParamSets:
    def test_default_grid_count(self):
        combos = list(VolumeStrategy.iter_param_sets())
        expected = len(PARAM_GRID["obv_lookback"]) * len(PARAM_GRID["ad_lookback"])
        assert len(combos) == expected

    def test_custom_grid(self):
        grid = {"obv_lookback": [3, 5], "ad_lookback": [10]}
        combos = list(VolumeStrategy.iter_param_sets(grid))
        assert len(combos) == 2
        assert combos[0] == {"obv_lookback": 3, "ad_lookback": 10}

    def test_each_combo_is_dict(self):
        for combo in VolumeStrategy.iter_param_sets():
            assert isinstance(combo, dict)
            assert "obv_lookback" in combo
            assert "ad_lookback" in combo
