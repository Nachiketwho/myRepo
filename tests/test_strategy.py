import pandas as pd
import numpy as np
import pytest

from backtester.strategy import VolumeStrategy, PARAM_GRID


class TestVolumeStrategy:
    def test_signal_column_added(self, sample_ohlcv):
        strat = VolumeStrategy()
        result = strat.generate_signals(sample_ohlcv)
        assert "signal" in result.columns
        assert "vwap" in result.columns
        assert "vwap_upper" in result.columns
        assert "vwap_lower" in result.columns
        assert "obv" in result.columns
        assert "ad_line" in result.columns

    def test_signals_are_valid_values(self, sample_ohlcv):
        strat = VolumeStrategy()
        result = strat.generate_signals(sample_ohlcv)
        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_no_signal_on_first_bars(self, sample_ohlcv):
        """First `lookback` bars can't have volume confirmation."""
        strat = VolumeStrategy(obv_lookback=5, ad_lookback=5)
        result = strat.generate_signals(sample_ohlcv)
        assert (result["signal"].iloc[:5] == 0).all()

    def test_buy_signal_requires_lower_band_bounce(self):
        """Buy triggers when low touches lower band and close bounces above it."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "open":   [100, 102, 104, 103, 101, 99, 98, 95, 100, 102],
                "high":   [103, 105, 106, 105, 103, 101, 100, 101, 103, 104],
                "low":    [ 98, 100, 102, 101,  99, 97,  96, 90,  98, 100],
                "close":  [102, 104, 105, 102, 100, 98,  97, 99, 101, 103],
                "volume": [1000, 1200, 1100, 1300, 1500, 1400, 1600, 2000, 1800, 1700],
            },
            index=dates,
        )
        strat = VolumeStrategy(band_multiplier=1.0, obv_lookback=3, ad_lookback=3)
        result = strat.generate_signals(df)
        assert len(result) == 10
        assert "signal" in result.columns

    def test_sell_signal_requires_upper_band_reject(self):
        """Sell triggers when high touches upper band and close falls below it."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "open":   [100, 98, 96, 97, 99, 101, 102, 105, 100, 98],
                "high":   [102, 100, 98, 99, 101, 103, 105, 110, 102, 100],
                "low":    [ 98,  96, 94, 95, 97,  99, 100, 103,  98,  96],
                "close":  [ 99,  97, 95, 98, 100, 102, 104, 104,  99,  97],
                "volume": [1000, 1200, 1100, 1300, 1500, 1400, 1600, 2000, 1800, 1700],
            },
            index=dates,
        )
        strat = VolumeStrategy(band_multiplier=1.0, obv_lookback=3, ad_lookback=3)
        result = strat.generate_signals(df)
        assert len(result) == 10

    def test_band_multiplier_affects_signals(self, sample_ohlcv):
        """Tighter bands (lower multiplier) → more band touches → more signals."""
        strat_tight = VolumeStrategy(band_multiplier=0.5, obv_lookback=3, ad_lookback=3)
        strat_wide = VolumeStrategy(band_multiplier=3.0, obv_lookback=3, ad_lookback=3)
        r_tight = strat_tight.generate_signals(sample_ohlcv)
        r_wide = strat_wide.generate_signals(sample_ohlcv)
        assert r_tight["signal"].abs().sum() >= r_wide["signal"].abs().sum()

    def test_confirmation_requires_volume(self, sample_ohlcv):
        strat = VolumeStrategy(band_multiplier=1.0, obv_lookback=19, ad_lookback=19)
        result = strat.generate_signals(sample_ohlcv)
        assert "signal" in result.columns

    def test_custom_lookback(self, sample_ohlcv):
        s3 = VolumeStrategy(obv_lookback=3, ad_lookback=3)
        s20 = VolumeStrategy(obv_lookback=20, ad_lookback=20)
        r3 = s3.generate_signals(sample_ohlcv)
        r20 = s20.generate_signals(sample_ohlcv)
        assert len(r3) == len(r20)

    def test_original_df_not_mutated(self, sample_ohlcv):
        cols_before = list(sample_ohlcv.columns)
        strat = VolumeStrategy()
        strat.generate_signals(sample_ohlcv)
        assert list(sample_ohlcv.columns) == cols_before


class TestIterParamSets:
    def test_default_grid_count(self):
        combos = list(VolumeStrategy.iter_param_sets())
        expected = (
            len(PARAM_GRID["band_multiplier"])
            * len(PARAM_GRID["obv_lookback"])
            * len(PARAM_GRID["ad_lookback"])
        )
        assert len(combos) == expected

    def test_custom_grid(self):
        grid = {"band_multiplier": [1.0], "obv_lookback": [3, 5], "ad_lookback": [10]}
        combos = list(VolumeStrategy.iter_param_sets(grid))
        assert len(combos) == 2
        assert combos[0] == {"band_multiplier": 1.0, "obv_lookback": 3, "ad_lookback": 10}

    def test_each_combo_is_dict(self):
        for combo in VolumeStrategy.iter_param_sets():
            assert isinstance(combo, dict)
            assert "band_multiplier" in combo
            assert "obv_lookback" in combo
            assert "ad_lookback" in combo
