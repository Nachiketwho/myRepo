import pandas as pd
import pytest

from backtester.optimizer import run_optimization, best_params, _build_combos


class TestBuildCombos:
    def test_counts(self):
        sg = {"obv_lookback": [3, 5], "ad_lookback": [3, 5]}
        eg = {"sl_pct": [2.0], "tp_pct": [3.0]}
        combos = _build_combos(sg, eg)
        assert len(combos) == 4  # 2*2*1*1

    def test_keys_present(self):
        sg = {"obv_lookback": [5]}
        eg = {"sl_pct": [2.0], "tsl_pct": [1.5], "tp_pct": [3.0], "ttp_pct": [1.0]}
        combos = _build_combos(sg, eg)
        for c in combos:
            assert "obv_lookback" in c
            assert "sl_pct" in c


class TestRunOptimization:
    def test_returns_dataframe(self, sample_ohlcv):
        sg = {"obv_lookback": [3, 5], "ad_lookback": [3]}
        eg = {"sl_pct": [2.0], "tsl_pct": [1.5], "tp_pct": [5.0], "ttp_pct": [1.0]}
        result = run_optimization(sample_ohlcv, sg, eg, min_trades=0)
        assert isinstance(result, pd.DataFrame)

    def test_sorted_by_pnl(self, sample_ohlcv):
        sg = {"obv_lookback": [3, 5], "ad_lookback": [3, 5]}
        eg = {"sl_pct": [2.0, 5.0], "tsl_pct": [1.5], "tp_pct": [3.0, 5.0], "ttp_pct": [1.0]}
        result = run_optimization(sample_ohlcv, sg, eg, min_trades=0)
        if len(result) > 1:
            assert result["total_pnl"].iloc[0] >= result["total_pnl"].iloc[1]

    def test_min_trades_filter(self, sample_ohlcv):
        sg = {"obv_lookback": [3]}
        eg = {"sl_pct": [2.0], "tsl_pct": [1.5], "tp_pct": [3.0], "ttp_pct": [1.0]}
        result = run_optimization(sample_ohlcv, sg, eg, min_trades=9999)
        assert len(result) == 0

    def test_has_expected_columns(self, sample_ohlcv):
        sg = {"obv_lookback": [3]}
        eg = {"sl_pct": [2.0], "tsl_pct": [1.5], "tp_pct": [5.0], "ttp_pct": [1.0]}
        result = run_optimization(sample_ohlcv, sg, eg, min_trades=0)
        if not result.empty:
            for col in ["num_trades", "total_pnl", "win_rate", "profit_factor", "max_drawdown"]:
                assert col in result.columns


class TestBestParams:
    def test_returns_dict(self, sample_ohlcv):
        sg = {"obv_lookback": [3]}
        eg = {"sl_pct": [2.0], "tsl_pct": [1.5], "tp_pct": [5.0], "ttp_pct": [1.0]}
        result = best_params(sample_ohlcv, sg, eg, min_trades=0)
        assert isinstance(result, dict)

    def test_empty_when_no_trades(self, sample_ohlcv):
        sg = {"obv_lookback": [3]}
        eg = {"sl_pct": [2.0], "tsl_pct": [1.5], "tp_pct": [5.0], "ttp_pct": [1.0]}
        result = best_params(sample_ohlcv, sg, eg, min_trades=9999)
        assert result == {}
