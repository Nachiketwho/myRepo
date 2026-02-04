"""Tests for F&O parameter optimizer."""

import datetime as dt
import pandas as pd
import pytest

from backtester.fno_optimizer import (
    run_fno_optimization, best_fno_params,
    _build_fno_combos, FNO_PARAM_GRID,
    FNO_SL_GRID, FNO_RR_GRID, FNO_EMA_GRID, FNO_STRENGTH_GRID,
)


@pytest.fixture
def vix_series():
    """VIX series at 13% for 90 days."""
    dates = pd.date_range("2024-01-01", periods=90, freq="B")
    return pd.Series(13.0, index=dates.date)


@pytest.fixture
def signals_with_trades():
    """40-bar data with a few signals (minimal for optimizer testing)."""
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    df = pd.DataFrame({
        "open":   [22000 + i * 5 for i in range(40)],
        "high":   [22010 + i * 5 for i in range(40)],
        "low":    [21990 + i * 5 for i in range(40)],
        "close":  [22000 + i * 5 for i in range(40)],
        "volume": [1000] * 40,
        "signal": [0] * 40,
    }, index=dates)
    df.iloc[3, df.columns.get_loc("signal")] = 1
    df.iloc[15, df.columns.get_loc("signal")] = -1
    df.iloc[30, df.columns.get_loc("signal")] = 1
    return df


class TestBuildFnoCombos:
    def test_default_grid_count(self):
        combos = _build_fno_combos()
        expected = (
            len(FNO_SL_GRID)
            * len(FNO_RR_GRID)
            * len(FNO_EMA_GRID)
            * len(FNO_STRENGTH_GRID)
        )
        assert len(combos) == expected

    def test_custom_grid(self):
        grid = {"base_sl": [20, 30], "rr_ratio": [1.0, 2.0]}
        combos = _build_fno_combos(grid)
        assert len(combos) == 4

    def test_combo_has_all_keys(self):
        combos = _build_fno_combos()
        for combo in combos:
            assert "base_sl" in combo
            assert "rr_ratio" in combo
            assert "ema_period" in combo
            assert "min_signal_strength" in combo


class TestRunFnoOptimization:
    def test_returns_dataframe(self, signals_with_trades, vix_series):
        # Use tiny grid for speed
        grid = {"base_sl": [30], "rr_ratio": [2.0]}
        result = run_fno_optimization(
            signals_with_trades, vix_series,
            param_grid=grid, num_otm=0, num_expiries=1,
            min_trades=1, min_signal_strength=0,
        )
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self, signals_with_trades, vix_series):
        grid = {"base_sl": [30], "rr_ratio": [2.0]}
        result = run_fno_optimization(
            signals_with_trades, vix_series,
            param_grid=grid, num_otm=0, num_expiries=1,
            min_trades=1, min_signal_strength=0,
        )
        if not result.empty:
            assert "total_pnl" in result.columns
            assert "total_trades" in result.columns
            assert "avg_win_rate" in result.columns

    def test_sorted_by_pnl(self, signals_with_trades, vix_series):
        grid = {"base_sl": [20, 40], "rr_ratio": [1.0, 2.0]}
        result = run_fno_optimization(
            signals_with_trades, vix_series,
            param_grid=grid, num_otm=0, num_expiries=1,
            min_trades=1, min_signal_strength=0,
        )
        if len(result) > 1:
            assert result["total_pnl"].iloc[0] >= result["total_pnl"].iloc[1]

    def test_no_signals_empty(self, vix_series):
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        df = pd.DataFrame({
            "open": [22000] * 20, "high": [22050] * 20,
            "low": [21950] * 20, "close": [22000] * 20,
            "volume": [1000] * 20, "signal": [0] * 20,
        }, index=dates)
        grid = {"base_sl": [30]}
        result = run_fno_optimization(
            df, vix_series, param_grid=grid,
            num_otm=0, num_expiries=1, min_signal_strength=0,
        )
        assert result.empty

    def test_min_trades_filter(self, signals_with_trades, vix_series):
        grid = {"base_sl": [30], "rr_ratio": [2.0]}
        result = run_fno_optimization(
            signals_with_trades, vix_series,
            param_grid=grid, num_otm=0, num_expiries=1,
            min_trades=999, min_signal_strength=0,
        )
        assert result.empty


class TestBestFnoParams:
    def test_returns_dict(self, signals_with_trades, vix_series):
        grid = {"base_sl": [30], "rr_ratio": [2.0]}
        result = best_fno_params(
            signals_with_trades, vix_series,
            param_grid=grid, num_otm=0, num_expiries=1,
            min_trades=1, min_signal_strength=0,
        )
        assert isinstance(result, dict)

    def test_empty_when_no_trades(self, vix_series):
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        df = pd.DataFrame({
            "open": [22000] * 20, "high": [22050] * 20,
            "low": [21950] * 20, "close": [22000] * 20,
            "volume": [1000] * 20, "signal": [0] * 20,
        }, index=dates)
        grid = {"base_sl": [30]}
        result = best_fno_params(
            df, vix_series, param_grid=grid,
            num_otm=0, num_expiries=1, min_signal_strength=0,
        )
        assert result == {}


class TestFnoParamGrid:
    def test_sl_grid_sorted(self):
        assert FNO_SL_GRID == sorted(FNO_SL_GRID)

    def test_rr_grid_sorted(self):
        assert FNO_RR_GRID == sorted(FNO_RR_GRID)

    def test_ema_grid_sorted(self):
        assert FNO_EMA_GRID == sorted(FNO_EMA_GRID)

    def test_strength_grid_sorted(self):
        assert FNO_STRENGTH_GRID == sorted(FNO_STRENGTH_GRID)

    def test_param_grid_keys(self):
        assert set(FNO_PARAM_GRID.keys()) == {
            "base_sl", "rr_ratio", "ema_period", "min_signal_strength",
        }
