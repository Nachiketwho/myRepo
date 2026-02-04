"""Tests for phantom trade detection."""

import datetime as dt
import pandas as pd
import pytest

from backtester.phantom_trades import find_phantom_trades, _col_check


@pytest.fixture
def vix_series():
    """VIX series at 13% for 90 days."""
    dates = pd.date_range("2024-01-01", periods=90, freq="B")
    return pd.Series(13.0, index=dates.date)


@pytest.fixture
def signals_with_move():
    """30-bar data: no signals, but a big move up from bar 5 to bar 10."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    closes = [22000.0] * 30
    # Inject a 100-point move up starting at bar 6
    for i in range(6, 16):
        closes[i] = 22000.0 + (i - 5) * 20  # up to 22200
    df = pd.DataFrame({
        "open":   closes,
        "high":   [c + 10 for c in closes],
        "low":    [c - 10 for c in closes],
        "close":  closes,
        "volume": [1000] * 30,
        "signal": [0] * 30,  # no signals
    }, index=dates)
    return df


@pytest.fixture
def signals_with_signal():
    """30-bar data with a signal on bar 5 (should NOT be a phantom)."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    closes = [22000.0] * 30
    for i in range(6, 16):
        closes[i] = 22000.0 + (i - 5) * 20
    df = pd.DataFrame({
        "open":   closes,
        "high":   [c + 10 for c in closes],
        "low":    [c - 10 for c in closes],
        "close":  closes,
        "volume": [1000] * 30,
        "signal": [0] * 30,
    }, index=dates)
    df.iloc[5, df.columns.get_loc("signal")] = 1  # has signal
    return df


class TestFindPhantomTrades:
    def test_returns_dataframe(self, signals_with_move, vix_series):
        result = find_phantom_trades(signals_with_move, vix_series)
        assert isinstance(result, pd.DataFrame)

    def test_detects_phantom_on_big_move(self, signals_with_move, vix_series):
        result = find_phantom_trades(
            signals_with_move, vix_series, spot_move_threshold=50.0,
        )
        # Should detect at least one phantom trade (bar 5 has no signal
        # but bars 6-15 have significant upward move)
        assert len(result) > 0

    def test_no_phantoms_below_threshold(self, signals_with_move, vix_series):
        result = find_phantom_trades(
            signals_with_move, vix_series, spot_move_threshold=500.0,
        )
        # Move is only ~200 points, threshold=500 means no phantoms
        assert len(result) == 0

    def test_signal_bars_not_phantom(self, signals_with_signal, vix_series):
        result = find_phantom_trades(
            signals_with_signal, vix_series, spot_move_threshold=50.0,
        )
        # Bar 5 has signal=1, should not appear as phantom
        if not result.empty:
            # Check no phantom at bar 5's date
            bar5_date = signals_with_signal.index[5]
            assert bar5_date not in result["date"].values

    def test_phantom_columns(self, signals_with_move, vix_series):
        result = find_phantom_trades(
            signals_with_move, vix_series, spot_move_threshold=50.0,
        )
        if not result.empty:
            expected_cols = [
                "date", "spot", "direction", "max_spot_move",
                "peak_bar", "conditions_met", "conditions_missing",
                "option_type", "strike", "expiry",
                "premium_entry", "premium_at_peak", "premium_pnl",
            ]
            for col in expected_cols:
                assert col in result.columns, f"Missing column: {col}"

    def test_direction_field(self, signals_with_move, vix_series):
        result = find_phantom_trades(
            signals_with_move, vix_series, spot_move_threshold=50.0,
        )
        if not result.empty:
            assert all(d in ("bullish", "bearish") for d in result["direction"])

    def test_custom_look_ahead(self, signals_with_move, vix_series):
        r1 = find_phantom_trades(
            signals_with_move, vix_series,
            spot_move_threshold=50.0, look_ahead=5,
        )
        r2 = find_phantom_trades(
            signals_with_move, vix_series,
            spot_move_threshold=50.0, look_ahead=15,
        )
        # Longer look_ahead may find more or same phantoms
        assert isinstance(r1, pd.DataFrame)
        assert isinstance(r2, pd.DataFrame)


class TestColCheck:
    def test_existing_column(self):
        row = pd.Series({"a": 1.0, "b": 2.0})
        assert _col_check(row, "a") is True

    def test_missing_column(self):
        row = pd.Series({"a": 1.0})
        assert _col_check(row, "b") is False

    def test_nan_value(self):
        row = pd.Series({"a": float("nan")})
        assert _col_check(row, "a") is False
