"""Tests for data fetching module — uses mocking to avoid real API calls."""

import sys
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
import pandas as pd
import pytest

from backtester.data import TIMEFRAMES, PERIOD_DEFAULTS, validate_ohlcv, DataValidationError


def _mock_history(**kwargs):
    """Return a small DataFrame mimicking yfinance output."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [95, 96, 97, 98, 99],
            "Close": [102, 103, 104, 105, 106],
            "Volume": [1000, 1100, 1200, 1300, 1400],
            "Dividends": [0, 0, 0, 0, 0],
            "Stock Splits": [0, 0, 0, 0, 0],
        },
        index=dates,
    )


@pytest.fixture
def mock_yfinance():
    """Patch yfinance so fetch_nifty50 never hits the network."""
    mock_yf = MagicMock()
    mock_ticker = MagicMock()
    mock_ticker.history = _mock_history
    mock_yf.Ticker.return_value = mock_ticker
    with patch.dict(sys.modules, {"yfinance": mock_yf}):
        yield mock_yf, mock_ticker


class TestFetchNifty50:
    def test_returns_ohlcv_columns(self, mock_yfinance):
        from backtester.data import fetch_nifty50
        df = fetch_nifty50(timeframe="daily")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_index_name(self, mock_yfinance):
        from backtester.data import fetch_nifty50
        df = fetch_nifty50()
        assert df.index.name == "date"

    def test_timeframe_mapping(self, mock_yfinance):
        from backtester.data import fetch_nifty50
        for name in TIMEFRAMES:
            df = fetch_nifty50(timeframe=name)
            assert len(df) == 5

    def test_custom_period(self, mock_yfinance):
        from backtester.data import fetch_nifty50
        df = fetch_nifty50(period="1y")
        assert len(df) == 5

    def test_start_end(self, mock_yfinance):
        from backtester.data import fetch_nifty50
        df = fetch_nifty50(start="2024-01-01", end="2024-06-01")
        assert len(df) == 5

    def test_empty_data_raises(self, mock_yfinance):
        mock_yf, mock_ticker = mock_yfinance
        mock_ticker.history = MagicMock(return_value=pd.DataFrame())

        from backtester.data import fetch_nifty50
        with pytest.raises(ValueError, match="No data returned"):
            fetch_nifty50()

    def test_period_defaults_exist(self):
        for tf in TIMEFRAMES:
            assert tf in PERIOD_DEFAULTS


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def _valid_df(n=20):
    """Build a valid Nifty50-like OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = [22000 + i * 10 for i in range(n)]
    return pd.DataFrame({
        "open": [c - 5 for c in closes],
        "high": [c + 10 for c in closes],
        "low": [c - 10 for c in closes],
        "close": closes,
        "volume": [1000] * n,
    }, index=dates)


class TestValidateOHLCV:
    def test_valid_data_no_warnings(self):
        warnings = validate_ohlcv(_valid_df())
        # May have freshness warning depending on current date
        critical = [w for w in warnings if "stale" not in w]
        assert len(critical) == 0

    def test_empty_raises(self):
        with pytest.raises(DataValidationError, match="empty"):
            validate_ohlcv(pd.DataFrame())

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"open": [1], "close": [1]})
        with pytest.raises(DataValidationError, match="missing columns"):
            validate_ohlcv(df)

    def test_zero_close_raises(self):
        df = _valid_df()
        df.iloc[5, df.columns.get_loc("close")] = 0
        with pytest.raises(DataValidationError, match="close <= 0"):
            validate_ohlcv(df)

    def test_nan_warning(self):
        df = _valid_df()
        df.iloc[0, df.columns.get_loc("close")] = np.nan
        warnings = validate_ohlcv(df)
        assert any("NaN" in w for w in warnings)

    def test_too_many_nans_raises(self):
        df = _valid_df(10)
        # Set >10% of OHLC cells to NaN
        for col in ["open", "high", "low", "close"]:
            df.iloc[:5, df.columns.get_loc(col)] = np.nan
        with pytest.raises(DataValidationError, match="too many NaN"):
            validate_ohlcv(df)

    def test_high_below_low_warning(self):
        df = _valid_df()
        df.iloc[3, df.columns.get_loc("high")] = df.iloc[3]["low"] - 10
        warnings = validate_ohlcv(df)
        assert any("high < low" in w for w in warnings)

    def test_unusual_price_range_warning(self):
        df = _valid_df()
        df["close"] = 100  # way below Nifty50 range
        df["open"] = 100
        df["high"] = 110
        df["low"] = 90
        warnings = validate_ohlcv(df)
        assert any("unusual for Nifty50" in w for w in warnings)

    def test_zero_volume_warning(self):
        df = _valid_df()
        df["volume"] = 0
        warnings = validate_ohlcv(df)
        assert any("zero volume" in w for w in warnings)

    def test_duplicate_index_warning(self):
        df = _valid_df(5)
        df.index = pd.DatetimeIndex(["2024-01-01"] * 5)
        warnings = validate_ohlcv(df)
        assert any("duplicate" in w for w in warnings)
