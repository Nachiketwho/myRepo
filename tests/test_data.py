"""Tests for data fetching module — uses mocking to avoid real API calls."""

import sys
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd
import pytest

from backtester.data import TIMEFRAMES, PERIOD_DEFAULTS


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
