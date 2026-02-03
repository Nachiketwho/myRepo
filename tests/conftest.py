"""Shared fixtures — realistic Nifty50-style OHLCV data for testing."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv():
    """20-bar daily OHLCV with known values for hand-checked indicator math.

    Prices trend up then reverse, with varying volume to trigger different
    indicator states.
    """
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    data = {
        "open":   [100, 102, 104, 103, 106, 108, 110, 109, 112, 115,
                    114, 112, 110, 108, 106, 104, 103, 101, 100, 98],
        "high":   [103, 105, 106, 107, 109, 111, 113, 112, 116, 117,
                    115, 113, 111, 109, 107, 106, 104, 102, 101, 99],
        "low":    [ 99, 101, 102, 102, 105, 107, 108, 107, 111, 113,
                    112, 110, 108, 106, 104, 102, 101,  99,  97, 96],
        "close":  [102, 104, 105, 106, 108, 110, 112, 111, 115, 116,
                    113, 111, 109, 107, 105, 103, 102, 100,  98, 97],
        "volume": [1000, 1200, 1100, 1300, 1500, 1400, 1600, 1100, 1800, 2000,
                    1700, 1900, 1500, 1300, 1200, 1000, 1100, 1400, 1600, 1800],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def flat_ohlcv():
    """10-bar data where high == low == close (edge case for A/D Line)."""
    dates = pd.date_range("2024-06-01", periods=10, freq="B")
    price = [100] * 10
    return pd.DataFrame(
        {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": [500] * 10,
        },
        index=dates,
    )


@pytest.fixture
def trending_up_ohlcv():
    """30-bar steadily rising data — every close > previous close."""
    dates = pd.date_range("2024-03-01", periods=30, freq="B")
    closes = [100 + i * 2 for i in range(30)]
    return pd.DataFrame(
        {
            "open":   [c - 1 for c in closes],
            "high":   [c + 1 for c in closes],
            "low":    [c - 2 for c in closes],
            "close":  closes,
            "volume": [1000 + i * 50 for i in range(30)],
        },
        index=dates,
    )


@pytest.fixture
def trending_down_ohlcv():
    """30-bar steadily falling data — every close < previous close."""
    dates = pd.date_range("2024-03-01", periods=30, freq="B")
    closes = [200 - i * 2 for i in range(30)]
    return pd.DataFrame(
        {
            "open":   [c + 1 for c in closes],
            "high":   [c + 2 for c in closes],
            "low":    [c - 1 for c in closes],
            "close":  closes,
            "volume": [1000 + i * 50 for i in range(30)],
        },
        index=dates,
    )


@pytest.fixture
def signals_with_buy(sample_ohlcv):
    """sample_ohlcv with a forced buy signal on bar 5 and sell on bar 15."""
    df = sample_ohlcv.copy()
    df["signal"] = 0
    df.iloc[5, df.columns.get_loc("signal")] = 1
    df.iloc[15, df.columns.get_loc("signal")] = -1
    return df
