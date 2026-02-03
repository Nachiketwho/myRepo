import pandas as pd
import numpy as np


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price.

    Expects columns: high, low, close, volume.
    Resets each trading day (uses date from index).
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_volume = typical_price * df["volume"]

    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
        cum_tp_vol = tp_volume.groupby(dates).cumsum()
        cum_vol = df["volume"].groupby(dates).cumsum()
    else:
        cum_tp_vol = tp_volume.cumsum()
        cum_vol = df["volume"].cumsum()

    return cum_tp_vol / cum_vol


def vwap_bands(
    df: pd.DataFrame,
    multiplier_inner: float = 1.0,
    multiplier_outer: float = 2.0,
    rolling_window: int | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """VWAP with dual upper and dual lower standard-deviation bands.

    Inner bands (1st level support/resistance) — more frequent touches.
    Outer bands (2nd level support/resistance) — stronger S/R zones.

    Args:
        rolling_window: When set, uses a rolling window instead of daily
            reset for the VWAP and std calculation.  Recommended for
            intraday timeframes (e.g. 20 for 15min ≈ 5h lookback).
            When None, uses daily-reset cumulative calculation.

    Returns:
        (vwap_line, upper_inner, upper_outer, lower_inner, lower_outer)
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, 1)  # guard against zero-volume bars
    tp_volume = typical_price * vol

    if rolling_window is not None:
        # Rolling VWAP — consistent band width, ideal for intraday
        rolling_tp_vol = tp_volume.rolling(rolling_window, min_periods=1).sum()
        rolling_vol = vol.rolling(rolling_window, min_periods=1).sum()
        vwap_line = rolling_tp_vol / rolling_vol

        sq_diff_vol = ((typical_price - vwap_line) ** 2) * vol
        rolling_sq_diff_vol = sq_diff_vol.rolling(rolling_window, min_periods=1).sum()
        variance = rolling_sq_diff_vol / rolling_vol
    elif isinstance(df.index, pd.DatetimeIndex):
        # Daily-reset cumulative VWAP — standard for daily/weekly
        dates = df.index.date
        cum_tp_vol = tp_volume.groupby(dates).cumsum()
        cum_vol = vol.groupby(dates).cumsum()
        vwap_line = cum_tp_vol / cum_vol
        sq_diff_vol = ((typical_price - vwap_line) ** 2) * vol
        cum_sq_diff_vol = sq_diff_vol.groupby(dates).cumsum()
        variance = cum_sq_diff_vol / cum_vol
    else:
        # Global cumulative — for non-datetime indices
        cum_tp_vol = tp_volume.cumsum()
        cum_vol = vol.cumsum()
        vwap_line = cum_tp_vol / cum_vol
        sq_diff_vol = ((typical_price - vwap_line) ** 2) * vol
        cum_sq_diff_vol = sq_diff_vol.cumsum()
        variance = cum_sq_diff_vol / cum_vol

    std = variance ** 0.5

    upper_inner = vwap_line + multiplier_inner * std
    upper_outer = vwap_line + multiplier_outer * std
    lower_inner = vwap_line - multiplier_inner * std
    lower_outer = vwap_line - multiplier_outer * std
    return vwap_line, upper_inner, upper_outer, lower_inner, lower_outer


def obv(df: pd.DataFrame) -> pd.Series:
    """On Balance Volume.

    Expects columns: close, volume.
    """
    direction = np.sign(df["close"].diff())
    direction.iloc[0] = 0
    return (direction * df["volume"]).cumsum()


def ad_line(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line.

    Expects columns: high, low, close, volume.
    """
    high_low = df["high"] - df["low"]
    # Avoid division by zero when high == low
    mfm = np.where(
        high_low == 0,
        0.0,
        ((df["close"] - df["low"]) - (df["high"] - df["close"])) / high_low,
    )
    mfv = mfm * df["volume"].values
    return pd.Series(mfv, index=df.index).cumsum()
