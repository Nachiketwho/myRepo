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
    df: pd.DataFrame, multiplier: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """VWAP with upper and lower standard-deviation bands.

    Lower band = dynamic support, upper band = dynamic resistance.
    Price bouncing off lower band with volume confirmation = buy.
    Price rejecting at upper band with volume confirmation = sell.

    Returns:
        (vwap_line, upper_band, lower_band)
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_volume = typical_price * df["volume"]

    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
        cum_tp_vol = tp_volume.groupby(dates).cumsum()
        cum_vol = df["volume"].groupby(dates).cumsum()
        vwap_line = cum_tp_vol / cum_vol
        sq_diff_vol = ((typical_price - vwap_line) ** 2) * df["volume"]
        cum_sq_diff_vol = sq_diff_vol.groupby(dates).cumsum()
    else:
        cum_tp_vol = tp_volume.cumsum()
        cum_vol = df["volume"].cumsum()
        vwap_line = cum_tp_vol / cum_vol
        sq_diff_vol = ((typical_price - vwap_line) ** 2) * df["volume"]
        cum_sq_diff_vol = sq_diff_vol.cumsum()

    variance = cum_sq_diff_vol / cum_vol
    std = variance ** 0.5

    upper = vwap_line + multiplier * std
    lower = vwap_line - multiplier * std
    return vwap_line, upper, lower


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
