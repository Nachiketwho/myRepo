import pandas as pd
import numpy as np


NIFTY50_SYMBOL = "^NSEI"
INDIA_VIX_SYMBOL = "^INDIAVIX"

TIMEFRAMES = {
    "daily": "1d",
    "weekly": "1wk",
    "hourly": "1h",
    "15min": "15m",
    "5min": "5m",
}

# Yahoo Finance limits history for intraday intervals
PERIOD_DEFAULTS = {
    "daily": "5y",
    "weekly": "10y",
    "hourly": "2y",
    "15min": "60d",
    "5min": "60d",
}


def fetch_nifty50(
    timeframe: str = "daily",
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch Nifty50 OHLCV data from Yahoo Finance.

    Args:
        timeframe: One of 'daily', 'weekly', 'hourly', '15min' or a raw
                   yfinance interval string.
        period: yfinance period like '1y', '6mo'. Ignored if start/end given.
        start: Start date string 'YYYY-MM-DD'.
        end: End date string 'YYYY-MM-DD'.

    Returns:
        DataFrame with columns: open, high, low, close, volume
        and a DatetimeIndex.
    """
    import yfinance as yf

    interval = TIMEFRAMES.get(timeframe, timeframe)

    if period is None and start is None:
        period = PERIOD_DEFAULTS.get(timeframe, "2y")

    ticker = yf.Ticker(NIFTY50_SYMBOL)

    kwargs = {"interval": interval}
    if start and end:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period

    df = ticker.history(**kwargs)

    if df.empty:
        raise ValueError(
            f"No data returned for {NIFTY50_SYMBOL} "
            f"(timeframe={timeframe}, period={period})"
        )

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"

    # Yahoo Finance returns 0 volume for some indices on intraday
    # timeframes.  Replace with 1 so VWAP/OBV/AD don't produce NaN / 0.
    if (df["volume"] == 0).all():
        df["volume"] = 1

    return df


def fetch_india_vix(
    period: str = "2y",
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Fetch India VIX daily close from Yahoo Finance.

    Returns a Series of VIX values (annualized vol in %) indexed by date.
    For Black-Scholes, divide by 100 to get decimal form.
    """
    import yfinance as yf

    ticker = yf.Ticker(INDIA_VIX_SYMBOL)

    kwargs = {"interval": "1d"}
    if start and end:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period

    df = ticker.history(**kwargs)

    if df.empty:
        raise ValueError(f"No VIX data returned for {INDIA_VIX_SYMBOL}")

    df.columns = [c.lower() for c in df.columns]
    vix = df["close"]
    vix.index = vix.index.date if hasattr(vix.index, 'date') else vix.index
    vix.index.name = "date"
    return vix


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

class DataValidationError(Exception):
    """Raised when fetched data fails integrity checks."""


def validate_ohlcv(df: pd.DataFrame, label: str = "data") -> list[str]:
    """Run integrity checks on OHLCV data.  Returns list of warnings.

    Raises DataValidationError for critical failures.
    """
    warnings: list[str] = []

    if df.empty:
        raise DataValidationError(f"{label}: DataFrame is empty — no data to validate")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"{label}: missing columns: {missing}")

    n = len(df)

    # --- NaN checks ---
    nan_counts = df[["open", "high", "low", "close"]].isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        pct = total_nans / (n * 4) * 100
        warnings.append(f"{label}: {total_nans} NaN values in OHLC ({pct:.1f}%)")
        if pct > 10:
            raise DataValidationError(
                f"{label}: too many NaN values ({pct:.1f}%) — data unreliable"
            )

    # --- OHLC relationship: high >= low, high >= open/close, low <= open/close ---
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl > 0:
        warnings.append(f"{label}: {bad_hl} bars where high < low")

    bad_ho = (df["high"] < df["open"]).sum()
    bad_hc = (df["high"] < df["close"]).sum()
    if bad_ho + bad_hc > 0:
        warnings.append(f"{label}: {bad_ho + bad_hc} bars where high < open or close")

    bad_lo = (df["low"] > df["open"]).sum()
    bad_lc = (df["low"] > df["close"]).sum()
    if bad_lo + bad_lc > 0:
        warnings.append(f"{label}: {bad_lo + bad_lc} bars where low > open or close")

    # --- Zero/negative prices ---
    zero_close = (df["close"] <= 0).sum()
    if zero_close > 0:
        raise DataValidationError(
            f"{label}: {zero_close} bars with close <= 0 — data corrupt"
        )

    # --- Duplicate index ---
    dupes = df.index.duplicated().sum()
    if dupes > 0:
        warnings.append(f"{label}: {dupes} duplicate timestamps in index")

    # --- Monotonic index ---
    if not df.index.is_monotonic_increasing:
        warnings.append(f"{label}: index is not monotonically increasing")

    # --- Price range sanity (Nifty50 specific) ---
    min_close = df["close"].min()
    max_close = df["close"].max()
    if min_close < 5000 or max_close > 50000:
        warnings.append(
            f"{label}: close range [{min_close:.0f}, {max_close:.0f}] "
            f"looks unusual for Nifty50"
        )

    # --- Volume sanity ---
    zero_vol_pct = (df["volume"] == 0).sum() / n * 100
    if zero_vol_pct > 50:
        warnings.append(
            f"{label}: {zero_vol_pct:.0f}% bars have zero volume "
            f"(Yahoo Finance known issue for index intraday)"
        )

    # --- Data freshness ---
    last_date = df.index[-1]
    if hasattr(last_date, 'date'):
        last_date = last_date.date()
    import datetime as _dt
    today = _dt.date.today()
    gap_days = (today - last_date).days if isinstance(last_date, _dt.date) else 0
    if gap_days > 5:
        warnings.append(
            f"{label}: last data point is {gap_days} days old "
            f"({last_date}) — data may be stale"
        )

    return warnings
