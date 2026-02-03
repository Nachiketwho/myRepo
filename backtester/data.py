import pandas as pd


NIFTY50_SYMBOL = "^NSEI"

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
