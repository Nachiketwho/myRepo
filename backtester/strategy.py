import pandas as pd
from backtester.indicators import vwap, obv, ad_line


class VolumeStrategy:
    """Simple strategy combining VWAP, OBV, and A/D Line signals.

    Buy when: price < VWAP (undervalued) AND OBV rising AND A/D rising
    Sell when: price > VWAP (overvalued) AND OBV falling AND A/D falling
    """

    def __init__(self, obv_lookback: int = 5, ad_lookback: int = 5):
        self.obv_lookback = obv_lookback
        self.ad_lookback = ad_lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with a 'signal' column: 1=buy, -1=sell, 0=hold."""
        result = df.copy()
        result["vwap"] = vwap(df)
        result["obv"] = obv(df)
        result["ad_line"] = ad_line(df)

        obv_rising = result["obv"] > result["obv"].shift(self.obv_lookback)
        ad_rising = result["ad_line"] > result["ad_line"].shift(self.ad_lookback)
        price_below_vwap = result["close"] < result["vwap"]

        obv_falling = result["obv"] < result["obv"].shift(self.obv_lookback)
        ad_falling = result["ad_line"] < result["ad_line"].shift(self.ad_lookback)
        price_above_vwap = result["close"] > result["vwap"]

        result["signal"] = 0
        result.loc[price_below_vwap & obv_rising & ad_rising, "signal"] = 1
        result.loc[price_above_vwap & obv_falling & ad_falling, "signal"] = -1

        return result
