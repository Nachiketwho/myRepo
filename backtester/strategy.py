from itertools import product
import pandas as pd
from backtester.indicators import vwap, obv, ad_line


# Default parameter grids — intentionally wide for more trades
PARAM_GRID = {
    "obv_lookback": [3, 5, 10, 14, 20],
    "ad_lookback": [3, 5, 10, 14, 20],
}

ENGINE_PARAM_GRID = {
    "sl_pct": [1.5, 2.0, 3.0, 5.0],
    "tsl_pct": [1.0, 1.5, 2.0, 3.0],
    "tp_pct": [2.0, 3.0, 5.0, 8.0],
    "ttp_pct": [0.5, 1.0, 1.5, 2.0],
}


class VolumeStrategy:
    """Strategy combining VWAP, OBV, and A/D Line signals.

    Buy when: close < VWAP AND OBV rising AND A/D rising
    Sell when: close > VWAP AND OBV falling AND A/D falling

    Args:
        obv_lookback: Bars to look back for OBV trend direction.
        ad_lookback: Bars to look back for A/D Line trend direction.
    """

    def __init__(self, obv_lookback: int = 5, ad_lookback: int = 5):
        self.obv_lookback = obv_lookback
        self.ad_lookback = ad_lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with added indicator and signal columns.

        Added columns: vwap, obv, ad_line, signal (1=buy, -1=sell, 0=hold).
        """
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

    @staticmethod
    def iter_param_sets(grid: dict | None = None):
        """Yield dicts of strategy parameter combinations.

        >>> list(VolumeStrategy.iter_param_sets({"obv_lookback": [3, 5]}))
        [{'obv_lookback': 3}, {'obv_lookback': 5}]
        """
        if grid is None:
            grid = PARAM_GRID
        keys = list(grid.keys())
        for vals in product(*grid.values()):
            yield dict(zip(keys, vals))
