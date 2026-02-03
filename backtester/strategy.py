from itertools import product
import pandas as pd
from backtester.indicators import vwap_bands, obv, ad_line


# Default parameter grids — wider exits for Nifty50 daily volatility
PARAM_GRID = {
    "band_multiplier": [1.0, 1.5, 2.0],
    "obv_lookback": [3, 5, 10, 14, 20],
    "ad_lookback": [3, 5, 10, 14, 20],
}

ENGINE_PARAM_GRID = {
    "sl_pct": [2.0, 3.0, 5.0],
    "tsl_pct": [1.5, 2.0, 3.0],
    "tp_pct": [3.0, 5.0, 8.0],
    "ttp_pct": [1.0, 1.5, 2.0],
}


class VolumeStrategy:
    """VWAP-band bounce strategy with volume confirmation.

    Factors (priority order):
        1. PRIMARY — VWAP band bounce:
           - Buy zone:  price touches/dips to lower band and bounces
                         (low <= lower_band AND close > lower_band)
           - Sell zone:  price touches/rises to upper band and rejects
                         (high >= upper_band AND close < upper_band)

        2. CONFIRMATION — at least 1 of 2 must confirm:
           - OBV rising over lookback period (volume supports direction)
           - A/D Line rising over lookback period (money flow supports)

    This replaces the old 3-of-3 cross logic which missed trend trades.
    """

    def __init__(
        self,
        band_multiplier: float = 2.0,
        obv_lookback: int = 5,
        ad_lookback: int = 5,
    ):
        self.band_multiplier = band_multiplier
        self.obv_lookback = obv_lookback
        self.ad_lookback = ad_lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with indicator and signal columns.

        Added columns: vwap, vwap_upper, vwap_lower, obv, ad_line, signal.
        signal: 1=buy, -1=sell, 0=hold.
        """
        result = df.copy()

        vwap_line, upper, lower = vwap_bands(df, self.band_multiplier)
        result["vwap"] = vwap_line
        result["vwap_upper"] = upper
        result["vwap_lower"] = lower
        result["obv"] = obv(df)
        result["ad_line"] = ad_line(df)

        # --- Primary: VWAP band bounce ---
        # Buy: price touched lower band and bounced (closed above it)
        buy_bounce = (result["low"] <= result["vwap_lower"]) & (
            result["close"] > result["vwap_lower"]
        )
        # Sell: price touched upper band and rejected (closed below it)
        sell_reject = (result["high"] >= result["vwap_upper"]) & (
            result["close"] < result["vwap_upper"]
        )

        # --- Confirmation: at least 1 of 2 ---
        obv_rising = result["obv"] > result["obv"].shift(self.obv_lookback)
        ad_rising = result["ad_line"] > result["ad_line"].shift(self.ad_lookback)
        obv_falling = result["obv"] < result["obv"].shift(self.obv_lookback)
        ad_falling = result["ad_line"] < result["ad_line"].shift(self.ad_lookback)

        buy_confirmed = obv_rising | ad_rising
        sell_confirmed = obv_falling | ad_falling

        result["signal"] = 0
        result.loc[buy_bounce & buy_confirmed, "signal"] = 1
        result.loc[sell_reject & sell_confirmed, "signal"] = -1

        return result

    @staticmethod
    def iter_param_sets(grid: dict | None = None):
        """Yield dicts of strategy parameter combinations."""
        if grid is None:
            grid = PARAM_GRID
        keys = list(grid.keys())
        for vals in product(*grid.values()):
            yield dict(zip(keys, vals))
