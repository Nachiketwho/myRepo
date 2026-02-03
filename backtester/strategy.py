from itertools import product
import pandas as pd
from backtester.indicators import vwap_bands, obv, ad_line


# ---------------------------------------------------------------------------
# Timeframe-aware defaults
# ---------------------------------------------------------------------------
# Daily/weekly: daily-reset VWAP (no rolling window), wider exits
# Intraday: rolling VWAP window so bands have consistent width,
#           tighter exits matching smaller bar-to-bar moves
# ---------------------------------------------------------------------------

TIMEFRAME_DEFAULTS = {
    "daily": {
        "band_multiplier_inner": 1.0,
        "band_multiplier_outer": 2.0,
        "vwap_window": None,          # daily-reset cumulative
        "obv_lookback": 5,
        "ad_lookback": 5,
        "sl_pct": 3.0,
        "tsl_pct": 2.0,
        "tp_pct": 5.0,
        "ttp_pct": 1.5,
    },
    "weekly": {
        "band_multiplier_inner": 1.5,
        "band_multiplier_outer": 2.5,
        "vwap_window": None,
        "obv_lookback": 5,
        "ad_lookback": 5,
        "sl_pct": 5.0,
        "tsl_pct": 3.0,
        "tp_pct": 8.0,
        "ttp_pct": 2.0,
    },
    "hourly": {
        "band_multiplier_inner": 0.8,
        "band_multiplier_outer": 1.5,
        "vwap_window": 20,            # ~20h rolling
        "obv_lookback": 4,
        "ad_lookback": 4,
        "sl_pct": 1.0,
        "tsl_pct": 0.6,
        "tp_pct": 2.0,
        "ttp_pct": 0.5,
    },
    "15min": {
        "band_multiplier_inner": 0.5,
        "band_multiplier_outer": 1.5,
        "vwap_window": 20,            # 20 bars × 15min = 5h rolling
        "obv_lookback": 3,
        "ad_lookback": 3,
        "sl_pct": 0.5,
        "tsl_pct": 0.3,
        "tp_pct": 1.0,
        "ttp_pct": 0.3,
    },
    "5min": {
        "band_multiplier_inner": 0.5,
        "band_multiplier_outer": 1.2,
        "vwap_window": 30,            # 30 bars × 5min = 2.5h rolling
        "obv_lookback": 3,
        "ad_lookback": 3,
        "sl_pct": 0.3,
        "tsl_pct": 0.2,
        "tp_pct": 0.8,
        "ttp_pct": 0.2,
    },
}


def get_defaults(timeframe: str = "daily") -> dict:
    """Return default parameters for a given timeframe."""
    return TIMEFRAME_DEFAULTS.get(timeframe, TIMEFRAME_DEFAULTS["daily"]).copy()


# ---------------------------------------------------------------------------
# Parameter grids (default = daily-oriented)
# ---------------------------------------------------------------------------

PARAM_GRID = {
    "band_multiplier_inner": [0.5, 1.0, 1.5],
    "band_multiplier_outer": [1.5, 2.0, 2.5],
    "obv_lookback": [3, 5, 10, 14, 20],
    "ad_lookback": [3, 5, 10, 14, 20],
}

ENGINE_PARAM_GRID = {
    "sl_pct": [2.0, 3.0, 5.0],
    "tsl_pct": [1.5, 2.0, 3.0],
    "tp_pct": [3.0, 5.0, 8.0],
    "ttp_pct": [1.0, 1.5, 2.0],
}

# 15min-specific grids — tighter ranges for intraday
PARAM_GRID_15MIN = {
    "band_multiplier_inner": [0.3, 0.5, 0.8],
    "band_multiplier_outer": [1.0, 1.5, 2.0],
    "obv_lookback": [2, 3, 5],
    "ad_lookback": [2, 3, 5],
}

ENGINE_PARAM_GRID_15MIN = {
    "sl_pct": [0.3, 0.5, 0.8],
    "tsl_pct": [0.2, 0.3, 0.5],
    "tp_pct": [0.5, 1.0, 1.5],
    "ttp_pct": [0.2, 0.3, 0.5],
}


class VolumeStrategy:
    """VWAP dual-band bounce strategy with volume confirmation.

    Uses two sets of bands:
        - Inner bands (1st std dev): frequent touches, initial S/R level
        - Outer bands (2nd std dev): strong S/R, higher conviction entries

    Factors (priority order):
        1. PRIMARY — VWAP band bounce:
           - Buy zone:  price touches/dips to inner OR outer lower band
                         and bounces (low <= band AND close > band)
           - Sell zone:  price touches/rises to inner OR outer upper band
                         and rejects (high >= band AND close < band)

        2. CONFIRMATION — at least 1 of 2 must confirm:
           - OBV rising over lookback period (volume supports direction)
           - A/D Line rising over lookback period (money flow supports)
    """

    def __init__(
        self,
        band_multiplier_inner: float = 1.0,
        band_multiplier_outer: float = 2.0,
        vwap_window: int | None = None,
        obv_lookback: int = 5,
        ad_lookback: int = 5,
    ):
        self.band_multiplier_inner = band_multiplier_inner
        self.band_multiplier_outer = band_multiplier_outer
        self.vwap_window = vwap_window
        self.obv_lookback = obv_lookback
        self.ad_lookback = ad_lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with indicator and signal columns.

        Added columns: vwap, vwap_upper_inner, vwap_upper_outer,
                        vwap_lower_inner, vwap_lower_outer,
                        obv, ad_line, signal.
        signal: 1=buy, -1=sell, 0=hold.
        """
        result = df.copy()

        vwap_line, upper_inner, upper_outer, lower_inner, lower_outer = vwap_bands(
            df, self.band_multiplier_inner, self.band_multiplier_outer,
            rolling_window=self.vwap_window,
        )
        result["vwap"] = vwap_line
        result["vwap_upper_inner"] = upper_inner
        result["vwap_upper_outer"] = upper_outer
        result["vwap_lower_inner"] = lower_inner
        result["vwap_lower_outer"] = lower_outer
        result["obv"] = obv(df)
        result["ad_line"] = ad_line(df)

        # --- Primary: VWAP band bounce (inner OR outer) ---
        # Buy: price touched lower band (inner or outer) and bounced
        buy_inner = (result["low"] <= result["vwap_lower_inner"]) & (
            result["close"] > result["vwap_lower_inner"]
        )
        buy_outer = (result["low"] <= result["vwap_lower_outer"]) & (
            result["close"] > result["vwap_lower_outer"]
        )
        buy_bounce = buy_inner | buy_outer

        # Sell: price touched upper band (inner or outer) and rejected
        sell_inner = (result["high"] >= result["vwap_upper_inner"]) & (
            result["close"] < result["vwap_upper_inner"]
        )
        sell_outer = (result["high"] >= result["vwap_upper_outer"]) & (
            result["close"] < result["vwap_upper_outer"]
        )
        sell_reject = sell_inner | sell_outer

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
