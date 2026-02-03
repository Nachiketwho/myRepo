import pandas as pd

from backtester.indicators import vwap_bands, obv, ad_line


def find_missed_trades(
    df: pd.DataFrame,
    band_multiplier_inner: float = 1.0,
    band_multiplier_outer: float = 2.0,
    rolling_window: int | None = None,
    obv_lookback: int = 5,
    ad_lookback: int = 5,
) -> pd.DataFrame:
    """Identify bars where a signal was *nearly* triggered.

    A "near miss" is a bar where the primary VWAP band condition was met
    but volume confirmation was missing, OR where confirmation was present
    but price didn't quite reach the band.

    Near-miss categories:
        - Band bounce without confirmation (primary met, no OBV/AD support)
        - Close to band with confirmation (price within 0.5% of band + confirmed)

    Returns a DataFrame with columns:
        date, close, vwap, vwap_upper_inner, vwap_upper_outer,
        vwap_lower_inner, vwap_lower_outer, obv, ad_line,
        met_conditions (list[str]), missed_condition (str), direction
    """
    data = df.copy()
    vwap_line, upper_inner, upper_outer, lower_inner, lower_outer = vwap_bands(
        df, band_multiplier_inner, band_multiplier_outer,
        rolling_window=rolling_window,
    )
    data["vwap"] = vwap_line
    data["vwap_upper_inner"] = upper_inner
    data["vwap_upper_outer"] = upper_outer
    data["vwap_lower_inner"] = lower_inner
    data["vwap_lower_outer"] = lower_outer
    data["obv"] = obv(df)
    data["ad_line"] = ad_line(df)

    obv_vals = data["obv"]
    ad_vals = data["ad_line"]
    obv_shifted = obv_vals.shift(obv_lookback)
    ad_shifted = ad_vals.shift(ad_lookback)

    obv_rising = obv_vals > obv_shifted
    ad_rising = ad_vals > ad_shifted
    obv_falling = obv_vals < obv_shifted
    ad_falling = ad_vals < ad_shifted

    # Primary conditions — inner or outer band bounce
    buy_inner = (data["low"] <= data["vwap_lower_inner"]) & (
        data["close"] > data["vwap_lower_inner"]
    )
    buy_outer = (data["low"] <= data["vwap_lower_outer"]) & (
        data["close"] > data["vwap_lower_outer"]
    )
    buy_bounce = buy_inner | buy_outer

    sell_inner = (data["high"] >= data["vwap_upper_inner"]) & (
        data["close"] < data["vwap_upper_inner"]
    )
    sell_outer = (data["high"] >= data["vwap_upper_outer"]) & (
        data["close"] < data["vwap_upper_outer"]
    )
    sell_reject = sell_inner | sell_outer

    # Near-band: price within 0.5% of inner band but didn't touch
    band_tolerance = 0.005
    near_lower = (
        (data["low"] > data["vwap_lower_inner"])
        & (data["low"] <= data["vwap_lower_inner"] * (1 + band_tolerance))
    )
    near_upper = (
        (data["high"] < data["vwap_upper_inner"])
        & (data["high"] >= data["vwap_upper_inner"] * (1 - band_tolerance))
    )

    buy_confirmed = obv_rising | ad_rising
    sell_confirmed = obv_falling | ad_falling

    rows: list[dict] = []

    for i in range(max(obv_lookback, ad_lookback), len(data)):
        date = data.index[i]
        base = {
            "date": date,
            "close": data["close"].iloc[i],
            "vwap": data["vwap"].iloc[i],
            "vwap_upper_inner": data["vwap_upper_inner"].iloc[i],
            "vwap_upper_outer": data["vwap_upper_outer"].iloc[i],
            "vwap_lower_inner": data["vwap_lower_inner"].iloc[i],
            "vwap_lower_outer": data["vwap_lower_outer"].iloc[i],
            "obv": data["obv"].iloc[i],
            "ad_line": data["ad_line"].iloc[i],
        }

        # Case 1: Band bounce happened but no volume confirmation
        if buy_bounce.iloc[i] and not buy_confirmed.iloc[i]:
            met = []
            if buy_inner.iloc[i]:
                met.append("inner_lower_bounce")
            if buy_outer.iloc[i]:
                met.append("outer_lower_bounce")
            rows.append(
                {
                    **base,
                    "met_conditions": met,
                    "missed_condition": "volume_confirmation",
                    "direction": "buy",
                }
            )

        if sell_reject.iloc[i] and not sell_confirmed.iloc[i]:
            met = []
            if sell_inner.iloc[i]:
                met.append("inner_upper_reject")
            if sell_outer.iloc[i]:
                met.append("outer_upper_reject")
            rows.append(
                {
                    **base,
                    "met_conditions": met,
                    "missed_condition": "volume_confirmation",
                    "direction": "sell",
                }
            )

        # Case 2: Volume confirmed but price didn't quite reach inner band
        if near_lower.iloc[i] and buy_confirmed.iloc[i]:
            rows.append(
                {
                    **base,
                    "met_conditions": ["volume_confirmation", "near_inner_lower"],
                    "missed_condition": "band_touch",
                    "direction": "buy",
                }
            )

        if near_upper.iloc[i] and sell_confirmed.iloc[i]:
            rows.append(
                {
                    **base,
                    "met_conditions": ["volume_confirmation", "near_inner_upper"],
                    "missed_condition": "band_touch",
                    "direction": "sell",
                }
            )

    return pd.DataFrame(rows)
