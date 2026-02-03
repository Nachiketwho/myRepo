import pandas as pd

from backtester.indicators import vwap_bands, obv, ad_line


def find_missed_trades(
    df: pd.DataFrame,
    band_multiplier: float = 2.0,
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
        date, close, vwap, vwap_upper, vwap_lower, obv, ad_line,
        met_conditions (list[str]), missed_condition (str), direction
    """
    data = df.copy()
    vwap_line, upper, lower = vwap_bands(df, band_multiplier)
    data["vwap"] = vwap_line
    data["vwap_upper"] = upper
    data["vwap_lower"] = lower
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

    # Primary conditions
    buy_bounce = (data["low"] <= data["vwap_lower"]) & (
        data["close"] > data["vwap_lower"]
    )
    sell_reject = (data["high"] >= data["vwap_upper"]) & (
        data["close"] < data["vwap_upper"]
    )

    # Near-band: price within 0.5% of band but didn't touch
    band_tolerance = 0.005
    near_lower = (
        (data["low"] > data["vwap_lower"])
        & (data["low"] <= data["vwap_lower"] * (1 + band_tolerance))
    )
    near_upper = (
        (data["high"] < data["vwap_upper"])
        & (data["high"] >= data["vwap_upper"] * (1 - band_tolerance))
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
            "vwap_upper": data["vwap_upper"].iloc[i],
            "vwap_lower": data["vwap_lower"].iloc[i],
            "obv": data["obv"].iloc[i],
            "ad_line": data["ad_line"].iloc[i],
        }

        # Case 1: Band bounce happened but no volume confirmation
        if buy_bounce.iloc[i] and not buy_confirmed.iloc[i]:
            rows.append(
                {
                    **base,
                    "met_conditions": ["lower_band_bounce"],
                    "missed_condition": "volume_confirmation",
                    "direction": "buy",
                }
            )

        if sell_reject.iloc[i] and not sell_confirmed.iloc[i]:
            rows.append(
                {
                    **base,
                    "met_conditions": ["upper_band_reject"],
                    "missed_condition": "volume_confirmation",
                    "direction": "sell",
                }
            )

        # Case 2: Volume confirmed but price didn't quite reach the band
        if near_lower.iloc[i] and buy_confirmed.iloc[i]:
            rows.append(
                {
                    **base,
                    "met_conditions": ["volume_confirmation", "near_lower_band"],
                    "missed_condition": "band_touch",
                    "direction": "buy",
                }
            )

        if near_upper.iloc[i] and sell_confirmed.iloc[i]:
            rows.append(
                {
                    **base,
                    "met_conditions": ["volume_confirmation", "near_upper_band"],
                    "missed_condition": "band_touch",
                    "direction": "sell",
                }
            )

    return pd.DataFrame(rows)
