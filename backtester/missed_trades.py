import pandas as pd

from backtester.indicators import vwap, obv, ad_line


def find_missed_trades(
    df: pd.DataFrame,
    obv_lookback: int = 5,
    ad_lookback: int = 5,
) -> pd.DataFrame:
    """Identify bars where a signal was *nearly* triggered.

    A "near miss" is a bar where exactly 2 of the 3 conditions were met
    (price vs VWAP, OBV trend, A/D trend).  These represent trades the
    strategy would have taken with slightly different parameters or price
    action.

    Returns a DataFrame with columns:
        date, close, vwap, obv, ad_line, met_conditions (list[str]),
        missed_condition (str), direction ('buy' | 'sell')
    """
    data = df.copy()
    data["vwap"] = vwap(df)
    data["obv"] = obv(df)
    data["ad_line"] = ad_line(df)

    obv_shifted = data["obv"].shift(obv_lookback)
    ad_shifted = data["ad_line"].shift(ad_lookback)

    buy_conditions = {
        "price_below_vwap": data["close"] < data["vwap"],
        "obv_rising": data["obv"] > obv_shifted,
        "ad_rising": data["ad_line"] > ad_shifted,
    }

    sell_conditions = {
        "price_above_vwap": data["close"] > data["vwap"],
        "obv_falling": data["obv"] < obv_shifted,
        "ad_falling": data["ad_line"] < ad_shifted,
    }

    rows: list[dict] = []

    for i in range(len(data)):
        date = data.index[i]
        close = data["close"].iloc[i]
        vwap_val = data["vwap"].iloc[i]
        obv_val = data["obv"].iloc[i]
        ad_val = data["ad_line"].iloc[i]

        for direction, conds in [("buy", buy_conditions), ("sell", sell_conditions)]:
            met = [name for name, mask in conds.items() if mask.iloc[i]]
            missed = [name for name, mask in conds.items() if not mask.iloc[i]]

            # Exactly 2 of 3 met → near miss
            if len(met) == 2:
                rows.append(
                    {
                        "date": date,
                        "close": close,
                        "vwap": vwap_val,
                        "obv": obv_val,
                        "ad_line": ad_val,
                        "met_conditions": met,
                        "missed_condition": missed[0],
                        "direction": direction,
                    }
                )

    return pd.DataFrame(rows)
