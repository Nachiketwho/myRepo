"""Phantom trade detection for F&O — identify missed opportunities.

Scans bars where signal == 0 but a significant spot move followed.
For each phantom trade, logs which conditions were met vs missing
and the theoretical premium PnL if a trade had been taken.

This is analysis-only (uses look-ahead) and must never feed back
into entry/exit decisions.
"""

import pandas as pd
import numpy as np

from backtester.greeks import bs_price
from backtester.strikes import (
    get_atm_strike, next_expiry_dates, time_to_expiry_years,
    STRIKE_INTERVAL,
)


def find_phantom_trades(
    signals_df: pd.DataFrame,
    vix_series: pd.Series,
    spot_move_threshold: float = 50.0,
    look_ahead: int = 10,
    risk_free_rate: float = 0.07,
) -> pd.DataFrame:
    """Identify bars where no signal fired but a significant move followed.

    Args:
        signals_df: DataFrame with signal, OHLCV, and indicator columns.
        vix_series: India VIX daily close (in %).
        spot_move_threshold: Minimum spot move (points) to flag as missed.
        look_ahead: Number of bars to look ahead for move detection.
        risk_free_rate: Risk-free rate for BS pricing.

    Returns:
        DataFrame of phantom trades with conditions met/missing and
        theoretical premium PnL.
    """
    from backtester.fno_engine import FnOEngine  # lazy import to avoid circular

    rows: list[dict] = []
    n = len(signals_df)

    for i in range(n - look_ahead):
        row = signals_df.iloc[i]

        # Only look at bars with no signal
        if row["signal"] != 0:
            continue

        current_close = row["close"]
        date = signals_df.index[i]
        current_date = date.date() if hasattr(date, "date") else date

        # Check future spot movement
        future_closes = [
            signals_df.iloc[i + j]["close"] for j in range(1, look_ahead + 1)
        ]
        max_up = max(future_closes) - current_close
        max_down = current_close - min(future_closes)

        if max_up < spot_move_threshold and max_down < spot_move_threshold:
            continue  # no significant move — skip

        # Determine direction of the missed move
        if max_up >= max_down:
            direction = "bullish"
            max_move = round(max_up, 2)
            peak_spot = max(future_closes)
            peak_bar = future_closes.index(peak_spot) + 1
            option_type = "CE"
        else:
            direction = "bearish"
            max_move = round(max_down, 2)
            peak_spot = min(future_closes)
            peak_bar = future_closes.index(peak_spot) + 1
            option_type = "PE"

        # Check which conditions were / weren't met
        conditions_met = []
        conditions_missing = []

        # Inner band touch
        if direction == "bullish":
            if _col_check(row, "vwap_lower_inner") and row["low"] <= row["vwap_lower_inner"]:
                conditions_met.append("inner_band")
            else:
                conditions_missing.append("inner_band")
            if _col_check(row, "vwap_lower_outer") and row["low"] <= row["vwap_lower_outer"]:
                conditions_met.append("outer_band")
            else:
                conditions_missing.append("outer_band")
        else:
            if _col_check(row, "vwap_upper_inner") and row["high"] >= row["vwap_upper_inner"]:
                conditions_met.append("inner_band")
            else:
                conditions_missing.append("inner_band")
            if _col_check(row, "vwap_upper_outer") and row["high"] >= row["vwap_upper_outer"]:
                conditions_met.append("outer_band")
            else:
                conditions_missing.append("outer_band")

        # OBV confirmation
        if "obv" in row.index and i >= 3:
            obv_now = row["obv"]
            obv_prev = signals_df.iloc[i - 3]["obv"]
            obv_ok = (direction == "bullish" and obv_now > obv_prev) or \
                     (direction == "bearish" and obv_now < obv_prev)
            if obv_ok:
                conditions_met.append("obv")
            else:
                conditions_missing.append("obv")
        else:
            conditions_missing.append("obv")

        # AD Line confirmation
        if "ad_line" in row.index and i >= 3:
            ad_now = row["ad_line"]
            ad_prev = signals_df.iloc[i - 3]["ad_line"]
            ad_ok = (direction == "bullish" and ad_now > ad_prev) or \
                    (direction == "bearish" and ad_now < ad_prev)
            if ad_ok:
                conditions_met.append("ad_line")
            else:
                conditions_missing.append("ad_line")
        else:
            conditions_missing.append("ad_line")

        # Theoretical premium PnL (ATM option, next week expiry)
        iv = FnOEngine._get_iv(vix_series, current_date)
        atm = get_atm_strike(current_close)
        expiries = next_expiry_dates(current_date, count=1)
        expiry = expiries[0]

        T_entry = time_to_expiry_years(current_date, expiry)
        if T_entry <= 0:
            continue

        premium_entry = bs_price(current_close, atm, T_entry, risk_free_rate,
                                 iv, option_type)
        if premium_entry <= 0:
            continue

        # Premium at peak spot (approximate T by subtracting bar count)
        peak_date = signals_df.index[i + peak_bar]
        peak_date_d = peak_date.date() if hasattr(peak_date, "date") else peak_date
        T_peak = time_to_expiry_years(peak_date_d, expiry)
        T_peak = max(T_peak, 1e-6)

        iv_peak = FnOEngine._get_iv(vix_series, peak_date_d)
        premium_peak = bs_price(peak_spot, atm, T_peak, risk_free_rate,
                                iv_peak, option_type)
        premium_pnl = round(premium_peak - premium_entry, 2)

        rows.append({
            "date": date,
            "spot": round(current_close, 2),
            "direction": direction,
            "max_spot_move": max_move,
            "peak_bar": peak_bar,
            "conditions_met": ", ".join(conditions_met),
            "conditions_missing": ", ".join(conditions_missing),
            "conditions_met_count": len(conditions_met),
            "conditions_missing_count": len(conditions_missing),
            "option_type": option_type,
            "strike": atm,
            "expiry": expiry,
            "premium_entry": round(premium_entry, 2),
            "premium_at_peak": round(premium_peak, 2),
            "premium_pnl": premium_pnl,
        })

    return pd.DataFrame(rows)


def _col_check(row: pd.Series, col: str) -> bool:
    """Check if column exists in row and has a non-NaN value."""
    return col in row.index and pd.notna(row[col])
