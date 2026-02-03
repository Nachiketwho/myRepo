"""Diagnostic script — print indicator values and condition breakdown.

Usage:
    python diagnose.py --timeframe daily
    python diagnose.py --timeframe 5min --period 60d
"""

import argparse
import pandas as pd
from backtester.data import fetch_nifty50, TIMEFRAMES
from backtester.strategy import VolumeStrategy


def diagnose(df: pd.DataFrame, obv_lb: int = 5, ad_lb: int = 5):
    strategy = VolumeStrategy(obv_lookback=obv_lb, ad_lookback=ad_lb)
    s = strategy.generate_signals(df)

    s["price_below_vwap"] = s["close"] < s["vwap"]
    s["price_above_vwap"] = s["close"] > s["vwap"]
    s["obv_rising"] = s["obv"] > s["obv"].shift(obv_lb)
    s["obv_falling"] = s["obv"] < s["obv"].shift(obv_lb)
    s["ad_rising"] = s["ad_line"] > s["ad_line"].shift(ad_lb)
    s["ad_falling"] = s["ad_line"] < s["ad_line"].shift(ad_lb)

    buy = s["price_below_vwap"] & s["obv_rising"] & s["ad_rising"]
    sell = s["price_above_vwap"] & s["obv_falling"] & s["ad_falling"]

    print("=" * 60)
    print("  INDICATOR VALUES (last 20 bars)")
    print("=" * 60)
    print(s[["close", "vwap", "obv", "ad_line", "signal"]].tail(20).to_string())

    print()
    print("=" * 60)
    print("  CONDITION BREAKDOWN (last 20 bars)")
    print("=" * 60)
    cond_cols = [
        "close", "price_below_vwap", "obv_rising", "ad_rising",
        "price_above_vwap", "obv_falling", "ad_falling", "signal",
    ]
    print(s[cond_cols].tail(20).to_string())

    print()
    print("=" * 60)
    print("  SIGNAL SUMMARY")
    print("=" * 60)
    print(f"  Total bars:           {len(s)}")
    print(f"  Buy signals (3/3):    {buy.sum()}")
    print(f"  Sell signals (3/3):   {sell.sum()}")
    print()
    print(f"  price < VWAP:         {s['price_below_vwap'].sum()} bars")
    print(f"  price > VWAP:         {s['price_above_vwap'].sum()} bars")
    print(f"  OBV rising:           {s['obv_rising'].sum()} bars")
    print(f"  OBV falling:          {s['obv_falling'].sum()} bars")
    print(f"  AD rising:            {s['ad_rising'].sum()} bars")
    print(f"  AD falling:           {s['ad_falling'].sum()} bars")
    print()
    print("  2-of-3 near misses (buy side):")
    print(f"    price<VWAP + OBV rising (missing AD):   {(s['price_below_vwap'] & s['obv_rising'] & ~s['ad_rising']).sum()}")
    print(f"    price<VWAP + AD rising  (missing OBV):  {(s['price_below_vwap'] & ~s['obv_rising'] & s['ad_rising']).sum()}")
    print(f"    OBV rising + AD rising  (missing price): {(~s['price_below_vwap'] & s['obv_rising'] & s['ad_rising']).sum()}")
    print()
    print("  2-of-3 near misses (sell side):")
    print(f"    price>VWAP + OBV falling (missing AD):  {(s['price_above_vwap'] & s['obv_falling'] & ~s['ad_falling']).sum()}")
    print(f"    price>VWAP + AD falling  (missing OBV): {(s['price_above_vwap'] & ~s['obv_falling'] & s['ad_falling']).sum()}")
    print(f"    OBV falling + AD falling (missing price): {(~s['price_above_vwap'] & s['obv_falling'] & s['ad_falling']).sum()}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose indicator values")
    parser.add_argument("--timeframe", default="daily", choices=list(TIMEFRAMES.keys()))
    parser.add_argument("--period", default=None, help="yfinance period e.g. 2y, 60d")
    parser.add_argument("--obv-lookback", type=int, default=5)
    parser.add_argument("--ad-lookback", type=int, default=5)
    args = parser.parse_args()

    df = fetch_nifty50(timeframe=args.timeframe, period=args.period)
    print(f"\nFetched {len(df)} bars ({args.timeframe}): {df.index[0]} → {df.index[-1]}\n")
    diagnose(df, args.obv_lookback, args.ad_lookback)


if __name__ == "__main__":
    main()
