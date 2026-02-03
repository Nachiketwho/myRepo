"""Diagnostic script — print indicator values and condition breakdown.

Usage:
    python diagnose.py --timeframe daily
    python diagnose.py --timeframe 5min --period 60d
    python diagnose.py --timeframe daily --band-multiplier 1.5
"""

import argparse
import pandas as pd
from backtester.data import fetch_nifty50, TIMEFRAMES
from backtester.strategy import VolumeStrategy


def diagnose(
    df: pd.DataFrame,
    band_mult: float = 2.0,
    obv_lb: int = 5,
    ad_lb: int = 5,
):
    strategy = VolumeStrategy(
        band_multiplier=band_mult, obv_lookback=obv_lb, ad_lookback=ad_lb
    )
    s = strategy.generate_signals(df)

    # Primary conditions
    buy_bounce = (s["low"] <= s["vwap_lower"]) & (s["close"] > s["vwap_lower"])
    sell_reject = (s["high"] >= s["vwap_upper"]) & (s["close"] < s["vwap_upper"])

    # Confirmation
    obv_rising = s["obv"] > s["obv"].shift(obv_lb)
    ad_rising = s["ad_line"] > s["ad_line"].shift(ad_lb)
    obv_falling = s["obv"] < s["obv"].shift(obv_lb)
    ad_falling = s["ad_line"] < s["ad_line"].shift(ad_lb)
    buy_confirmed = obv_rising | ad_rising
    sell_confirmed = obv_falling | ad_falling

    print("=" * 60)
    print("  INDICATOR VALUES (last 20 bars)")
    print("=" * 60)
    cols = ["close", "vwap", "vwap_upper", "vwap_lower", "obv", "ad_line", "signal"]
    print(s[cols].tail(20).to_string())

    print()
    print("=" * 60)
    print("  CONDITION BREAKDOWN (last 20 bars)")
    print("=" * 60)
    s["buy_bounce"] = buy_bounce
    s["sell_reject"] = sell_reject
    s["buy_confirmed"] = buy_confirmed
    s["sell_confirmed"] = sell_confirmed
    cond_cols = [
        "close", "vwap_lower", "vwap_upper",
        "buy_bounce", "buy_confirmed", "sell_reject", "sell_confirmed", "signal",
    ]
    print(s[cond_cols].tail(20).to_string())

    print()
    print("=" * 60)
    print(f"  SIGNAL SUMMARY (band_mult={band_mult})")
    print("=" * 60)
    print(f"  Total bars:                   {len(s)}")
    print(f"  Buy signals:                  {(s['signal'] == 1).sum()}")
    print(f"  Sell signals:                 {(s['signal'] == -1).sum()}")
    print()
    print("  --- Primary (VWAP band) ---")
    print(f"  Lower band bounces:           {buy_bounce.sum()} bars")
    print(f"  Upper band rejections:        {sell_reject.sum()} bars")
    print()
    print("  --- Confirmation (volume) ---")
    print(f"  OBV rising:                   {obv_rising.sum()} bars")
    print(f"  OBV falling:                  {obv_falling.sum()} bars")
    print(f"  AD rising:                    {ad_rising.sum()} bars")
    print(f"  AD falling:                   {ad_falling.sum()} bars")
    print(f"  Buy confirmed (OBV|AD up):    {buy_confirmed.sum()} bars")
    print(f"  Sell confirmed (OBV|AD dn):   {sell_confirmed.sum()} bars")
    print()
    print("  --- Near misses ---")
    print(f"  Band bounce, no vol confirm:  {(buy_bounce & ~buy_confirmed).sum()} buy, "
          f"{(sell_reject & ~sell_confirmed).sum()} sell")
    print(f"  Vol confirmed, band missed:   check missed_trades output")


def main():
    parser = argparse.ArgumentParser(description="Diagnose indicator values")
    parser.add_argument("--timeframe", default="daily", choices=list(TIMEFRAMES.keys()))
    parser.add_argument("--period", default=None, help="yfinance period e.g. 2y, 60d")
    parser.add_argument("--band-multiplier", type=float, default=2.0)
    parser.add_argument("--obv-lookback", type=int, default=5)
    parser.add_argument("--ad-lookback", type=int, default=5)
    args = parser.parse_args()

    df = fetch_nifty50(timeframe=args.timeframe, period=args.period)
    print(f"\nFetched {len(df)} bars ({args.timeframe}): {df.index[0]} → {df.index[-1]}\n")
    diagnose(df, args.band_multiplier, args.obv_lookback, args.ad_lookback)


if __name__ == "__main__":
    main()
