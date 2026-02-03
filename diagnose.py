"""Diagnostic script — print indicator values and condition breakdown.

Usage:
    python diagnose.py --timeframe daily
    python diagnose.py --timeframe 15min
    python diagnose.py --timeframe 15min --period 60d
"""

import argparse
import pandas as pd
from backtester.data import fetch_nifty50, TIMEFRAMES
from backtester.strategy import VolumeStrategy, get_defaults


def diagnose(
    df: pd.DataFrame,
    band_inner: float = 1.0,
    band_outer: float = 2.0,
    obv_lb: int = 5,
    ad_lb: int = 5,
):
    strategy = VolumeStrategy(
        band_multiplier_inner=band_inner,
        band_multiplier_outer=band_outer,
        obv_lookback=obv_lb,
        ad_lookback=ad_lb,
    )
    s = strategy.generate_signals(df)

    # Primary conditions — inner bands
    buy_inner = (s["low"] <= s["vwap_lower_inner"]) & (s["close"] > s["vwap_lower_inner"])
    sell_inner = (s["high"] >= s["vwap_upper_inner"]) & (s["close"] < s["vwap_upper_inner"])

    # Primary conditions — outer bands
    buy_outer = (s["low"] <= s["vwap_lower_outer"]) & (s["close"] > s["vwap_lower_outer"])
    sell_outer = (s["high"] >= s["vwap_upper_outer"]) & (s["close"] < s["vwap_upper_outer"])

    buy_bounce = buy_inner | buy_outer
    sell_reject = sell_inner | sell_outer

    # Confirmation
    obv_rising = s["obv"] > s["obv"].shift(obv_lb)
    ad_rising = s["ad_line"] > s["ad_line"].shift(ad_lb)
    obv_falling = s["obv"] < s["obv"].shift(obv_lb)
    ad_falling = s["ad_line"] < s["ad_line"].shift(ad_lb)
    buy_confirmed = obv_rising | ad_rising
    sell_confirmed = obv_falling | ad_falling

    print("=" * 70)
    print("  INDICATOR VALUES (last 20 bars)")
    print("=" * 70)
    cols = ["close", "vwap", "vwap_upper_inner", "vwap_upper_outer",
            "vwap_lower_inner", "vwap_lower_outer", "obv", "ad_line", "signal"]
    print(s[cols].tail(20).to_string())

    print()
    print("=" * 70)
    print("  CONDITION BREAKDOWN (last 20 bars)")
    print("=" * 70)
    s["buy_inner"] = buy_inner
    s["buy_outer"] = buy_outer
    s["sell_inner"] = sell_inner
    s["sell_outer"] = sell_outer
    s["buy_confirmed"] = buy_confirmed
    s["sell_confirmed"] = sell_confirmed
    cond_cols = [
        "close", "vwap_lower_inner", "vwap_lower_outer",
        "vwap_upper_inner", "vwap_upper_outer",
        "buy_inner", "buy_outer", "buy_confirmed",
        "sell_inner", "sell_outer", "sell_confirmed", "signal",
    ]
    print(s[cond_cols].tail(20).to_string())

    print()
    print("=" * 70)
    print(f"  SIGNAL SUMMARY (inner={band_inner}, outer={band_outer})")
    print("=" * 70)
    print(f"  Total bars:                       {len(s)}")
    print(f"  Buy signals:                      {(s['signal'] == 1).sum()}")
    print(f"  Sell signals:                     {(s['signal'] == -1).sum()}")
    print()
    print("  --- Primary (VWAP bands) ---")
    print(f"  Inner lower band bounces:         {buy_inner.sum()} bars")
    print(f"  Outer lower band bounces:         {buy_outer.sum()} bars")
    print(f"  Combined buy bounces:             {buy_bounce.sum()} bars")
    print(f"  Inner upper band rejections:      {sell_inner.sum()} bars")
    print(f"  Outer upper band rejections:      {sell_outer.sum()} bars")
    print(f"  Combined sell rejections:         {sell_reject.sum()} bars")
    print()
    print("  --- Confirmation (volume) ---")
    print(f"  OBV rising:                       {obv_rising.sum()} bars")
    print(f"  OBV falling:                      {obv_falling.sum()} bars")
    print(f"  AD rising:                        {ad_rising.sum()} bars")
    print(f"  AD falling:                       {ad_falling.sum()} bars")
    print(f"  Buy confirmed (OBV|AD up):        {buy_confirmed.sum()} bars")
    print(f"  Sell confirmed (OBV|AD dn):       {sell_confirmed.sum()} bars")
    print()
    print("  --- Near misses ---")
    print(f"  Band bounce, no vol confirm:      {(buy_bounce & ~buy_confirmed).sum()} buy, "
          f"{(sell_reject & ~sell_confirmed).sum()} sell")
    print(f"  Vol confirmed, band missed:       check missed_trades output")


def main():
    parser = argparse.ArgumentParser(description="Diagnose indicator values")
    parser.add_argument("--timeframe", default="daily", choices=list(TIMEFRAMES.keys()))
    parser.add_argument("--period", default=None, help="yfinance period e.g. 2y, 60d")
    parser.add_argument("--obv-lookback", type=int, default=None)
    parser.add_argument("--ad-lookback", type=int, default=None)
    args = parser.parse_args()

    # Use timeframe-aware defaults
    defaults = get_defaults(args.timeframe)
    band_inner = defaults["band_multiplier_inner"]
    band_outer = defaults["band_multiplier_outer"]
    obv_lb = args.obv_lookback or defaults["obv_lookback"]
    ad_lb = args.ad_lookback or defaults["ad_lookback"]

    df = fetch_nifty50(timeframe=args.timeframe, period=args.period)
    print(f"\nFetched {len(df)} bars ({args.timeframe}): {df.index[0]} → {df.index[-1]}")
    print(f"Using: inner_band={band_inner}, outer_band={band_outer}, "
          f"obv_lb={obv_lb}, ad_lb={ad_lb}\n")
    diagnose(df, band_inner, band_outer, obv_lb, ad_lb)


if __name__ == "__main__":
    main()
