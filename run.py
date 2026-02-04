"""Run full backtest pipeline on Nifty50 data.

Usage:
    python run.py                        # fetches live data from Yahoo Finance
    python run.py --timeframes daily weekly 5min
    python run.py --timeframes 15min     # uses 15min-specific parameters
    python run.py --fno                  # F&O analysis (ATM + OTM × 4 expiries)
    python run.py --fno --timeframes 15min
"""

import argparse
import os
import pandas as pd

from backtester.data import fetch_nifty50, TIMEFRAMES
from backtester.strategy import (
    VolumeStrategy, get_defaults,
    PARAM_GRID, ENGINE_PARAM_GRID,
    PARAM_GRID_15MIN, ENGINE_PARAM_GRID_15MIN,
)
from backtester.engine import BacktestEngine
from backtester.optimizer import run_optimization
from backtester.missed_trades import find_missed_trades
from backtester.report import format_report, export_csv, build_trade_log
from backtester.data import fetch_india_vix
from backtester.fno_engine import (
    FnOEngine, FNO_DEFAULTS, run_fno_analysis, build_fno_trade_log,
)


def _grids_for_timeframe(tf: str):
    """Return (strategy_grid, engine_grid) appropriate for the timeframe."""
    if tf in ("15min", "5min"):
        return PARAM_GRID_15MIN, ENGINE_PARAM_GRID_15MIN
    return PARAM_GRID, ENGINE_PARAM_GRID


def run_for_timeframe(df: pd.DataFrame, label: str, output_dir: str):
    """Run strategy, optimize, find missed trades, export CSV for one timeframe."""
    defaults = get_defaults(label)
    vwap_win = defaults.get("vwap_window")

    print(f"\n{'='*60}")
    print(f"  TIMEFRAME: {label}  ({len(df)} bars)")
    print(f"  Range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Defaults: inner_band={defaults['band_multiplier_inner']}, "
          f"outer_band={defaults['band_multiplier_outer']}, "
          f"vwap_window={vwap_win}, "
          f"SL={defaults['sl_pct']}%, TP={defaults['tp_pct']}%")
    print(f"{'='*60}")

    # --- 1. Default-param run (timeframe-aware) ---
    strategy = VolumeStrategy(
        band_multiplier_inner=defaults["band_multiplier_inner"],
        band_multiplier_outer=defaults["band_multiplier_outer"],
        vwap_window=vwap_win,
        obv_lookback=defaults["obv_lookback"],
        ad_lookback=defaults["ad_lookback"],
    )
    signals = strategy.generate_signals(df)
    engine = BacktestEngine(
        sl_pct=defaults["sl_pct"],
        tsl_pct=defaults["tsl_pct"],
        tp_pct=defaults["tp_pct"],
        ttp_pct=defaults["ttp_pct"],
    )
    result = engine.run(signals)

    print("\n--- Default params run ---")
    print(format_report(result))

    # --- 2. Optimizer (timeframe-aware grids) ---
    strat_grid, eng_grid = _grids_for_timeframe(label)
    print("\n--- Optimizer (finding best params) ---")
    opt_results = run_optimization(df, strategy_grid=strat_grid,
                                    engine_grid=eng_grid, min_trades=3,
                                    vwap_window=vwap_win)
    if not opt_results.empty:
        print(f"  Tested {len(opt_results)} valid combos")
        top5 = opt_results.head(5)
        print("\n  Top 5 by profit:")
        print(top5.to_string(index=False))

        bp = opt_results.iloc[0].to_dict()
        print(f"\n  BEST: inner_band={bp.get('band_multiplier_inner')}, "
              f"outer_band={bp.get('band_multiplier_outer')}, "
              f"obv_lb={int(bp['obv_lookback'])}, ad_lb={int(bp['ad_lookback'])}, "
              f"SL={bp['sl_pct']}%, TSL={bp['tsl_pct']}%, "
              f"TP={bp['tp_pct']}%, TTP={bp['ttp_pct']}%")
        print(f"  → {int(bp['num_trades'])} trades, PnL={bp['total_pnl']:.2f}, "
              f"WR={bp['win_rate']:.1f}%, PF={bp['profit_factor']:.2f}")

        # Run with best params
        best_strat = VolumeStrategy(
            band_multiplier_inner=bp.get("band_multiplier_inner", 1.0),
            band_multiplier_outer=bp.get("band_multiplier_outer", 2.0),
            vwap_window=vwap_win,
            obv_lookback=int(bp["obv_lookback"]),
            ad_lookback=int(bp["ad_lookback"]),
        )
        best_signals = best_strat.generate_signals(df)
        best_engine = BacktestEngine(
            sl_pct=bp["sl_pct"], tsl_pct=bp["tsl_pct"],
            tp_pct=bp["tp_pct"], ttp_pct=bp["ttp_pct"],
        )
        best_result = best_engine.run(best_signals)

        # Export best-param results
        tl_path = os.path.join(output_dir, f"trade_log_{label}.csv")
        sm_path = os.path.join(output_dir, f"summary_{label}.csv")
        export_csv(best_result, tl_path, sm_path)
        print(f"\n  Exported: {tl_path}, {sm_path}")

        # Export trade log for display
        log = build_trade_log(best_result)
        print(f"\n--- Trade Log ({label}, best params) ---")
        print(log.to_string(index=False))

        # Export optimizer results
        opt_path = os.path.join(output_dir, f"optimizer_{label}.csv")
        opt_results.to_csv(opt_path, index=False)
        print(f"\n  Optimizer results: {opt_path}")
    else:
        print("  No combos met min_trades threshold.")

    # --- 3. Missed trades (timeframe-aware) ---
    missed = find_missed_trades(
        df,
        band_multiplier_inner=defaults["band_multiplier_inner"],
        band_multiplier_outer=defaults["band_multiplier_outer"],
        rolling_window=vwap_win,
        obv_lookback=defaults["obv_lookback"],
        ad_lookback=defaults["ad_lookback"],
    )
    print(f"\n--- Missed Trades ({len(missed)} near-misses) ---")
    if not missed.empty:
        print(missed.head(10).to_string(index=False))
        missed_path = os.path.join(output_dir, f"missed_trades_{label}.csv")
        missed.to_csv(missed_path, index=False)
        print(f"\n  Full list: {missed_path}")
    else:
        print("  None found.")


def run_fno_for_timeframe(
    df: pd.DataFrame, label: str, output_dir: str,
    sl_points: float = 50.0, tp_points: float = 50.0,
    num_otm: int = 4, num_expiries: int = 4,
):
    """Run F&O strike × expiry analysis for one timeframe."""
    defaults = get_defaults(label)
    vwap_win = defaults.get("vwap_window")

    print(f"\n{'='*60}")
    print(f"  F&O ANALYSIS: {label}  ({len(df)} bars)")
    print(f"  Range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  SL={sl_points}pts, TP={tp_points}pts, "
          f"strikes=ATM+{num_otm} OTM, expiries={num_expiries}")
    print(f"{'='*60}")

    # Generate signals from spot strategy
    strategy = VolumeStrategy(
        band_multiplier_inner=defaults["band_multiplier_inner"],
        band_multiplier_outer=defaults["band_multiplier_outer"],
        vwap_window=vwap_win,
        obv_lookback=defaults["obv_lookback"],
        ad_lookback=defaults["ad_lookback"],
    )
    signals = strategy.generate_signals(df)
    num_signals = (signals["signal"] != 0).sum()
    print(f"\n  Spot signals generated: {num_signals} "
          f"({(signals['signal']==1).sum()} buy, "
          f"{(signals['signal']==-1).sum()} sell)")

    if num_signals == 0:
        print("  No signals — skipping F&O analysis.")
        return

    # Fetch India VIX
    print("  Fetching India VIX...")
    try:
        vix = fetch_india_vix()
        print(f"  VIX data: {len(vix)} days "
              f"(mean={vix.mean():.1f}%, last={vix.iloc[-1]:.1f}%)")
    except Exception as e:
        print(f"  VIX fetch failed ({e}), using default IV=15%")
        vix = pd.Series(dtype=float)

    # Run strike × expiry sweep
    print(f"\n  Running {num_otm + 1} strikes × {num_expiries} expiries "
          f"= {(num_otm + 1) * num_expiries} combos...")
    summary = run_fno_analysis(
        signals, vix,
        num_otm=num_otm,
        num_expiries=num_expiries,
        sl_points=sl_points,
        tp_points=tp_points,
    )

    if summary.empty:
        print("  No trades generated across any combo.")
        return

    print(f"\n--- F&O Strike × Expiry Summary ({len(summary)} combos with trades) ---")
    print(summary.to_string(index=False))

    # Export summary
    summary_path = os.path.join(output_dir, f"fno_summary_{label}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n  Summary: {summary_path}")

    # Show best combo trade log
    best = summary.iloc[0]
    print(f"\n--- Best combo: {best['strike_type']} / week {int(best['expiry_week'])} ---")
    print(f"  Trades={int(best['num_trades'])}, PnL={best['total_pnl']:.2f}, "
          f"WR={best['win_rate']:.1f}%, Avg hold={best['avg_hold_days']:.1f}d")

    engine = FnOEngine(sl_points=sl_points, tp_points=tp_points)
    best_result = engine.run(
        signals, vix,
        strike_offset=int(best["strike_offset"]),
        expiry_week=int(best["expiry_week"]),
    )
    trade_log = build_fno_trade_log(best_result)
    print(f"\n--- Trade Log (best combo, {len(trade_log)} trades) ---")
    print(trade_log.to_string(index=False))

    log_path = os.path.join(output_dir, f"fno_trades_{label}.csv")
    trade_log.to_csv(log_path, index=False)
    print(f"\n  Trade log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Nifty50 Backtester")
    parser.add_argument(
        "--timeframes", nargs="+", default=["daily"],
        choices=list(TIMEFRAMES.keys()),
        help="Timeframes to test (default: daily)",
    )
    parser.add_argument("--output", default="output", help="Output directory for CSVs")
    parser.add_argument("--fno", action="store_true",
                        help="Run F&O strike × expiry analysis")
    parser.add_argument("--sl-points", type=float, default=50.0,
                        help="F&O stop-loss in premium points (default: 50)")
    parser.add_argument("--tp-points", type=float, default=50.0,
                        help="F&O take-profit in premium points (default: 50)")
    parser.add_argument("--num-otm", type=int, default=4,
                        help="Number of OTM strikes to test (default: 4)")
    parser.add_argument("--num-expiries", type=int, default=4,
                        help="Number of weekly expiries to test (default: 4)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for tf in args.timeframes:
        df = fetch_nifty50(timeframe=tf)

        if args.fno:
            run_fno_for_timeframe(
                df, tf, args.output,
                sl_points=args.sl_points,
                tp_points=args.tp_points,
                num_otm=args.num_otm,
                num_expiries=args.num_expiries,
            )
        else:
            run_for_timeframe(df, tf, args.output)

    print(f"\n{'='*60}")
    print(f"  All CSVs saved to: {args.output}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
