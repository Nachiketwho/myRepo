#!/usr/bin/env python3
"""Run EMA crossover backtests on multiple timeframes and symbols.

Usage:
    python run_ema_backtest.py --timeframe 15min --symbol NIFTY
    python run_ema_backtest.py --timeframe 5min --all-symbols
    python run_ema_backtest.py --compare-all

Symbols: NIFTY, BANKNIFTY, RELIANCE, TCS, INFY, HDFC, ICICIBANK
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

from backtester.ema_crossover import (
    EMACrossoverEngine, EMACrossoverResult,
    SLStrategy, PositionSizing, EMA_PAIRS,
    run_ema_backtest_grid, build_ema_trade_log,
    compare_for_trading_style,
)


# Major Indian stocks and indices for testing
SYMBOLS = {
    "NIFTY": {"lot_size": 25, "tick_size": 0.05},
    "BANKNIFTY": {"lot_size": 15, "tick_size": 0.05},
    "RELIANCE": {"lot_size": 250, "tick_size": 0.05},
    "TCS": {"lot_size": 150, "tick_size": 0.05},
    "INFY": {"lot_size": 300, "tick_size": 0.05},
    "HDFCBANK": {"lot_size": 550, "tick_size": 0.05},
    "ICICIBANK": {"lot_size": 700, "tick_size": 0.05},
}

TIMEFRAMES = ["5min", "15min"]


def generate_sample_data(
    symbol: str,
    timeframe: str,
    days: int = 90,
    base_price: float = None,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """Generate realistic sample OHLCV data for testing.

    Creates data with trend, mean-reversion, and volatility clustering
    to simulate real market conditions.
    """
    np.random.seed(hash(f"{symbol}_{timeframe}") % 2**32)

    # Determine bars per day based on timeframe
    if timeframe == "5min":
        bars_per_day = 75  # 6.25 hours * 12 bars
    elif timeframe == "15min":
        bars_per_day = 25  # 6.25 hours * 4 bars
    else:
        bars_per_day = 1

    total_bars = days * bars_per_day

    # Base prices for different symbols
    base_prices = {
        "NIFTY": 22000, "BANKNIFTY": 48000, "RELIANCE": 2800,
        "TCS": 3800, "INFY": 1600, "HDFCBANK": 1700, "ICICIBANK": 1100,
    }
    if base_price is None:
        base_price = base_prices.get(symbol, 1000)

    # Generate returns with trend and mean-reversion
    trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3]) * 0.0001
    returns = np.random.normal(trend, volatility / np.sqrt(bars_per_day), total_bars)

    # Add volatility clustering (GARCH-like)
    vol_factor = np.ones(total_bars)
    for i in range(1, total_bars):
        vol_factor[i] = 0.9 * vol_factor[i-1] + 0.1 * abs(returns[i-1]) / volatility * np.sqrt(bars_per_day)
    returns = returns * np.clip(vol_factor, 0.5, 2.0)

    # Generate price series
    close = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close
    noise = np.random.uniform(0.001, 0.005, total_bars)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_price = np.roll(close, 1)
    open_price[0] = base_price

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    # Generate volume with activity patterns
    base_volume = 100000 if symbol in ["NIFTY", "BANKNIFTY"] else 50000
    volume = base_volume * (1 + np.abs(returns) * 10) * np.random.uniform(0.5, 1.5, total_bars)

    # Create datetime index
    start = pd.Timestamp("2024-01-01 09:15:00")
    if timeframe == "5min":
        freq = "5min"
    elif timeframe == "15min":
        freq = "15min"
    else:
        freq = "D"

    dates = pd.date_range(start, periods=total_bars, freq=freq)
    # Filter to trading hours (9:15 - 15:30)
    dates = dates[(dates.hour >= 9) & (dates.hour < 16)][:total_bars]

    if len(dates) < total_bars:
        dates = pd.date_range(start, periods=total_bars, freq=freq)

    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume.astype(int),
    }, index=dates[:len(close)])

    return df


def run_single_backtest(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    ema_fast: int,
    ema_slow: int,
    sl_strategy: SLStrategy,
    tp_rr_ratio: float = 2.0,
    capital: float = 100000.0,
) -> EMACrossoverResult:
    """Run a single backtest configuration."""
    lot_size = SYMBOLS.get(symbol, {}).get("lot_size", 25)

    engine = EMACrossoverEngine(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        sl_strategy=sl_strategy,
        position_sizing=PositionSizing.RISK_BASED,
        tp_rr_ratio=tp_rr_ratio,
        capital=capital,
        risk_per_trade_pct=1.0,
        lot_size=lot_size,
    )

    return engine.run(df, symbol, timeframe)


def run_grid_backtest(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    capital: float = 100000.0,
) -> pd.DataFrame:
    """Run full grid of EMA pairs and SL strategies."""
    lot_size = SYMBOLS.get(symbol, {}).get("lot_size", 25)

    return run_ema_backtest_grid(
        df,
        symbol=symbol,
        timeframe=timeframe,
        ema_pairs=EMA_PAIRS,
        sl_strategies=list(SLStrategy),
        tp_rr_ratio=2.0,
        capital=capital,
        risk_per_trade_pct=1.0,
        lot_size=lot_size,
    )


def print_kpi_summary(results_df: pd.DataFrame, title: str = ""):
    """Print formatted KPI summary."""
    if results_df.empty:
        print("No results to display.")
        return

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    # Top 5 by total PnL
    print("\n--- TOP 5 by Total PnL ---")
    top5 = results_df.head(5)
    print(top5[[
        "ema_pair", "sl_strategy", "num_trades", "win_rate",
        "total_pnl", "profit_factor", "expectancy", "max_drawdown_pct"
    ]].to_string(index=False))

    # Best by win rate (min 10 trades)
    filtered = results_df[results_df["num_trades"] >= 10]
    if not filtered.empty:
        print("\n--- Best by Win Rate (min 10 trades) ---")
        best_wr = filtered.nlargest(3, "win_rate")
        print(best_wr[[
            "ema_pair", "sl_strategy", "num_trades", "win_rate",
            "total_pnl", "profit_factor"
        ]].to_string(index=False))

    # Best by profit factor
    if not filtered.empty:
        print("\n--- Best by Profit Factor (min 10 trades) ---")
        best_pf = filtered.nlargest(3, "profit_factor")
        print(best_pf[[
            "ema_pair", "sl_strategy", "num_trades", "win_rate",
            "total_pnl", "profit_factor", "avg_r_multiple"
        ]].to_string(index=False))

    # Best by expectancy
    if not filtered.empty:
        print("\n--- Best by Expectancy (min 10 trades) ---")
        best_exp = filtered.nlargest(3, "expectancy")
        print(best_exp[[
            "ema_pair", "sl_strategy", "num_trades", "expectancy",
            "avg_r_multiple", "sharpe_ratio"
        ]].to_string(index=False))

    # SL Strategy comparison
    print("\n--- SL Strategy Comparison (aggregated) ---")
    sl_summary = results_df.groupby("sl_strategy").agg({
        "num_trades": "sum",
        "win_rate": "mean",
        "total_pnl": "sum",
        "profit_factor": "mean",
        "expectancy": "mean",
    }).round(2)
    print(sl_summary.to_string())

    # EMA Pair comparison
    print("\n--- EMA Pair Comparison (aggregated) ---")
    ema_summary = results_df.groupby("ema_pair").agg({
        "num_trades": "sum",
        "win_rate": "mean",
        "total_pnl": "sum",
        "profit_factor": "mean",
        "expectancy": "mean",
        "avg_bars_held": "mean",
    }).round(2)
    print(ema_summary.to_string())


def print_trading_style_recommendations(results_df: pd.DataFrame):
    """Print recommendations for intraday vs swing trading."""
    recommendations = compare_for_trading_style(results_df)

    print(f"\n{'='*80}")
    print("  TRADING STYLE RECOMMENDATIONS")
    print(f"{'='*80}")

    if recommendations["intraday"]:
        print("\n--- INTRADAY (Scalping/Day Trading) ---")
        rec = recommendations["intraday"]
        print(f"  Best EMA Pair:    {rec['ema_pair']}")
        print(f"  Best SL Strategy: {rec['sl_strategy']}")
        print(f"  Win Rate:         {rec['win_rate']:.1f}%")
        print(f"  Profit Factor:    {rec['profit_factor']:.2f}")
        print(f"  Avg Bars Held:    {rec['avg_bars_held']:.1f}")
        print(f"  Total PnL:        Rs. {rec['total_pnl']:,.2f}")

    if recommendations["swing"]:
        print("\n--- SWING TRADING (Multi-day) ---")
        rec = recommendations["swing"]
        print(f"  Best EMA Pair:    {rec['ema_pair']}")
        print(f"  Best SL Strategy: {rec['sl_strategy']}")
        print(f"  Expectancy (R):   {rec['expectancy']:.3f}")
        print(f"  Avg R-Multiple:   {rec['avg_r_multiple']:.2f}")
        print(f"  Calmar Ratio:     {rec['calmar_ratio']:.2f}")
        print(f"  Total PnL:        Rs. {rec['total_pnl']:,.2f}")


def run_multi_symbol_comparison(timeframe: str, output_dir: str):
    """Run backtests across all symbols and compare."""
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nRunning EMA crossover backtests for {timeframe} timeframe...")
    print(f"Symbols: {', '.join(SYMBOLS.keys())}")

    for symbol in SYMBOLS:
        print(f"\n  Processing {symbol}...")

        # Generate sample data (replace with real data loading)
        df = generate_sample_data(symbol, timeframe, days=90)

        # Run grid backtest
        results = run_grid_backtest(df, symbol, timeframe)
        results["symbol"] = symbol
        all_results.append(results)

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)

    # Print summary
    print_kpi_summary(combined, f"All Symbols - {timeframe}")

    # Symbol-wise best performers
    print(f"\n--- Best Performer by Symbol ---")
    for symbol in SYMBOLS:
        symbol_df = combined[combined["symbol"] == symbol]
        if not symbol_df.empty:
            best = symbol_df.iloc[0]
            print(f"  {symbol:12s}: {best['ema_pair']:6s} + {best['sl_strategy']:10s} "
                  f"-> PnL: Rs.{best['total_pnl']:>10,.0f}, WR: {best['win_rate']:>5.1f}%")

    # Save combined results
    combined_path = os.path.join(output_dir, f"ema_all_symbols_{timeframe}_{timestamp}.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nResults saved: {combined_path}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="EMA Crossover Backtest")
    parser.add_argument("--symbol", default="NIFTY",
                        choices=list(SYMBOLS.keys()),
                        help="Symbol to test")
    parser.add_argument("--timeframe", default="15min",
                        choices=TIMEFRAMES,
                        help="Timeframe (5min or 15min)")
    parser.add_argument("--days", type=int, default=90,
                        help="Number of days of data")
    parser.add_argument("--capital", type=float, default=100000.0,
                        help="Trading capital")
    parser.add_argument("--all-symbols", action="store_true",
                        help="Run on all symbols")
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all symbols and timeframes")
    parser.add_argument("--output", default="output",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.compare_all:
        # Run comprehensive comparison
        print("\n" + "="*80)
        print("  COMPREHENSIVE EMA CROSSOVER ANALYSIS")
        print("="*80)

        all_combined = []
        for tf in TIMEFRAMES:
            results = run_multi_symbol_comparison(tf, args.output)
            all_combined.append(results)

        # Final combined analysis
        final = pd.concat(all_combined, ignore_index=True)

        print(f"\n{'='*80}")
        print("  FINAL RECOMMENDATIONS")
        print(f"{'='*80}")

        # Best overall for 5min
        df_5m = final[final["timeframe"] == "5min"]
        if not df_5m.empty:
            best_5m = df_5m.iloc[0]
            print(f"\n  5-MINUTE TIMEFRAME:")
            print(f"    Best: {best_5m['symbol']} with {best_5m['ema_pair']} + {best_5m['sl_strategy']}")
            print(f"    PnL: Rs.{best_5m['total_pnl']:,.0f}, WR: {best_5m['win_rate']:.1f}%, "
                  f"PF: {best_5m['profit_factor']:.2f}")

        # Best overall for 15min
        df_15m = final[final["timeframe"] == "15min"]
        if not df_15m.empty:
            best_15m = df_15m.iloc[0]
            print(f"\n  15-MINUTE TIMEFRAME:")
            print(f"    Best: {best_15m['symbol']} with {best_15m['ema_pair']} + {best_15m['sl_strategy']}")
            print(f"    PnL: Rs.{best_15m['total_pnl']:,.0f}, WR: {best_15m['win_rate']:.1f}%, "
                  f"PF: {best_15m['profit_factor']:.2f}")

        print_trading_style_recommendations(final)

        # Save final
        final_path = os.path.join(args.output, f"ema_comprehensive_{timestamp}.csv")
        final.to_csv(final_path, index=False)
        print(f"\nComprehensive results: {final_path}")

    elif args.all_symbols:
        run_multi_symbol_comparison(args.timeframe, args.output)

    else:
        # Single symbol backtest
        print(f"\n{'='*80}")
        print(f"  EMA CROSSOVER BACKTEST: {args.symbol} - {args.timeframe}")
        print(f"{'='*80}")

        df = generate_sample_data(args.symbol, args.timeframe, days=args.days)
        print(f"\nData: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

        results = run_grid_backtest(df, args.symbol, args.timeframe, args.capital)
        print_kpi_summary(results, f"{args.symbol} - {args.timeframe}")
        print_trading_style_recommendations(results)

        # Show trade log for best config
        if not results.empty:
            best = results.iloc[0]
            engine = EMACrossoverEngine(
                ema_fast=best["ema_fast"],
                ema_slow=best["ema_slow"],
                sl_strategy=SLStrategy(best["sl_strategy"]),
                capital=args.capital,
                lot_size=SYMBOLS[args.symbol]["lot_size"],
            )
            result = engine.run(df, args.symbol, args.timeframe)
            trade_log = build_ema_trade_log(result)

            if not trade_log.empty:
                print(f"\n--- Trade Log (Best Config: {best['ema_pair']} + {best['sl_strategy']}) ---")
                print(trade_log.head(10).to_string(index=False))

            # Save outputs
            results_path = os.path.join(args.output, f"ema_{args.symbol}_{args.timeframe}_{timestamp}.csv")
            results.to_csv(results_path, index=False)
            print(f"\nResults: {results_path}")

            log_path = os.path.join(args.output, f"ema_trades_{args.symbol}_{args.timeframe}_{timestamp}.csv")
            trade_log.to_csv(log_path, index=False)
            print(f"Trade log: {log_path}")


if __name__ == "__main__":
    main()
