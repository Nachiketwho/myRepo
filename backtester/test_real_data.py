#!/usr/bin/env python3
"""Test SVP+VWAP strategy on realistic synthetic NIFTY data.

Since Yahoo Finance is blocked in this environment, we generate
synthetic data that mimics real NIFTY characteristics:
- Price range: 22,000 - 24,000
- Typical 5-min volatility patterns
- Volume patterns with spikes at support/resistance
- Different market regimes (uptrend, downtrend, ranging)
"""

import sys
sys.path.insert(0, "/home/user/myRepo")

import numpy as np
import pandas as pd

from backtester.svp_vwap import (
    SVPVWAPEngine,
    build_svp_trade_log,
    run_svp_vwap_grid,
)
from backtester.ema_crossover import SLStrategy, TrailingMode


def generate_nifty_data(
    n_sessions: int = 60,
    bars_per_session: int = 75,
    scenario: str = "mixed",
) -> pd.DataFrame:
    """Generate realistic NIFTY 5-minute data.

    Args:
        n_sessions: Number of trading sessions (days)
        bars_per_session: Bars per session (75 for 5-min = 6.25 hours)
        scenario: 'uptrend', 'downtrend', 'ranging', or 'mixed'

    Returns:
        OHLCV DataFrame with realistic NIFTY characteristics
    """
    total_bars = n_sessions * bars_per_session
    base_price = 22500.0  # Typical NIFTY level

    # Generate price based on scenario
    np.random.seed(42)

    if scenario == "uptrend":
        # Steady uptrend with pullbacks
        trend = np.linspace(0, 1500, total_bars)  # +1500 points over 60 days
        cycles = 100 * np.sin(np.arange(total_bars) / 30)  # Pullbacks
        noise = np.cumsum(np.random.randn(total_bars) * 3)  # Random walk
        close = base_price + trend + cycles + noise

    elif scenario == "downtrend":
        # Downtrend with bounces
        trend = np.linspace(0, -1200, total_bars)  # -1200 points
        cycles = 80 * np.sin(np.arange(total_bars) / 25)  # Bounces
        noise = np.cumsum(np.random.randn(total_bars) * 3)
        close = base_price + trend + cycles + noise

    elif scenario == "ranging":
        # Sideways market oscillating around base
        amplitude = 300
        period = total_bars // 6
        close = base_price + amplitude * np.sin(np.arange(total_bars) / period * 2 * np.pi)
        noise = np.cumsum(np.random.randn(total_bars) * 2)
        close = close + noise

    else:  # "mixed" - realistic with multiple regimes
        close = np.zeros(total_bars)
        close[0] = base_price

        for i in range(1, total_bars):
            session = i // bars_per_session
            phase = session % 4  # Cycle through regimes

            if phase == 0:  # Uptrend
                drift = 0.4
            elif phase == 1:  # Sideways
                drift = 0.0
            elif phase == 2:  # Downtrend
                drift = -0.3
            else:  # Recovery
                drift = 0.2

            # Add intraday patterns (volatile at open/close)
            bar_in_session = i % bars_per_session
            if bar_in_session < 10 or bar_in_session > 65:
                vol_mult = 1.5  # Higher volatility at open/close
            else:
                vol_mult = 1.0

            close[i] = close[i-1] + drift + np.random.randn() * 6 * vol_mult

    # Generate OHLC from close
    # Typical NIFTY 5-min bar range: 10-30 points
    bar_range = np.random.uniform(10, 30, total_bars)
    high_pct = np.random.uniform(0.3, 0.7, total_bars)  # Where in the range is close

    high = close + bar_range * high_pct
    low = close - bar_range * (1 - high_pct)
    open_price = low + np.random.uniform(0.2, 0.8, total_bars) * (high - low)

    # Generate volume with realistic patterns
    # Higher volume at session start/end, at support/resistance
    volume = np.random.randint(500000, 2000000, total_bars)

    for i in range(total_bars):
        bar_in_session = i % bars_per_session
        # Opening hour surge
        if bar_in_session < 12:
            volume[i] = int(volume[i] * 1.8)
        # Closing hour surge
        elif bar_in_session > 63:
            volume[i] = int(volume[i] * 1.5)

        # Volume spike on large moves
        if i > 0 and abs(close[i] - close[i-1]) > 15:
            volume[i] = int(volume[i] * 2.0)

    # Create DataFrame with datetime index
    dates = pd.date_range(
        "2025-12-01 09:15",
        periods=total_bars,
        freq="5min"
    )
    # Filter to market hours (9:15 AM to 3:30 PM)
    # For simplicity, just use consecutive dates

    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    return df


def main():
    print("=" * 70)
    print("SVP+VWAP Strategy - Realistic Synthetic NIFTY Backtest")
    print("=" * 70)
    print("\nNote: Using synthetic data that mimics NIFTY characteristics")
    print("      (Yahoo Finance blocked in this environment)")

    # Generate data for multiple scenarios
    scenarios = ["mixed", "uptrend", "downtrend", "ranging"]
    all_results = []

    for scenario in scenarios:
        print(f"\n{'=' * 70}")
        print(f"SCENARIO: {scenario.upper()}")
        print("=" * 70)

        # Generate 60 sessions of 5-min data (75 bars per session)
        df = generate_nifty_data(n_sessions=60, bars_per_session=75, scenario=scenario)
        print(f"\n[Data] {len(df)} bars | {df['close'].min():.0f} - {df['close'].max():.0f}")

        # Configuration optimized from previous testing
        config = {
            "session_lookback": 75,  # 75 bars = 1 session (5min)
            "capital": 100000.0,
            "lot_size": 25,
            "risk_per_trade_pct": 1.0,
            "tp_rr_ratio": 2.0,
            "atr_period": 14,
            "atr_multiplier": 2.0,
        }

        # Run with best configuration: ATR SL + Stepped Trailing
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.ATR,
            trailing_mode=TrailingMode.STEPPED,
            **config,
        )
        result = engine.run(df, symbol="NIFTY", timeframe="5min")

        # Display KPIs
        kpis = result.to_dict()
        print(f"\nTotal Trades:       {kpis['num_trades']}")
        print(f"Win Rate:           {kpis['win_rate']:.1f}%")
        print(f"Total P&L:          Rs.{kpis['total_pnl']:,.2f}")
        print(f"Profit Factor:      {kpis['profit_factor']:.2f}")
        print(f"Max Drawdown:       Rs.{kpis['max_drawdown']:,.2f} ({kpis['max_drawdown_pct']:.1f}%)")
        print(f"Avg R-Multiple:     {kpis['avg_r_multiple']:.2f}R")
        print(f"Sharpe Ratio:       {kpis['sharpe_ratio']:.2f}")

        # Signal type breakdown
        trade_log = build_svp_trade_log(result)
        if not trade_log.empty:
            print("\nSignal Type Breakdown:")
            sig_stats = trade_log.groupby("signal_type").agg({
                "pnl_amount": ["count", "sum"],
            }).round(0)
            sig_stats.columns = ["Trades", "P&L"]
            print(sig_stats.to_string())

        all_results.append({
            "scenario": scenario,
            **kpis
        })

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: ALL SCENARIOS")
    print("=" * 70)
    summary_df = pd.DataFrame(all_results)
    print(summary_df[["scenario", "num_trades", "win_rate", "total_pnl",
                      "profit_factor", "max_drawdown_pct"]].to_string(index=False))

    # Detailed test on mixed scenario with grid
    print("\n" + "=" * 70)
    print("CONFIGURATION GRID (Mixed Scenario)")
    print("=" * 70)

    df_mixed = generate_nifty_data(n_sessions=60, scenario="mixed")
    grid_results = run_svp_vwap_grid(
        df_mixed,
        symbol="NIFTY",
        timeframe="5min",
        sl_strategies=[SLStrategy.ATR, SLStrategy.SWING, SLStrategy.PERCENTAGE],
        trailing_modes=[TrailingMode.NONE, TrailingMode.BREAKEVEN, TrailingMode.STEPPED],
        capital=100000.0,
        session_lookback=75,
    )
    print(grid_results[["sl_strategy", "trailing_mode", "num_trades", "win_rate",
                        "total_pnl", "profit_factor"]].to_string())

    # Sample trades from best config
    print("\n" + "=" * 70)
    print("SAMPLE TRADES (Mixed Scenario - First 15)")
    print("=" * 70)
    engine = SVPVWAPEngine(
        sl_strategy=SLStrategy.ATR,
        trailing_mode=TrailingMode.STEPPED,
        session_lookback=75,
        capital=100000.0,
    )
    result = engine.run(df_mixed, symbol="NIFTY", timeframe="5min")
    trade_log = build_svp_trade_log(result)
    if not trade_log.empty:
        cols = ["entry_date", "direction", "signal_type", "trade_instrument",
                "entry_price", "exit_price", "exit_reason", "pnl_points", "pnl_amount"]
        print(trade_log[cols].head(15).to_string())

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
