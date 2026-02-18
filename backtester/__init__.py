from backtester.indicators import vwap, vwap_bands, obv, ad_line
from backtester.strategy import (
    VolumeStrategy, TIMEFRAME_DEFAULTS, get_defaults,
    PARAM_GRID_15MIN, ENGINE_PARAM_GRID_15MIN,
)
from backtester.engine import BacktestEngine, BacktestResult, Trade
from backtester.optimizer import run_optimization, best_params
from backtester.missed_trades import find_missed_trades
from backtester.report import build_summary, build_trade_log, format_report, export_csv
from backtester.data import fetch_nifty50, validate_ohlcv, DataValidationError  # noqa: lazy yf import inside
from backtester.greeks import bs_price, greeks_snapshot, implied_vol
from backtester.strikes import (
    get_atm_strike, get_otm_strikes, get_strike_range,
    next_thursdays, next_expiry_dates, is_trading_day, adjust_expiry,
    NSE_HOLIDAYS, LOT_SIZE, STRIKE_INTERVAL,
)
from backtester.fno_engine import (
    FnOEngine, FnOTrade, FnOResult, FnOExitReason,
    FNO_DEFAULTS, ADAPTIVE_SL_MAP, TRAIL_MILESTONES,
    run_fno_analysis, build_fno_trade_log,
)
from backtester.costs import TxnCosts, NSE_FNO_COSTS
from backtester.phantom_trades import find_phantom_trades as find_phantom_trades_fno
from backtester.fno_optimizer import (
    run_fno_optimization, best_fno_params,
    FNO_PARAM_GRID, FNO_SL_GRID, FNO_RR_GRID, FNO_EMA_GRID, FNO_STRENGTH_GRID,
)
from backtester.ema_crossover import (
    EMACrossoverEngine, EMACrossoverResult, EMACrossoverTrade,
    SLStrategy, PositionSizing, SLCalculator, PositionSizer,
    EMACrossoverSignals, EMA_PAIRS,
    run_ema_backtest_grid, build_ema_trade_log, compare_for_trading_style,
    ema, atr, swing_high, swing_low,
)

__all__ = [
    "vwap",
    "vwap_bands",
    "obv",
    "ad_line",
    "VolumeStrategy",
    "TIMEFRAME_DEFAULTS",
    "get_defaults",
    "PARAM_GRID_15MIN",
    "ENGINE_PARAM_GRID_15MIN",
    "BacktestEngine",
    "BacktestResult",
    "Trade",
    "run_optimization",
    "best_params",
    "find_missed_trades",
    "build_summary",
    "build_trade_log",
    "format_report",
    "export_csv",
    "fetch_nifty50",
    "validate_ohlcv",
    "DataValidationError",
    # Black-Scholes & Greeks
    "bs_price",
    "greeks_snapshot",
    "implied_vol",
    # Strike selection & expiry
    "get_atm_strike",
    "get_otm_strikes",
    "get_strike_range",
    "next_thursdays",
    "next_expiry_dates",
    "is_trading_day",
    "adjust_expiry",
    "NSE_HOLIDAYS",
    "LOT_SIZE",
    "STRIKE_INTERVAL",
    # F&O engine
    "FnOEngine",
    "FnOTrade",
    "FnOResult",
    "FnOExitReason",
    "FNO_DEFAULTS",
    "ADAPTIVE_SL_MAP",
    "TRAIL_MILESTONES",
    "run_fno_analysis",
    "build_fno_trade_log",
    # Transaction costs
    "TxnCosts",
    "NSE_FNO_COSTS",
    # Phantom trades
    "find_phantom_trades_fno",
    # F&O optimizer
    "run_fno_optimization",
    "best_fno_params",
    "FNO_PARAM_GRID",
    "FNO_SL_GRID",
    "FNO_RR_GRID",
    "FNO_EMA_GRID",
    "FNO_STRENGTH_GRID",
    # EMA crossover strategy
    "EMACrossoverEngine",
    "EMACrossoverResult",
    "EMACrossoverTrade",
    "SLStrategy",
    "PositionSizing",
    "SLCalculator",
    "PositionSizer",
    "EMACrossoverSignals",
    "EMA_PAIRS",
    "run_ema_backtest_grid",
    "build_ema_trade_log",
    "compare_for_trading_style",
    "ema",
    "atr",
    "swing_high",
    "swing_low",
]
