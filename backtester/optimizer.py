from itertools import product
import pandas as pd

from backtester.strategy import VolumeStrategy, PARAM_GRID, ENGINE_PARAM_GRID
from backtester.engine import BacktestEngine, BacktestResult


STRAT_PARAM_KEYS = ("band_multiplier_inner", "band_multiplier_outer",
                     "obv_lookback", "ad_lookback")
ENGINE_PARAM_KEYS = ("sl_pct", "tsl_pct", "tp_pct", "ttp_pct")


def _build_combos(
    strategy_grid: dict | None = None,
    engine_grid: dict | None = None,
) -> list[dict]:
    """Return flat list of combined strategy + engine param dicts."""
    strategy_grid = strategy_grid or PARAM_GRID
    engine_grid = engine_grid or ENGINE_PARAM_GRID

    merged = {**strategy_grid, **engine_grid}
    keys = list(merged.keys())
    combos = []
    for vals in product(*merged.values()):
        combos.append(dict(zip(keys, vals)))
    return combos


def run_optimization(
    df: pd.DataFrame,
    strategy_grid: dict | None = None,
    engine_grid: dict | None = None,
    min_trades: int = 5,
    vwap_window: int | None = None,
) -> pd.DataFrame:
    """Grid-search over strategy + engine params.

    Args:
        vwap_window: Rolling window for VWAP band calculation.
            Pass the timeframe-appropriate value (e.g. 20 for 15min).

    Returns a DataFrame sorted by total_pnl descending with one row per
    parameter combination, filtered to at least *min_trades*.
    """
    combos = _build_combos(strategy_grid, engine_grid)
    rows: list[dict] = []

    for combo in combos:
        strat_params = {k: combo[k] for k in STRAT_PARAM_KEYS if k in combo}
        strat_params["vwap_window"] = vwap_window
        engine_params = {k: combo[k] for k in ENGINE_PARAM_KEYS if k in combo}

        strategy = VolumeStrategy(**strat_params)
        signals = strategy.generate_signals(df)
        engine = BacktestEngine(**engine_params)
        result = engine.run(signals)

        if result.num_trades < min_trades:
            continue

        row = {**combo}
        row["num_trades"] = result.num_trades
        row["total_pnl"] = round(result.total_pnl, 2)
        row["total_pnl_pct"] = round(result.total_pnl_pct, 2)
        row["win_rate"] = round(result.win_rate, 2)
        row["profit_factor"] = round(result.profit_factor, 2)
        row["max_drawdown"] = round(result.max_drawdown, 2)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df

    results_df.sort_values("total_pnl", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    return results_df


def best_params(
    df: pd.DataFrame,
    strategy_grid: dict | None = None,
    engine_grid: dict | None = None,
    min_trades: int = 5,
    vwap_window: int | None = None,
) -> dict:
    """Return the single best parameter set (highest profit, ≥ min_trades)."""
    results = run_optimization(df, strategy_grid, engine_grid, min_trades,
                                vwap_window=vwap_window)
    if results.empty:
        return {}
    return results.iloc[0].to_dict()
