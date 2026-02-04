"""F&O parameter optimizer — grid search over risk management params.

Sweeps base_sl, rr_ratio, ema_period, min_signal_strength across
strike × expiry combinations to find the best risk-adjusted config.
"""

from itertools import product
import pandas as pd

from backtester.fno_engine import FnOEngine, FnOResult, run_fno_analysis


# Default parameter grids for F&O optimization
FNO_SL_GRID = [20, 25, 30, 35, 40]
FNO_RR_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
FNO_EMA_GRID = [9, 13, 21, 34, 50]
FNO_STRENGTH_GRID = [1, 2, 3]

FNO_PARAM_GRID = {
    "base_sl": FNO_SL_GRID,
    "rr_ratio": FNO_RR_GRID,
    "ema_period": FNO_EMA_GRID,
    "min_signal_strength": FNO_STRENGTH_GRID,
}


def _build_fno_combos(param_grid: dict | None = None) -> list[dict]:
    """Return flat list of F&O parameter dicts from a grid."""
    grid = param_grid or FNO_PARAM_GRID
    keys = list(grid.keys())
    combos = []
    for vals in product(*grid.values()):
        combos.append(dict(zip(keys, vals)))
    return combos


def run_fno_optimization(
    signals_df: pd.DataFrame,
    vix_series: pd.Series,
    param_grid: dict | None = None,
    num_otm: int = 4,
    num_expiries: int = 4,
    min_trades: int = 3,
    **engine_kwargs,
) -> pd.DataFrame:
    """Grid-search over F&O risk management parameters.

    For each parameter combo, runs the full strike × expiry sweep via
    run_fno_analysis and records aggregate metrics.

    Args:
        signals_df: DataFrame with 'signal' column and OHLCV.
        vix_series: India VIX daily close (in %).
        param_grid: Dict of param_name → list of values to sweep.
            Defaults to FNO_PARAM_GRID.
        num_otm: Number of OTM strikes (0=ATM only).
        num_expiries: Number of weekly expiries.
        min_trades: Minimum trades required to include a combo.
        **engine_kwargs: Additional FnOEngine params (e.g. lot_size).

    Returns:
        DataFrame sorted by total_pnl descending, one row per param combo.
    """
    combos = _build_fno_combos(param_grid)
    rows: list[dict] = []

    for combo in combos:
        merged = {**engine_kwargs, **combo}

        try:
            summary = run_fno_analysis(
                signals_df, vix_series,
                num_otm=num_otm,
                num_expiries=num_expiries,
                **merged,
            )
        except Exception:
            continue

        if summary.empty:
            continue

        total_trades = int(summary["num_trades"].sum())
        if total_trades < min_trades:
            continue

        total_pnl = round(summary["total_pnl"].sum(), 2)
        avg_win_rate = round(summary["win_rate"].mean(), 1)
        best_combo_pnl = round(summary["total_pnl"].iloc[0], 2)
        best_strike = summary.iloc[0]["strike_type"]
        best_expiry = int(summary.iloc[0]["expiry_week"])
        num_combos = len(summary)

        row = {**combo}
        row["total_trades"] = total_trades
        row["total_pnl"] = total_pnl
        row["avg_win_rate"] = avg_win_rate
        row["strike_expiry_combos"] = num_combos
        row["best_combo_pnl"] = best_combo_pnl
        row["best_strike"] = best_strike
        row["best_expiry_week"] = best_expiry
        rows.append(row)

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df

    results_df.sort_values("total_pnl", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    return results_df


def best_fno_params(
    signals_df: pd.DataFrame,
    vix_series: pd.Series,
    param_grid: dict | None = None,
    num_otm: int = 4,
    num_expiries: int = 4,
    min_trades: int = 3,
    **engine_kwargs,
) -> dict:
    """Return the single best F&O parameter set (highest total PnL)."""
    results = run_fno_optimization(
        signals_df, vix_series,
        param_grid=param_grid,
        num_otm=num_otm,
        num_expiries=num_expiries,
        min_trades=min_trades,
        **engine_kwargs,
    )
    if results.empty:
        return {}
    return results.iloc[0].to_dict()
