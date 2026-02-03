import pandas as pd
from backtester.engine import BacktestResult, ExitReason


def build_trade_log(result: BacktestResult) -> pd.DataFrame:
    """Flat DataFrame of all trades."""
    rows = []
    for t in result.trades:
        rows.append(
            {
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "direction": "long" if t.direction == 1 else "short",
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct, 2),
                "exit_reason": t.exit_reason.value if t.exit_reason else None,
                "duration_days": t.duration,
            }
        )
    return pd.DataFrame(rows)


def build_summary(result: BacktestResult) -> dict:
    """Single-dict summary of backtest performance."""
    exit_counts = {}
    for t in result.trades:
        reason = t.exit_reason.value if t.exit_reason else "unknown"
        exit_counts[reason] = exit_counts.get(reason, 0) + 1

    return {
        "total_trades": result.num_trades,
        "winning_trades": len(result.winning_trades),
        "losing_trades": len(result.losing_trades),
        "win_rate_pct": round(result.win_rate, 2),
        "total_pnl": round(result.total_pnl, 2),
        "total_pnl_pct": round(result.total_pnl_pct, 2),
        "avg_pnl_per_trade": round(result.avg_pnl, 2),
        "profit_factor": round(result.profit_factor, 2),
        "max_drawdown_pct": round(result.max_drawdown, 2),
        "avg_trade_duration_days": (
            round(result.avg_trade_duration, 1)
            if result.avg_trade_duration is not None
            else None
        ),
        "exit_reason_counts": exit_counts,
        "params": result.params,
    }


def format_report(result: BacktestResult) -> str:
    """Human-readable multi-line report string."""
    s = build_summary(result)
    lines = [
        "=" * 50,
        "  BACKTEST REPORT",
        "=" * 50,
        f"  Total trades      : {s['total_trades']}",
        f"  Winning / Losing  : {s['winning_trades']} / {s['losing_trades']}",
        f"  Win rate           : {s['win_rate_pct']:.1f}%",
        f"  Total PnL          : {s['total_pnl']:,.2f} ({s['total_pnl_pct']:.2f}%)",
        f"  Avg PnL / trade    : {s['avg_pnl_per_trade']:,.2f}",
        f"  Profit factor      : {s['profit_factor']:.2f}",
        f"  Max drawdown       : {s['max_drawdown_pct']:.2f}%",
    ]

    if s["avg_trade_duration_days"] is not None:
        lines.append(f"  Avg duration       : {s['avg_trade_duration_days']:.1f} days")

    lines.append("")
    lines.append("  Exit reasons:")
    for reason, count in s["exit_reason_counts"].items():
        lines.append(f"    {reason:20s}: {count}")

    if s["params"]:
        lines.append("")
        lines.append("  Parameters:")
        for k, v in s["params"].items():
            lines.append(f"    {k:20s}: {v}")

    lines.append("=" * 50)
    return "\n".join(lines)
