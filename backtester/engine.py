from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np


class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TAKE_PROFIT = "take_profit"
    TRAILING_TP = "trailing_tp"
    SIGNAL = "signal_exit"
    END_OF_DATA = "end_of_data"


@dataclass
class Trade:
    entry_date: object
    entry_price: float
    direction: int  # 1 = long, -1 = short
    exit_date: object = None
    exit_price: float = 0.0
    exit_reason: ExitReason = None
    highest_since_entry: float = 0.0
    tp_hit: bool = False

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.direction

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return self.pnl / self.entry_price * 100

    @property
    def duration(self) -> int | None:
        if self.exit_date is None or self.entry_date is None:
            return None
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    params: dict = field(default_factory=dict)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losing_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return len(self.winning_trades) / len(self.trades) * 100

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_pnl_pct(self) -> float:
        return sum(t.pnl_pct for t in self.trades)

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.winning_trades)
        gross_loss = abs(sum(t.pnl for t in self.losing_trades))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak * 100
        return drawdown.min()

    @property
    def avg_trade_duration(self) -> float | None:
        durations = [t.duration for t in self.trades if t.duration is not None]
        if not durations:
            return None
        return sum(durations) / len(durations)


class BacktestEngine:
    """Bar-by-bar backtesting engine with SL/TSL/TP/TTP support.

    Args:
        sl_pct: Fixed stop loss percentage (e.g. 2.0 for 2%).
        tsl_pct: Trailing stop loss percentage below highest price.
        tp_pct: Take profit percentage above entry price.
        ttp_pct: Once TP is hit, trail exit at this % below highest price
                 instead of exiting immediately.
        initial_capital: Starting capital.
    """

    def __init__(
        self,
        sl_pct: float = 2.0,
        tsl_pct: float = 1.5,
        tp_pct: float = 3.0,
        ttp_pct: float = 1.0,
        initial_capital: float = 100_000.0,
    ):
        self.sl_pct = sl_pct / 100
        self.tsl_pct = tsl_pct / 100
        self.tp_pct = tp_pct / 100
        self.ttp_pct = ttp_pct / 100
        self.initial_capital = initial_capital

    def run(self, signals_df: pd.DataFrame) -> BacktestResult:
        """Run backtest on a DataFrame that has a 'signal' column.

        signal: 1 = buy/long, -1 = sell/short, 0 = hold.

        Returns a BacktestResult with trades, equity curve, and stats.
        """
        trades: list[Trade] = []
        capital = self.initial_capital
        equity = []
        position: Trade | None = None

        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            date = signals_df.index[i]
            close = row["close"]
            high = row["high"]
            low = row["low"]
            signal = row["signal"]

            # --- Check exits for open position ---
            if position is not None:
                position.highest_since_entry = max(
                    position.highest_since_entry, high
                )
                exit_price, exit_reason = self._check_exit(
                    position, high, low, close, signal
                )

                if exit_reason is not None:
                    position.exit_date = date
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason
                    capital += position.pnl
                    trades.append(position)
                    position = None

            # --- Check entries ---
            if position is None and signal != 0:
                position = Trade(
                    entry_date=date,
                    entry_price=close,
                    direction=int(signal),
                    highest_since_entry=close,
                )

            equity.append(capital + (self._unrealized_pnl(position, close)))

        # Close any open position at end
        if position is not None:
            position.exit_date = signals_df.index[-1]
            position.exit_price = signals_df.iloc[-1]["close"]
            position.exit_reason = ExitReason.END_OF_DATA
            capital += position.pnl
            trades.append(position)

        equity_series = pd.Series(equity, index=signals_df.index, name="equity")

        return BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            params={
                "sl_pct": self.sl_pct * 100,
                "tsl_pct": self.tsl_pct * 100,
                "tp_pct": self.tp_pct * 100,
                "ttp_pct": self.ttp_pct * 100,
            },
        )

    def _check_exit(
        self, pos: Trade, high: float, low: float, close: float, signal: int
    ) -> tuple[float, ExitReason | None]:
        """Check all exit conditions. Returns (exit_price, reason) or (0, None)."""
        entry = pos.entry_price
        direction = pos.direction

        # --- Fixed stop loss ---
        sl_level = entry * (1 - self.sl_pct * direction)
        if direction == 1 and low <= sl_level:
            return sl_level, ExitReason.STOP_LOSS
        if direction == -1 and high >= sl_level:
            return sl_level, ExitReason.STOP_LOSS

        # --- Trailing stop loss ---
        if direction == 1:
            tsl_level = pos.highest_since_entry * (1 - self.tsl_pct)
            if low <= tsl_level and tsl_level > sl_level:
                return tsl_level, ExitReason.TRAILING_STOP
        else:
            # For shorts, trail above lowest (we track highest, invert logic)
            tsl_level = pos.highest_since_entry * (1 + self.tsl_pct)
            if high >= tsl_level:
                return tsl_level, ExitReason.TRAILING_STOP

        # --- Take profit / Trailing take profit ---
        tp_level = entry * (1 + self.tp_pct * direction)

        if not pos.tp_hit:
            if direction == 1 and high >= tp_level:
                if self.ttp_pct > 0:
                    pos.tp_hit = True  # Activate trailing TP mode
                else:
                    return tp_level, ExitReason.TAKE_PROFIT
            elif direction == -1 and low <= tp_level:
                if self.ttp_pct > 0:
                    pos.tp_hit = True
                else:
                    return tp_level, ExitReason.TAKE_PROFIT

        if pos.tp_hit:
            # Trail from highest since entry
            if direction == 1:
                ttp_level = pos.highest_since_entry * (1 - self.ttp_pct)
                if low <= ttp_level:
                    return ttp_level, ExitReason.TRAILING_TP
            else:
                ttp_level = pos.highest_since_entry * (1 + self.ttp_pct)
                if high >= ttp_level:
                    return ttp_level, ExitReason.TRAILING_TP

        # --- Signal reversal exit ---
        if signal != 0 and signal != direction:
            return close, ExitReason.SIGNAL

        return 0.0, None

    @staticmethod
    def _unrealized_pnl(position: Trade | None, price: float) -> float:
        if position is None:
            return 0.0
        return (price - position.entry_price) * position.direction
