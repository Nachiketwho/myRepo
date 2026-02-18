"""EMA Crossover Strategy with multiple SL strategies and position sizing.

Supports:
- EMA pairs: 5/9, 9/15, 9/21, 13/21, 21/50 (customizable)
- SL strategies: ATR-based, swing high/low, percentage, fixed points
- Position sizing: fixed lots, risk-based, Kelly criterion
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------

class SLStrategy(Enum):
    ATR = "atr"
    SWING = "swing"
    PERCENTAGE = "percentage"
    FIXED = "fixed"


class PositionSizing(Enum):
    FIXED = "fixed"
    RISK_BASED = "risk_based"
    KELLY = "kelly"


# Default EMA pairs to test
EMA_PAIRS = [
    (5, 9),
    (9, 15),
    (9, 21),
    (13, 21),
    (21, 50),
]

# Default config
DEFAULT_CONFIG = {
    "atr_period": 14,
    "atr_multiplier": 2.0,
    "swing_lookback": 5,
    "sl_pct": 1.0,
    "sl_fixed_points": 50.0,
    "tp_rr_ratio": 2.0,
    "capital": 100000.0,
    "risk_per_trade_pct": 1.0,
    "lot_size": 25,  # Nifty lot size
}


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window=period).mean()


def swing_high(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Rolling swing high (highest high in lookback)."""
    return df["high"].rolling(window=lookback).max()


def swing_low(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Rolling swing low (lowest low in lookback)."""
    return df["low"].rolling(window=lookback).min()


# ---------------------------------------------------------------------------
# EMA Crossover Signal Generator
# ---------------------------------------------------------------------------

@dataclass
class EMACrossoverSignals:
    """Generate EMA crossover signals."""

    fast_period: int = 9
    slow_period: int = 21

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate crossover signals.

        Returns DataFrame with:
        - ema_fast, ema_slow: EMA values
        - signal: 1 (bullish cross), -1 (bearish cross), 0 (no signal)
        - crossover_type: 'golden_cross', 'death_cross', or ''
        """
        result = df.copy()

        result["ema_fast"] = ema(df["close"], self.fast_period)
        result["ema_slow"] = ema(df["close"], self.slow_period)

        # Crossover detection
        fast_above = result["ema_fast"] > result["ema_slow"]
        fast_above_prev = fast_above.shift(1).fillna(False)

        # Golden cross: fast crosses above slow
        golden = fast_above & (~fast_above_prev)
        # Death cross: fast crosses below slow
        death = (~fast_above) & fast_above_prev

        result["signal"] = 0
        result.loc[golden, "signal"] = 1
        result.loc[death, "signal"] = -1

        result["crossover_type"] = ""
        result.loc[golden, "crossover_type"] = "golden_cross"
        result.loc[death, "crossover_type"] = "death_cross"

        return result


# ---------------------------------------------------------------------------
# Stop Loss Calculator
# ---------------------------------------------------------------------------

@dataclass
class SLCalculator:
    """Calculate stop loss levels using different strategies."""

    strategy: SLStrategy = SLStrategy.ATR
    atr_period: int = 14
    atr_multiplier: float = 2.0
    swing_lookback: int = 5
    sl_pct: float = 1.0
    sl_fixed_points: float = 50.0

    def calculate(self, df: pd.DataFrame, signal: int, entry_idx: int) -> dict:
        """Calculate SL level for a trade.

        Args:
            df: OHLCV DataFrame with ATR/swing columns pre-computed
            signal: 1 for long, -1 for short
            entry_idx: Index position of entry bar

        Returns:
            dict with sl_level, sl_points, sl_strategy
        """
        entry_price = df.iloc[entry_idx]["close"]

        if self.strategy == SLStrategy.ATR:
            atr_val = df.iloc[entry_idx].get("atr", 50.0)
            sl_points = atr_val * self.atr_multiplier

        elif self.strategy == SLStrategy.SWING:
            if signal == 1:  # Long: SL below swing low
                sl_level = df.iloc[entry_idx].get("swing_low", entry_price * 0.99)
                sl_points = entry_price - sl_level
            else:  # Short: SL above swing high
                sl_level = df.iloc[entry_idx].get("swing_high", entry_price * 1.01)
                sl_points = sl_level - entry_price
            return {
                "sl_level": sl_level,
                "sl_points": max(sl_points, 1.0),
                "sl_strategy": self.strategy.value,
            }

        elif self.strategy == SLStrategy.PERCENTAGE:
            sl_points = entry_price * (self.sl_pct / 100)

        else:  # FIXED
            sl_points = self.sl_fixed_points

        # Calculate SL level from points
        if signal == 1:
            sl_level = entry_price - sl_points
        else:
            sl_level = entry_price + sl_points

        return {
            "sl_level": sl_level,
            "sl_points": sl_points,
            "sl_strategy": self.strategy.value,
        }


# ---------------------------------------------------------------------------
# Position Sizer
# ---------------------------------------------------------------------------

@dataclass
class PositionSizer:
    """Calculate position size using different methods."""

    method: PositionSizing = PositionSizing.RISK_BASED
    capital: float = 100000.0
    risk_per_trade_pct: float = 1.0
    fixed_lots: int = 1
    lot_size: int = 25
    win_rate: float = 0.5  # For Kelly
    avg_win_loss_ratio: float = 2.0  # For Kelly

    def calculate(self, entry_price: float, sl_points: float) -> dict:
        """Calculate position size.

        Returns:
            dict with num_lots, qty, risk_amount, position_value
        """
        if self.method == PositionSizing.FIXED:
            num_lots = self.fixed_lots

        elif self.method == PositionSizing.RISK_BASED:
            risk_amount = self.capital * (self.risk_per_trade_pct / 100)
            risk_per_lot = sl_points * self.lot_size
            num_lots = max(1, int(risk_amount / risk_per_lot)) if risk_per_lot > 0 else 1

        else:  # KELLY
            # Kelly fraction: f* = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - p
            b = self.avg_win_loss_ratio
            p = self.win_rate
            q = 1 - p
            kelly_fraction = (b * p - q) / b
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

            risk_amount = self.capital * kelly_fraction
            risk_per_lot = sl_points * self.lot_size
            num_lots = max(1, int(risk_amount / risk_per_lot)) if risk_per_lot > 0 else 1

        qty = num_lots * self.lot_size
        position_value = entry_price * qty
        risk_amount = sl_points * qty

        return {
            "num_lots": num_lots,
            "qty": qty,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "risk_pct": (risk_amount / self.capital) * 100 if self.capital > 0 else 0,
        }


# ---------------------------------------------------------------------------
# EMA Crossover Trade
# ---------------------------------------------------------------------------

@dataclass
class EMACrossoverTrade:
    """Single EMA crossover trade."""

    entry_date: pd.Timestamp
    entry_price: float
    direction: int  # 1 = long, -1 = short
    ema_fast: float
    ema_slow: float
    crossover_type: str

    # Risk management
    sl_level: float = 0.0
    sl_points: float = 0.0
    sl_strategy: str = ""
    tp_level: float = 0.0
    tp_points: float = 0.0
    rr_ratio: float = 2.0

    # Position sizing
    num_lots: int = 1
    qty: int = 25
    position_value: float = 0.0
    risk_amount: float = 0.0

    # Exit
    exit_date: Optional[pd.Timestamp] = None
    exit_price: float = 0.0
    exit_reason: str = ""

    # Results
    pnl_points: float = 0.0
    pnl_amount: float = 0.0
    pnl_pct: float = 0.0
    mae_points: float = 0.0  # Max adverse excursion
    mfe_points: float = 0.0  # Max favorable excursion
    bars_held: int = 0

    @property
    def is_winner(self) -> bool:
        return self.pnl_points > 0

    @property
    def r_multiple(self) -> float:
        """Return in terms of risk units (R)."""
        if self.sl_points > 0:
            return self.pnl_points / self.sl_points
        return 0.0


# ---------------------------------------------------------------------------
# EMA Crossover Backtest Result
# ---------------------------------------------------------------------------

@dataclass
class EMACrossoverResult:
    """Backtest result for one EMA pair + config combination."""

    ema_fast: int
    ema_slow: int
    sl_strategy: str
    position_sizing: str
    timeframe: str
    symbol: str

    trades: list = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> list:
        return [t for t in self.trades if t.is_winner]

    @property
    def losers(self) -> list:
        return [t for t in self.trades if not t.is_winner]

    @property
    def win_rate(self) -> float:
        if self.num_trades == 0:
            return 0.0
        return (len(self.winners) / self.num_trades) * 100

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_amount for t in self.trades)

    @property
    def total_pnl_points(self) -> float:
        return sum(t.pnl_points for t in self.trades)

    @property
    def avg_win(self) -> float:
        if not self.winners:
            return 0.0
        return np.mean([t.pnl_amount for t in self.winners])

    @property
    def avg_loss(self) -> float:
        if not self.losers:
            return 0.0
        return abs(np.mean([t.pnl_amount for t in self.losers]))

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_amount for t in self.winners)
        gross_loss = abs(sum(t.pnl_amount for t in self.losers))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def avg_r_multiple(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.r_multiple for t in self.trades])

    @property
    def expectancy(self) -> float:
        """Expected value per trade in R."""
        if self.num_trades == 0:
            return 0.0
        win_rate = len(self.winners) / self.num_trades
        avg_win_r = np.mean([t.r_multiple for t in self.winners]) if self.winners else 0
        avg_loss_r = abs(np.mean([t.r_multiple for t in self.losers])) if self.losers else 0
        return (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        equity = [0.0]
        for t in self.trades:
            equity.append(equity[-1] + t.pnl_amount)
        equity = np.array(equity)
        peaks = np.maximum.accumulate(equity)
        drawdowns = peaks - equity
        return float(np.max(drawdowns))

    @property
    def max_drawdown_pct(self) -> float:
        if not self.trades or "capital" not in self.params:
            return 0.0
        return (self.max_drawdown / self.params["capital"]) * 100

    @property
    def sharpe_ratio(self) -> float:
        """Simplified Sharpe (annualized)."""
        if len(self.trades) < 2:
            return 0.0
        returns = [t.pnl_pct for t in self.trades]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0.0
        # Assume ~250 trading days, ~26 trades per year for daily
        annualization = np.sqrt(250 / max(1, len(self.trades)))
        return (mean_ret / std_ret) * annualization

    @property
    def avg_bars_held(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.bars_held for t in self.trades])

    @property
    def calmar_ratio(self) -> float:
        """Total return / Max drawdown."""
        if self.max_drawdown == 0:
            return float("inf") if self.total_pnl > 0 else 0.0
        return self.total_pnl / self.max_drawdown

    def to_dict(self) -> dict:
        """Convert to dict for DataFrame."""
        return {
            "ema_pair": f"{self.ema_fast}/{self.ema_slow}",
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "sl_strategy": self.sl_strategy,
            "position_sizing": self.position_sizing,
            "timeframe": self.timeframe,
            "symbol": self.symbol,
            "num_trades": self.num_trades,
            "win_rate": round(self.win_rate, 1),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_points": round(self.total_pnl_points, 1),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "avg_r_multiple": round(self.avg_r_multiple, 2),
            "expectancy": round(self.expectancy, 3),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "avg_bars_held": round(self.avg_bars_held, 1),
        }


# ---------------------------------------------------------------------------
# EMA Crossover Engine
# ---------------------------------------------------------------------------

class EMACrossoverEngine:
    """Backtest engine for EMA crossover strategy."""

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        sl_strategy: SLStrategy = SLStrategy.ATR,
        position_sizing: PositionSizing = PositionSizing.RISK_BASED,
        tp_rr_ratio: float = 2.0,
        **config,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.sl_strategy = sl_strategy
        self.position_sizing = position_sizing
        self.tp_rr_ratio = tp_rr_ratio

        # Merge with defaults
        self.config = {**DEFAULT_CONFIG, **config}

        # Initialize components
        self.signal_gen = EMACrossoverSignals(ema_fast, ema_slow)
        self.sl_calc = SLCalculator(
            strategy=sl_strategy,
            atr_period=self.config["atr_period"],
            atr_multiplier=self.config["atr_multiplier"],
            swing_lookback=self.config["swing_lookback"],
            sl_pct=self.config["sl_pct"],
            sl_fixed_points=self.config["sl_fixed_points"],
        )
        self.pos_sizer = PositionSizer(
            method=position_sizing,
            capital=self.config["capital"],
            risk_per_trade_pct=self.config["risk_per_trade_pct"],
            lot_size=self.config["lot_size"],
        )

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators."""
        result = self.signal_gen.generate(df)
        result["atr"] = atr(df, self.config["atr_period"])
        result["swing_high"] = swing_high(df, self.config["swing_lookback"])
        result["swing_low"] = swing_low(df, self.config["swing_lookback"])
        return result

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "NIFTY",
        timeframe: str = "15min",
    ) -> EMACrossoverResult:
        """Run backtest on OHLCV data."""

        data = self._prepare_data(df)
        trades: list[EMACrossoverTrade] = []
        position: Optional[EMACrossoverTrade] = None

        for i in range(self.ema_slow + 1, len(data)):
            row = data.iloc[i]
            prev_row = data.iloc[i - 1]
            signal = int(row["signal"])

            # Check exit conditions if in position
            if position is not None:
                exit_reason = None
                exit_price = None

                # Check SL
                if position.direction == 1:  # Long
                    if row["low"] <= position.sl_level:
                        exit_reason = "sl_hit"
                        exit_price = position.sl_level
                    elif row["high"] >= position.tp_level:
                        exit_reason = "tp_hit"
                        exit_price = position.tp_level
                else:  # Short
                    if row["high"] >= position.sl_level:
                        exit_reason = "sl_hit"
                        exit_price = position.sl_level
                    elif row["low"] <= position.tp_level:
                        exit_reason = "tp_hit"
                        exit_price = position.tp_level

                # Signal reversal exit
                if exit_reason is None and signal != 0 and signal != position.direction:
                    exit_reason = "signal_reversal"
                    exit_price = row["close"]

                # Update MAE/MFE
                if position.direction == 1:
                    adverse = position.entry_price - row["low"]
                    favorable = row["high"] - position.entry_price
                else:
                    adverse = row["high"] - position.entry_price
                    favorable = position.entry_price - row["low"]
                position.mae_points = max(position.mae_points, adverse)
                position.mfe_points = max(position.mfe_points, favorable)
                position.bars_held += 1

                # Exit trade
                if exit_reason:
                    position.exit_date = row.name
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason

                    if position.direction == 1:
                        position.pnl_points = exit_price - position.entry_price
                    else:
                        position.pnl_points = position.entry_price - exit_price

                    position.pnl_amount = position.pnl_points * position.qty
                    position.pnl_pct = (position.pnl_points / position.entry_price) * 100

                    trades.append(position)
                    position = None

            # Open new position on signal
            if position is None and signal != 0:
                entry_price = row["close"]

                # Calculate SL
                sl_info = self.sl_calc.calculate(data, signal, i)

                # Calculate TP
                tp_points = sl_info["sl_points"] * self.tp_rr_ratio
                if signal == 1:
                    tp_level = entry_price + tp_points
                else:
                    tp_level = entry_price - tp_points

                # Calculate position size
                pos_info = self.pos_sizer.calculate(entry_price, sl_info["sl_points"])

                position = EMACrossoverTrade(
                    entry_date=row.name,
                    entry_price=entry_price,
                    direction=signal,
                    ema_fast=row["ema_fast"],
                    ema_slow=row["ema_slow"],
                    crossover_type=row["crossover_type"],
                    sl_level=sl_info["sl_level"],
                    sl_points=sl_info["sl_points"],
                    sl_strategy=sl_info["sl_strategy"],
                    tp_level=tp_level,
                    tp_points=tp_points,
                    rr_ratio=self.tp_rr_ratio,
                    num_lots=pos_info["num_lots"],
                    qty=pos_info["qty"],
                    position_value=pos_info["position_value"],
                    risk_amount=pos_info["risk_amount"],
                )

        # Close any open position at end
        if position is not None:
            last_row = data.iloc[-1]
            position.exit_date = last_row.name
            position.exit_price = last_row["close"]
            position.exit_reason = "end_of_data"

            if position.direction == 1:
                position.pnl_points = position.exit_price - position.entry_price
            else:
                position.pnl_points = position.entry_price - position.exit_price

            position.pnl_amount = position.pnl_points * position.qty
            position.pnl_pct = (position.pnl_points / position.entry_price) * 100
            trades.append(position)

        return EMACrossoverResult(
            ema_fast=self.ema_fast,
            ema_slow=self.ema_slow,
            sl_strategy=self.sl_strategy.value,
            position_sizing=self.position_sizing.value,
            timeframe=timeframe,
            symbol=symbol,
            trades=trades,
            params=self.config,
        )


# ---------------------------------------------------------------------------
# Batch Runner
# ---------------------------------------------------------------------------

def run_ema_backtest_grid(
    df: pd.DataFrame,
    symbol: str = "NIFTY",
    timeframe: str = "15min",
    ema_pairs: list[tuple[int, int]] = None,
    sl_strategies: list[SLStrategy] = None,
    tp_rr_ratio: float = 2.0,
    **config,
) -> pd.DataFrame:
    """Run grid of EMA crossover backtests.

    Args:
        df: OHLCV DataFrame
        symbol: Symbol name
        timeframe: Timeframe label
        ema_pairs: List of (fast, slow) EMA pairs to test
        sl_strategies: List of SL strategies to test
        tp_rr_ratio: Risk-reward ratio for TP
        **config: Additional config params

    Returns:
        DataFrame with results for each combination, sorted by total_pnl
    """
    if ema_pairs is None:
        ema_pairs = EMA_PAIRS
    if sl_strategies is None:
        sl_strategies = list(SLStrategy)

    results = []

    for fast, slow in ema_pairs:
        for sl_strat in sl_strategies:
            engine = EMACrossoverEngine(
                ema_fast=fast,
                ema_slow=slow,
                sl_strategy=sl_strat,
                tp_rr_ratio=tp_rr_ratio,
                **config,
            )
            result = engine.run(df, symbol, timeframe)
            results.append(result.to_dict())

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.sort_values("total_pnl", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)

    return results_df


def build_ema_trade_log(result: EMACrossoverResult) -> pd.DataFrame:
    """Build detailed trade log from backtest result."""
    if not result.trades:
        return pd.DataFrame()

    rows = []
    for t in result.trades:
        rows.append({
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "crossover": t.crossover_type,
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2),
            "sl_level": round(t.sl_level, 2),
            "tp_level": round(t.tp_level, 2),
            "sl_points": round(t.sl_points, 1),
            "sl_strategy": t.sl_strategy,
            "num_lots": t.num_lots,
            "qty": t.qty,
            "risk_amount": round(t.risk_amount, 2),
            "exit_reason": t.exit_reason,
            "pnl_points": round(t.pnl_points, 1),
            "pnl_amount": round(t.pnl_amount, 2),
            "pnl_pct": round(t.pnl_pct, 2),
            "r_multiple": round(t.r_multiple, 2),
            "mae_points": round(t.mae_points, 1),
            "mfe_points": round(t.mfe_points, 1),
            "bars_held": t.bars_held,
        })

    return pd.DataFrame(rows)


def compare_for_trading_style(results_df: pd.DataFrame) -> dict:
    """Analyze results and recommend best EMA pairs for different trading styles.

    Returns dict with recommendations for:
    - intraday: Best for 5min/15min scalping (low bars_held, high win_rate)
    - swing: Best for multi-day holds (high expectancy, good RR)
    """
    if results_df.empty:
        return {"intraday": None, "swing": None}

    # Intraday: prioritize win_rate and low bars_held
    intraday_score = (
        results_df["win_rate"] * 0.3 +
        results_df["profit_factor"].clip(upper=5) * 10 +
        (1 / results_df["avg_bars_held"].clip(lower=1)) * 20 +
        results_df["total_pnl"].clip(lower=0) / results_df["total_pnl"].abs().max() * 30
    )

    # Swing: prioritize expectancy and R-multiple
    swing_score = (
        results_df["expectancy"] * 50 +
        results_df["avg_r_multiple"] * 20 +
        results_df["calmar_ratio"].clip(upper=10) * 5 +
        results_df["total_pnl"].clip(lower=0) / results_df["total_pnl"].abs().max() * 25
    )

    intraday_idx = intraday_score.idxmax()
    swing_idx = swing_score.idxmax()

    return {
        "intraday": {
            "ema_pair": results_df.loc[intraday_idx, "ema_pair"],
            "sl_strategy": results_df.loc[intraday_idx, "sl_strategy"],
            "win_rate": results_df.loc[intraday_idx, "win_rate"],
            "profit_factor": results_df.loc[intraday_idx, "profit_factor"],
            "avg_bars_held": results_df.loc[intraday_idx, "avg_bars_held"],
            "total_pnl": results_df.loc[intraday_idx, "total_pnl"],
        },
        "swing": {
            "ema_pair": results_df.loc[swing_idx, "ema_pair"],
            "sl_strategy": results_df.loc[swing_idx, "sl_strategy"],
            "expectancy": results_df.loc[swing_idx, "expectancy"],
            "avg_r_multiple": results_df.loc[swing_idx, "avg_r_multiple"],
            "calmar_ratio": results_df.loc[swing_idx, "calmar_ratio"],
            "total_pnl": results_df.loc[swing_idx, "total_pnl"],
        },
    }
