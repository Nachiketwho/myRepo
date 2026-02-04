"""F&O backtesting engine — option-buying only, premium-based exits.

Workflow:
    1. Run spot strategy to get signals (buy=1, sell=-1)
    2. Buy signal → buy CE at selected strike
       Sell signal → buy PE at selected strike
    3. Price option via Black-Scholes (spot, strike, T, r, IV from VIX)
    4. Each bar: re-price option, check premium-based SL/TP
    5. Hold overnight is allowed — theta decays premium naturally
    6. Exit on: SL points / TP points / trailing SL / signal reversal /
       expiry eve
"""

from dataclasses import dataclass, field
from enum import Enum
import datetime as dt
import pandas as pd
import numpy as np

from backtester.greeks import bs_price, greeks_snapshot
from backtester.strikes import (
    get_atm_strike, get_strike_range, next_thursdays,
    time_to_expiry_years, days_to_expiry, describe_strike,
    LOT_SIZE, STRIKE_INTERVAL,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

FNO_DEFAULTS = {
    "lot_size": LOT_SIZE,
    "num_lots": 1,
    "sl_points": 50.0,
    "tp_points": 50.0,
    "tsl_points": 25.0,       # trail by 25pts after premium rises
    "tsl_activation": 30.0,   # activate trailing SL after 30pt rise
    "strike_interval": STRIKE_INTERVAL,
    "risk_free_rate": 0.07,
    "exit_before_expiry_days": 1,
    "num_otm_strikes": 4,     # ATM + 4 OTM = 5 strikes
    "num_expiries": 4,        # next 4 Thursdays
}


class FnOExitReason(str, Enum):
    SL_POINTS = "sl_points"
    TP_POINTS = "tp_points"
    TRAILING_SL = "trailing_sl"
    SIGNAL_REVERSAL = "signal_reversal"
    EXPIRY_EVE = "expiry_eve"
    END_OF_DATA = "end_of_data"


@dataclass
class FnOTrade:
    # Entry
    entry_date: object
    entry_spot: float
    strike: int
    option_type: str            # "CE" or "PE"
    expiry: dt.date
    premium_entry: float
    lot_size: int
    num_lots: int
    greeks_entry: dict          # {delta, gamma, theta, vega, iv}

    # Exit (filled on close)
    exit_date: object = None
    exit_spot: float = 0.0
    premium_exit: float = 0.0
    exit_reason: FnOExitReason = None
    greeks_exit: dict = field(default_factory=dict)

    # Tracking
    premium_high: float = 0.0   # highest premium seen during trade
    tsl_active: bool = False

    @property
    def pnl_per_lot(self) -> float:
        return self.premium_exit - self.premium_entry

    @property
    def pnl(self) -> float:
        return self.pnl_per_lot * self.lot_size * self.num_lots

    @property
    def pnl_pct(self) -> float:
        if self.premium_entry == 0:
            return 0.0
        return self.pnl_per_lot / self.premium_entry * 100

    @property
    def hold_days(self) -> int | None:
        if self.exit_date is None or self.entry_date is None:
            return None
        d1 = self.entry_date.date() if hasattr(self.entry_date, 'date') else self.entry_date
        d2 = self.exit_date.date() if hasattr(self.exit_date, 'date') else self.exit_date
        return (d2 - d1).days

    @property
    def strike_label(self) -> str:
        return describe_strike(self.entry_spot, self.strike, self.option_type)

    @property
    def contract_label(self) -> str:
        return f"{self.strike}{self.option_type} ({self.strike_label})"


@dataclass
class FnOResult:
    trades: list[FnOTrade]
    strike: int
    option_type_used: str       # "CE"/"PE" or "mixed"
    expiry_week: int            # 1=next Thu, 2=Thu after, etc.
    params: dict = field(default_factory=dict)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades) * 100

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)

    @property
    def avg_hold_days(self) -> float:
        durations = [t.hold_days for t in self.trades if t.hold_days is not None]
        if not durations:
            return 0.0
        return sum(durations) / len(durations)

    @property
    def exit_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.trades:
            r = t.exit_reason.value if t.exit_reason else "unknown"
            counts[r] = counts.get(r, 0) + 1
        return counts


class FnOEngine:
    """F&O backtesting engine — option buying only.

    Takes signals from the spot strategy and converts them to
    option trades with premium-based SL/TP.
    """

    def __init__(
        self,
        sl_points: float = 50.0,
        tp_points: float = 50.0,
        tsl_points: float = 25.0,
        tsl_activation: float = 30.0,
        lot_size: int = LOT_SIZE,
        num_lots: int = 1,
        risk_free_rate: float = 0.07,
        exit_before_expiry_days: int = 1,
    ):
        self.sl_points = sl_points
        self.tp_points = tp_points
        self.tsl_points = tsl_points
        self.tsl_activation = tsl_activation
        self.lot_size = lot_size
        self.num_lots = num_lots
        self.r = risk_free_rate
        self.exit_before_expiry_days = exit_before_expiry_days

    def run(
        self,
        signals_df: pd.DataFrame,
        vix_series: pd.Series,
        strike_offset: int = 0,
        expiry_week: int = 1,
    ) -> FnOResult:
        """Run F&O backtest on signal data.

        Args:
            signals_df: DataFrame with 'signal' column and OHLCV.
            vix_series: India VIX daily close indexed by date.
                        Values in % (e.g. 13.5 = 13.5%).
            strike_offset: 0=ATM, 1=1-OTM, 2=2-OTM, etc.
            expiry_week: 1=next Thursday, 2=Thursday after, etc.

        Returns:
            FnOResult with all trades for this strike/expiry combo.
        """
        trades: list[FnOTrade] = []
        position: FnOTrade | None = None

        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            date = signals_df.index[i]
            spot = row["close"]
            signal = row["signal"]

            # Get current date as date object
            current_date = date.date() if hasattr(date, 'date') else date

            # Look up IV from VIX (forward-fill for missing dates)
            iv = self._get_iv(vix_series, current_date)

            # --- Check exits for open position ---
            if position is not None:
                exit_price, exit_reason = self._check_exit(
                    position, spot, current_date, iv, signal
                )

                if exit_reason is not None:
                    T_exit = time_to_expiry_years(current_date, position.expiry)
                    position.exit_date = date
                    position.exit_spot = spot
                    position.premium_exit = exit_price
                    position.exit_reason = exit_reason
                    position.greeks_exit = greeks_snapshot(
                        spot, position.strike, T_exit, self.r,
                        iv, position.option_type,
                    )
                    trades.append(position)
                    position = None

            # --- Check entries ---
            if position is None and signal != 0:
                option_type = "CE" if signal == 1 else "PE"

                # Select strike
                atm = get_atm_strike(spot)
                if strike_offset == 0:
                    strike = atm
                else:
                    if option_type == "CE":
                        strike = atm + strike_offset * STRIKE_INTERVAL
                    else:
                        strike = atm - strike_offset * STRIKE_INTERVAL

                # Select expiry
                expiries = next_thursdays(current_date, count=expiry_week)
                expiry = expiries[-1]  # the Nth Thursday

                T = time_to_expiry_years(current_date, expiry)
                if T <= 0:
                    continue  # skip if expiry already passed

                premium = bs_price(spot, strike, T, self.r, iv, option_type)
                if premium <= 0:
                    continue  # skip degenerate pricing

                greeks = greeks_snapshot(
                    spot, strike, T, self.r, iv, option_type
                )

                position = FnOTrade(
                    entry_date=date,
                    entry_spot=spot,
                    strike=strike,
                    option_type=option_type,
                    expiry=expiry,
                    premium_entry=round(premium, 2),
                    lot_size=self.lot_size,
                    num_lots=self.num_lots,
                    greeks_entry=greeks,
                    premium_high=round(premium, 2),
                )

        # Close any open position at end of data
        if position is not None:
            last_date = signals_df.index[-1]
            last_spot = signals_df.iloc[-1]["close"]
            current_date = last_date.date() if hasattr(last_date, 'date') else last_date
            iv = self._get_iv(vix_series, current_date)
            T = time_to_expiry_years(current_date, position.expiry)
            premium = bs_price(last_spot, position.strike, T, self.r,
                               iv, position.option_type)

            position.exit_date = last_date
            position.exit_spot = last_spot
            position.premium_exit = round(premium, 2)
            position.exit_reason = FnOExitReason.END_OF_DATA
            position.greeks_exit = greeks_snapshot(
                last_spot, position.strike, T, self.r,
                iv, position.option_type,
            )
            trades.append(position)

        return FnOResult(
            trades=trades,
            strike=strike_offset,
            option_type_used="mixed",
            expiry_week=expiry_week,
            params={
                "sl_points": self.sl_points,
                "tp_points": self.tp_points,
                "tsl_points": self.tsl_points,
                "lot_size": self.lot_size,
                "num_lots": self.num_lots,
                "strike_offset": strike_offset,
                "expiry_week": expiry_week,
            },
        )

    def _check_exit(
        self,
        pos: FnOTrade,
        spot: float,
        current_date: dt.date,
        iv: float,
        signal: int,
    ) -> tuple[float, FnOExitReason | None]:
        """Check all exit conditions. Returns (exit_premium, reason)."""

        T = time_to_expiry_years(current_date, pos.expiry)
        current_premium = bs_price(
            spot, pos.strike, T, self.r, iv, pos.option_type
        )
        current_premium = round(current_premium, 2)

        # Track highest premium
        pos.premium_high = max(pos.premium_high, current_premium)

        premium_change = current_premium - pos.premium_entry

        # --- Expiry eve: exit day before expiry ---
        dte = days_to_expiry(current_date, pos.expiry)
        if dte <= self.exit_before_expiry_days:
            return current_premium, FnOExitReason.EXPIRY_EVE

        # --- SL: premium dropped by sl_points ---
        if premium_change <= -self.sl_points:
            return current_premium, FnOExitReason.SL_POINTS

        # --- TP: premium rose by tp_points ---
        if premium_change >= self.tp_points:
            return current_premium, FnOExitReason.TP_POINTS

        # --- Trailing SL ---
        if not pos.tsl_active and premium_change >= self.tsl_activation:
            pos.tsl_active = True

        if pos.tsl_active:
            trail_level = pos.premium_high - self.tsl_points
            if current_premium <= trail_level:
                return current_premium, FnOExitReason.TRAILING_SL

        # --- Signal reversal ---
        if signal != 0:
            # Buy signal while in PE, or sell signal while in CE
            if (pos.option_type == "CE" and signal == -1) or \
               (pos.option_type == "PE" and signal == 1):
                return current_premium, FnOExitReason.SIGNAL_REVERSAL

        return 0.0, None

    @staticmethod
    def _get_iv(vix_series: pd.Series, current_date: dt.date) -> float:
        """Look up IV from VIX series. Forward-fills missing dates.

        Returns IV as decimal (e.g. 0.13 for 13%).
        Falls back to 0.15 (15%) if VIX data unavailable.
        """
        DEFAULT_IV = 0.15

        if vix_series is None or vix_series.empty:
            return DEFAULT_IV

        # Try exact match
        if current_date in vix_series.index:
            val = vix_series[current_date]
            if pd.notna(val) and val > 0:
                return val / 100.0

        # Forward-fill: find most recent date <= current_date
        past = vix_series[vix_series.index <= current_date]
        if not past.empty:
            val = past.iloc[-1]
            if pd.notna(val) and val > 0:
                return val / 100.0

        return DEFAULT_IV


# ---------------------------------------------------------------------------
# Multi-strike × multi-expiry analysis
# ---------------------------------------------------------------------------

def run_fno_analysis(
    signals_df: pd.DataFrame,
    vix_series: pd.Series,
    num_otm: int = 4,
    num_expiries: int = 4,
    **engine_kwargs,
) -> pd.DataFrame:
    """Run F&O backtest across all strike × expiry combinations.

    Tests ATM + num_otm OTM strikes across num_expiries weekly expiries.
    Returns a summary DataFrame sorted by total PnL.
    """
    engine = FnOEngine(**engine_kwargs)
    rows: list[dict] = []

    for strike_offset in range(num_otm + 1):  # 0=ATM, 1..num_otm=OTM
        for exp_week in range(1, num_expiries + 1):
            result = engine.run(signals_df, vix_series,
                                strike_offset=strike_offset,
                                expiry_week=exp_week)

            if result.num_trades == 0:
                continue

            label = "ATM" if strike_offset == 0 else f"{strike_offset}-OTM"
            rows.append({
                "strike_type": label,
                "strike_offset": strike_offset,
                "expiry_week": exp_week,
                "num_trades": result.num_trades,
                "total_pnl": round(result.total_pnl, 2),
                "avg_pnl": round(result.avg_pnl, 2),
                "win_rate": round(result.win_rate, 1),
                "avg_hold_days": round(result.avg_hold_days, 1),
                **{f"exit_{k}": v for k, v in result.exit_reasons.items()},
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("total_pnl", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def build_fno_trade_log(result: FnOResult) -> pd.DataFrame:
    """Build a readable trade log from an FnOResult."""
    rows = []
    for t in result.trades:
        rows.append({
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "contract": t.contract_label,
            "expiry": t.expiry,
            "entry_spot": round(t.entry_spot, 2),
            "exit_spot": round(t.exit_spot, 2),
            "premium_in": t.premium_entry,
            "premium_out": t.premium_exit,
            "pnl_per_lot": round(t.pnl_per_lot, 2),
            "pnl_total": round(t.pnl, 2),
            "hold_days": t.hold_days,
            "exit_reason": t.exit_reason.value if t.exit_reason else "",
            "delta_in": t.greeks_entry.get("delta", ""),
            "theta_in": t.greeks_entry.get("theta", ""),
            "iv_in": t.greeks_entry.get("iv", ""),
            "delta_out": t.greeks_exit.get("delta", ""),
            "theta_out": t.greeks_exit.get("theta", ""),
        })
    return pd.DataFrame(rows)
