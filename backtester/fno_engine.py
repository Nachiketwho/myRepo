"""F&O backtesting engine — option-buying only, risk-managed exits.

Risk management:
    - Adaptive SL: tighter SL for high-conviction signals, wider for weaker
    - RR-based TP: take-profit derived from SL × risk-reward ratio
    - Intra-bar SL: checks premium at bar worst-case (low for CE, high for PE)
    - Stepped trailing SL: milestones at 1:1.5, 1:2, 1:2.5 of SL
    - Trend filter: EMA-based trend validation, logged per trade
    - Transaction costs: brokerage + STT + GST included in break-even
    - Trade classification: good/neutral/bad with reasons

Workflow:
    1. Run spot strategy to get signals (buy=1, sell=-1)
    2. Buy signal → buy CE; Sell signal → buy PE
    3. Adaptive SL based on signal strength; TP from SL × RR ratio
    4. Each bar: reprice via BS, check intra-bar SL, then bar-close exits
    5. Stepped trailing locks profit at milestones
    6. Post-exit: decompose PnL (delta/theta/vega), classify trade quality
"""

from dataclasses import dataclass, field
from enum import Enum
import datetime as dt
import pandas as pd
import numpy as np

from backtester.greeks import bs_price, greeks_snapshot
from backtester.strikes import (
    get_atm_strike, get_strike_range, next_expiry_dates,
    time_to_expiry_years, days_to_expiry, describe_strike,
    LOT_SIZE, STRIKE_INTERVAL,
)
from backtester.costs import NSE_FNO_COSTS, TxnCosts


# ---------------------------------------------------------------------------
# Defaults & config
# ---------------------------------------------------------------------------

# Adaptive SL map: signal_strength → SL in premium points
ADAPTIVE_SL_MAP: dict[int, float] = {
    4: 20.0,
    3: 25.0,
    2: 30.0,
    1: 35.0,
    0: 40.0,
}

# Trailing SL milestones (as multiples of SL)
#   (premium_rise_multiple, profit_lock_multiple)
#   e.g. (1.5, 0) → at 1.5× SL rise, lock at break-even
#        (2.0, 1.0) → at 2× SL rise, lock 1× SL profit
TRAIL_MILESTONES = [
    (1.5, 0.0),   # 1:1.5 → move SL to break-even
    (2.0, 1.0),   # 1:2   → lock 1:1 profit
    (2.5, 1.5),   # 1:2.5 → lock 1:1.5 profit
]

FNO_DEFAULTS = {
    "lot_size": LOT_SIZE,
    "num_lots": 1,
    "base_sl": 30.0,
    "rr_ratio": 2.0,
    "ema_period": 21,
    "slippage": 1.0,
    "strike_interval": STRIKE_INTERVAL,
    "risk_free_rate": 0.07,
    "exit_before_expiry_days": 1,
    "num_otm_strikes": 4,
    "num_expiries": 4,
    "min_signal_strength": 2,
}


class FnOExitReason(str, Enum):
    SL_POINTS = "sl_points"
    TP_POINTS = "tp_points"
    TRAILING_SL = "trailing_sl"
    SIGNAL_REVERSAL = "signal_reversal"
    EXPIRY_EVE = "expiry_eve"
    END_OF_DATA = "end_of_data"


# ---------------------------------------------------------------------------
# FnOTrade dataclass
# ---------------------------------------------------------------------------

@dataclass
class FnOTrade:
    # ---- Entry ----
    entry_date: object
    entry_spot: float
    strike: int
    option_type: str            # "CE" or "PE"
    expiry: dt.date
    premium_entry: float
    lot_size: int
    num_lots: int
    greeks_entry: dict

    # ---- Risk levels (set at entry) ----
    sl_points: float = 0.0
    tp_points: float = 0.0
    rr_ratio: float = 1.0
    sl_level: float = 0.0           # premium price for SL
    tp_level: float = 0.0           # premium price for TP
    breakeven_level: float = 0.0    # premium to cover costs
    trail_milestones: list = field(default_factory=list)

    # ---- Exit ----
    exit_date: object = None
    exit_spot: float = 0.0
    premium_exit: float = 0.0
    exit_reason: FnOExitReason = None
    greeks_exit: dict = field(default_factory=dict)
    exit_pnl_points: float = 0.0

    # ---- Indicator snapshots ----
    indicators_entry: dict = field(default_factory=dict)
    indicators_exit: dict = field(default_factory=dict)

    # ---- Indicator flags (boolean breakdown) ----
    inner_band_touch: bool = False
    outer_band_touch: bool = False
    obv_confirm: bool = False
    ad_confirm: bool = False
    signal_strength: int = 0

    # ---- Trend ----
    ema_value: float = 0.0
    trend_aligned: bool = False
    trend_distance_pct: float = 0.0

    # ---- Tracking ----
    premium_high: float = 0.0
    tsl_current_level: float = 0.0
    active_milestone: int = 0

    # ---- Transaction costs ----
    txn_cost: float = 0.0
    txn_cost_points: float = 0.0

    # ---- PnL decomposition ----
    delta_pnl: float = 0.0
    theta_pnl: float = 0.0
    vega_pnl: float = 0.0

    # ---- Post-exit analysis ----
    post_exit_move: float = 0.0

    # ---- Trade quality ----
    trade_quality: str = ""
    quality_reasons: str = ""

    # ---- Computed properties ----
    @property
    def pnl_per_lot(self) -> float:
        return self.premium_exit - self.premium_entry

    @property
    def pnl_gross(self) -> float:
        return self.pnl_per_lot * self.lot_size * self.num_lots

    @property
    def pnl(self) -> float:
        """Net PnL after transaction costs."""
        return self.pnl_gross - self.txn_cost

    @property
    def pnl_pct(self) -> float:
        if self.premium_entry == 0:
            return 0.0
        return self.pnl_per_lot / self.premium_entry * 100

    @property
    def hold_days(self) -> int | None:
        if self.exit_date is None or self.entry_date is None:
            return None
        d1 = self.entry_date.date() if hasattr(self.entry_date, "date") else self.entry_date
        d2 = self.exit_date.date() if hasattr(self.exit_date, "date") else self.exit_date
        return (d2 - d1).days

    @property
    def strike_label(self) -> str:
        return describe_strike(self.entry_spot, self.strike, self.option_type)

    @property
    def contract_label(self) -> str:
        return f"{self.strike}{self.option_type} ({self.strike_label})"


# ---------------------------------------------------------------------------
# FnOResult
# ---------------------------------------------------------------------------

@dataclass
class FnOResult:
    trades: list[FnOTrade]
    strike: int
    option_type_used: str
    expiry_week: int
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


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

_INDICATOR_COLS = [
    "vwap", "vwap_upper_inner", "vwap_upper_outer",
    "vwap_lower_inner", "vwap_lower_outer", "obv", "ad_line",
]


def _capture_indicators(row: pd.Series) -> dict:
    """Extract indicator values from a signals_df row."""
    snap: dict = {}
    for col in _INDICATOR_COLS:
        if col in row.index:
            val = row[col]
            snap[col] = round(float(val), 2) if pd.notna(val) else None
    if "vwap" in snap and snap["vwap"] is not None:
        close = float(row["close"])
        snap["close_vs_vwap"] = round(close - snap["vwap"], 2)
        snap["close_vs_vwap_pct"] = round(
            (close - snap["vwap"]) / snap["vwap"] * 100, 3
        ) if snap["vwap"] != 0 else 0.0
    return snap


def _indicator_flags(row: pd.Series, signals_df: pd.DataFrame, idx: int,
                     obv_lookback: int = 3, ad_lookback: int = 3) -> dict:
    """Return boolean flags for each indicator confirmation.

    Returns dict with keys: inner_band, outer_band, obv, ad_line, strength
    """
    signal = row["signal"]
    flags = {
        "inner_band": False,
        "outer_band": False,
        "obv": False,
        "ad_line": False,
    }
    if signal == 0:
        return {**flags, "strength": 0}

    # Band touches
    if signal == 1:  # buy
        if "vwap_lower_inner" in row.index and pd.notna(row.get("vwap_lower_inner")):
            if row["low"] <= row["vwap_lower_inner"]:
                flags["inner_band"] = True
        if "vwap_lower_outer" in row.index and pd.notna(row.get("vwap_lower_outer")):
            if row["low"] <= row["vwap_lower_outer"]:
                flags["outer_band"] = True
    else:  # sell
        if "vwap_upper_inner" in row.index and pd.notna(row.get("vwap_upper_inner")):
            if row["high"] >= row["vwap_upper_inner"]:
                flags["inner_band"] = True
        if "vwap_upper_outer" in row.index and pd.notna(row.get("vwap_upper_outer")):
            if row["high"] >= row["vwap_upper_outer"]:
                flags["outer_band"] = True

    # Volume confirmations
    if "obv" in row.index and idx >= obv_lookback:
        obv_now = row["obv"]
        obv_prev = signals_df.iloc[idx - obv_lookback]["obv"]
        if (signal == 1 and obv_now > obv_prev) or \
           (signal == -1 and obv_now < obv_prev):
            flags["obv"] = True

    if "ad_line" in row.index and idx >= ad_lookback:
        ad_now = row["ad_line"]
        ad_prev = signals_df.iloc[idx - ad_lookback]["ad_line"]
        if (signal == 1 and ad_now > ad_prev) or \
           (signal == -1 and ad_now < ad_prev):
            flags["ad_line"] = True

    strength = sum(flags.values())
    return {**flags, "strength": strength}


# ---------------------------------------------------------------------------
# FnOEngine
# ---------------------------------------------------------------------------

class FnOEngine:
    """F&O backtesting engine with risk management.

    Risk model:
        - Adaptive SL: signal strength 4→20pts, 3→25pts, 2→30pts, 1→35pts
        - TP = SL × rr_ratio (e.g. SL=30, RR=2 → TP=60)
        - Stepped trailing at 1:1.5 (break-even), 1:2 (lock 1×SL), 1:2.5 (lock 1.5×SL)
        - Intra-bar SL check at bar worst-case (low for CE, high for PE)
        - Transaction costs deducted from PnL; break-even computed per trade
    """

    def __init__(
        self,
        base_sl: float = 30.0,
        rr_ratio: float = 2.0,
        ema_period: int = 21,
        slippage: float = 1.0,
        lot_size: int = LOT_SIZE,
        num_lots: int = 1,
        risk_free_rate: float = 0.07,
        exit_before_expiry_days: int = 1,
        min_signal_strength: int = 2,
        min_premium: float = 0.0,
        min_delta: float = 0.0,
        sl_map: dict | None = None,
        costs: TxnCosts | None = None,
        # Backward compat: if sl_points/tp_points passed, use fixed SL/TP
        sl_points: float | None = None,
        tp_points: float | None = None,
    ):
        self.base_sl = base_sl
        self.rr_ratio = rr_ratio
        self.ema_period = ema_period
        self.slippage = slippage
        self.lot_size = lot_size
        self.num_lots = num_lots
        self.r = risk_free_rate
        self.exit_before_expiry_days = exit_before_expiry_days
        self.min_signal_strength = min_signal_strength
        self.min_premium = min_premium
        self.min_delta = min_delta
        self.sl_map = sl_map or ADAPTIVE_SL_MAP
        self.costs = costs or NSE_FNO_COSTS

        # Backward compat: fixed SL/TP override
        self._fixed_sl = sl_points
        self._fixed_tp = tp_points

    def _get_sl(self, strength: int) -> float:
        """Return SL points for a given signal strength."""
        if self._fixed_sl is not None:
            return self._fixed_sl
        return self.sl_map.get(strength, self.base_sl)

    def _get_tp(self, sl: float) -> float:
        """Return TP points derived from SL × RR ratio."""
        if self._fixed_tp is not None:
            return self._fixed_tp
        return sl * self.rr_ratio

    def _build_trail_milestones(
        self, entry_premium: float, sl: float, be_points: float,
    ) -> list[tuple[float, float]]:
        """Build stepped trailing SL milestones for a trade.

        Returns list of (premium_threshold, new_sl_floor) tuples.
        """
        milestones = []
        for rise_mult, lock_mult in TRAIL_MILESTONES:
            threshold = entry_premium + sl * rise_mult
            if lock_mult == 0.0:
                # Lock at break-even (entry + costs)
                floor = entry_premium + be_points
            else:
                floor = entry_premium + sl * lock_mult
            milestones.append((round(threshold, 2), round(floor, 2)))
        return milestones

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
            vix_series: India VIX daily close indexed by date (in %).
            strike_offset: 0=ATM, 1=1-OTM, 2=2-OTM, etc.
            expiry_week: 1=next Tuesday, 2=Tuesday after, etc.
        """
        trades: list[FnOTrade] = []
        position: FnOTrade | None = None

        # Compute EMA for trend validation
        ema = signals_df["close"].ewm(span=self.ema_period, adjust=False).mean()

        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            date = signals_df.index[i]
            spot = row["close"]
            signal = row["signal"]
            current_date = date.date() if hasattr(date, "date") else date
            iv = self._get_iv(vix_series, current_date)

            # --- Check exits for open position ---
            if position is not None:
                exit_price, exit_reason = self._check_exit(
                    position, row, current_date, iv, signal,
                )

                if exit_reason is not None:
                    T_exit = time_to_expiry_years(current_date, position.expiry)
                    position.exit_date = date
                    position.exit_spot = spot
                    position.premium_exit = exit_price
                    position.exit_reason = exit_reason
                    position.exit_pnl_points = round(exit_price - position.premium_entry, 2)
                    position.greeks_exit = greeks_snapshot(
                        spot, position.strike, T_exit, self.r,
                        iv, position.option_type,
                    )
                    position.indicators_exit = _capture_indicators(row)

                    # Transaction costs
                    cost_info = self.costs.round_trip(
                        position.premium_entry, exit_price,
                        position.lot_size, position.num_lots,
                    )
                    position.txn_cost = cost_info["total"]

                    # PnL decomposition
                    position.delta_pnl, position.theta_pnl, position.vega_pnl = \
                        self._decompose_pnl(position, iv)

                    # Post-exit analysis (look ahead 5 bars)
                    position.post_exit_move = self._post_exit_move(
                        signals_df, i, position,
                    )

                    # Classify trade
                    position.trade_quality, position.quality_reasons = \
                        self._classify_trade(position)

                    trades.append(position)
                    position = None

            # --- Check entries ---
            if position is None and signal != 0:
                flags = _indicator_flags(row, signals_df, i)
                strength = flags["strength"]

                if strength < self.min_signal_strength:
                    continue

                option_type = "CE" if signal == 1 else "PE"
                atm = get_atm_strike(spot)
                if strike_offset == 0:
                    strike = atm
                elif option_type == "CE":
                    strike = atm + strike_offset * STRIKE_INTERVAL
                else:
                    strike = atm - strike_offset * STRIKE_INTERVAL

                expiries = next_expiry_dates(current_date, count=expiry_week)
                expiry = expiries[-1]

                T = time_to_expiry_years(current_date, expiry)
                if T <= 0:
                    continue

                # Skip entries too close to expiry (< 2 days)
                dte = days_to_expiry(current_date, expiry)
                if dte < 2:
                    continue

                premium = bs_price(spot, strike, T, self.r, iv, option_type)
                if premium <= 0:
                    continue

                if premium < self.min_premium:
                    continue

                greeks = greeks_snapshot(spot, strike, T, self.r, iv, option_type)
                if abs(greeks.get("delta", 0)) < self.min_delta:
                    continue

                # Adaptive SL and RR-based TP
                sl = self._get_sl(strength)
                tp = self._get_tp(sl)

                # Break-even
                be_points = self.costs.breakeven_points(
                    premium, self.lot_size, self.num_lots,
                )

                # Trail milestones
                trail_ms = self._build_trail_milestones(premium, sl, be_points)

                # Trend
                ema_val = float(ema.iloc[i])
                if signal == 1:
                    trend_aligned = spot > ema_val
                else:
                    trend_aligned = spot < ema_val
                trend_dist = round((spot - ema_val) / ema_val * 100, 3) if ema_val != 0 else 0.0

                ind_entry = _capture_indicators(row)

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
                    # Risk levels
                    sl_points=sl,
                    tp_points=tp,
                    rr_ratio=self.rr_ratio if self._fixed_tp is None else (
                        tp / sl if sl > 0 else 0
                    ),
                    sl_level=round(premium - sl, 2),
                    tp_level=round(premium + tp, 2),
                    breakeven_level=round(premium + be_points, 2),
                    trail_milestones=trail_ms,
                    # Txn cost estimate (per point)
                    txn_cost_points=be_points,
                    # Indicator flags
                    inner_band_touch=flags["inner_band"],
                    outer_band_touch=flags["outer_band"],
                    obv_confirm=flags["obv"],
                    ad_confirm=flags["ad_line"],
                    signal_strength=strength,
                    # Trend
                    ema_value=round(ema_val, 2),
                    trend_aligned=trend_aligned,
                    trend_distance_pct=trend_dist,
                    # Tracking
                    indicators_entry=ind_entry,
                    premium_high=round(premium, 2),
                )

        # Close any open position at end of data
        if position is not None:
            last_row = signals_df.iloc[-1]
            last_date = signals_df.index[-1]
            last_spot = last_row["close"]
            current_date = last_date.date() if hasattr(last_date, "date") else last_date
            iv = self._get_iv(vix_series, current_date)
            T = time_to_expiry_years(current_date, position.expiry)
            premium = bs_price(last_spot, position.strike, T, self.r,
                               iv, position.option_type)
            premium = round(premium, 2)

            position.exit_date = last_date
            position.exit_spot = last_spot
            position.premium_exit = premium
            position.exit_reason = FnOExitReason.END_OF_DATA
            position.exit_pnl_points = round(premium - position.premium_entry, 2)
            position.greeks_exit = greeks_snapshot(
                last_spot, position.strike, T, self.r,
                iv, position.option_type,
            )
            position.indicators_exit = _capture_indicators(last_row)
            cost_info = self.costs.round_trip(
                position.premium_entry, premium,
                position.lot_size, position.num_lots,
            )
            position.txn_cost = cost_info["total"]
            position.delta_pnl, position.theta_pnl, position.vega_pnl = \
                self._decompose_pnl(position, iv)
            position.post_exit_move = self._post_exit_move(
                signals_df, len(signals_df) - 1, position,
            )
            position.trade_quality, position.quality_reasons = \
                self._classify_trade(position)
            trades.append(position)

        return FnOResult(
            trades=trades,
            strike=strike_offset,
            option_type_used="mixed",
            expiry_week=expiry_week,
            params={
                "base_sl": self.base_sl,
                "rr_ratio": self.rr_ratio,
                "ema_period": self.ema_period,
                "lot_size": self.lot_size,
                "num_lots": self.num_lots,
                "strike_offset": strike_offset,
                "expiry_week": expiry_week,
                # Backward compat
                "sl_points": self._fixed_sl or self.base_sl,
                "tp_points": self._fixed_tp or (self.base_sl * self.rr_ratio),
            },
        )

    def _check_exit(
        self,
        pos: FnOTrade,
        row: pd.Series,
        current_date: dt.date,
        iv: float,
        signal: int,
    ) -> tuple[float, FnOExitReason | None]:
        """Check all exit conditions with proper risk management.

        Order: expiry eve → intra-bar SL → bar-close SL → TP →
               trailing milestone update → trailing SL → signal reversal.
        """
        spot = row["close"]
        T = time_to_expiry_years(current_date, pos.expiry)

        # Premium at bar close
        close_premium = bs_price(spot, pos.strike, T, self.r, iv, pos.option_type)
        close_premium = round(close_premium, 2)

        # --- Expiry eve ---
        dte = days_to_expiry(current_date, pos.expiry)
        if dte <= self.exit_before_expiry_days:
            return close_premium, FnOExitReason.EXPIRY_EVE

        # --- Intra-bar SL check (worst case within bar) ---
        if pos.option_type == "CE":
            worst_spot = row["low"]
        else:
            worst_spot = row["high"]

        worst_premium = bs_price(worst_spot, pos.strike, T, self.r, iv, pos.option_type)
        worst_premium = round(worst_premium, 2)

        if worst_premium <= pos.sl_level:
            # SL breached intra-bar — cap exit at SL price + slippage
            sl_exit = max(pos.sl_level - self.slippage, 0.0)
            return round(sl_exit, 2), FnOExitReason.SL_POINTS

        # --- Bar-close SL (if not caught intra-bar) ---
        if close_premium <= pos.sl_level:
            sl_exit = max(pos.sl_level - self.slippage, 0.0)
            return round(sl_exit, 2), FnOExitReason.SL_POINTS

        # --- TP check ---
        # Also check intra-bar best case for TP
        if pos.option_type == "CE":
            best_spot = row["high"]
        else:
            best_spot = row["low"]
        best_premium = bs_price(best_spot, pos.strike, T, self.r, iv, pos.option_type)

        if best_premium >= pos.tp_level or close_premium >= pos.tp_level:
            tp_exit = pos.tp_level - self.slippage
            return round(tp_exit, 2), FnOExitReason.TP_POINTS

        # Update premium high
        pos.premium_high = max(pos.premium_high, close_premium, best_premium)

        # --- Trailing SL milestone updates ---
        for threshold, new_floor in pos.trail_milestones:
            if pos.premium_high >= threshold and pos.tsl_current_level < new_floor:
                pos.tsl_current_level = new_floor
                pos.active_milestone += 1

        # --- Trailing SL exit ---
        if pos.tsl_current_level > 0 and close_premium <= pos.tsl_current_level:
            return round(pos.tsl_current_level - self.slippage, 2), FnOExitReason.TRAILING_SL

        # --- Signal reversal ---
        if signal != 0:
            if (pos.option_type == "CE" and signal == -1) or \
               (pos.option_type == "PE" and signal == 1):
                return close_premium, FnOExitReason.SIGNAL_REVERSAL

        return 0.0, None

    @staticmethod
    def _decompose_pnl(trade: FnOTrade, iv_exit: float) -> tuple[float, float, float]:
        """Approximate PnL decomposition into delta, theta, vega components.

        Uses Greeks at entry as a first-order approximation.
        """
        delta = trade.greeks_entry.get("delta", 0)
        theta = trade.greeks_entry.get("theta", 0)
        vega = trade.greeks_entry.get("vega", 0)
        iv_in = trade.greeks_entry.get("iv", 0)

        spot_change = trade.exit_spot - trade.entry_spot
        hold = trade.hold_days or 0
        iv_change = (iv_exit - iv_in / 100) * 100 if iv_in > 1 else (iv_exit - iv_in) * 100

        delta_pnl = round(delta * spot_change, 2)
        theta_pnl = round(theta * hold, 2)
        vega_pnl = round(vega * iv_change, 2)

        return delta_pnl, theta_pnl, vega_pnl

    @staticmethod
    def _post_exit_move(
        signals_df: pd.DataFrame, exit_idx: int, trade: FnOTrade,
        look_ahead: int = 5,
    ) -> float:
        """Spot movement after exit (next N bars).

        Positive = move was in the direction opposite to the original trade
        (i.e. reversal was correct). For analysis only.
        """
        n = len(signals_df)
        bars = min(look_ahead, n - exit_idx - 1)
        if bars <= 0:
            return 0.0

        future_closes = [
            signals_df.iloc[exit_idx + j]["close"] for j in range(1, bars + 1)
        ]
        exit_spot = trade.exit_spot

        if trade.option_type == "CE":
            # CE trade — if spot dropped after exit, reversal was correct
            move = exit_spot - min(future_closes)
        else:
            # PE trade — if spot rose after exit, reversal was correct
            move = max(future_closes) - exit_spot

        return round(move, 2)

    @staticmethod
    def _classify_trade(trade: FnOTrade) -> tuple[str, str]:
        """Classify trade as good/neutral/bad with reasons."""
        reasons = []

        # Exit outcome
        if trade.exit_reason in (FnOExitReason.TP_POINTS, FnOExitReason.TRAILING_SL):
            reasons.append("profit_taken")
        elif trade.exit_reason == FnOExitReason.SL_POINTS:
            reasons.append("sl_hit")
        elif trade.exit_reason == FnOExitReason.EXPIRY_EVE:
            reasons.append("expiry_forced")
        elif trade.exit_reason == FnOExitReason.END_OF_DATA:
            reasons.append("data_ended")

        # Signal strength
        if trade.signal_strength >= 3:
            reasons.append("strong_signal")
        elif trade.signal_strength <= 1:
            reasons.append("weak_signal")

        # Trend
        if trade.trend_aligned:
            reasons.append("trend_aligned")
        else:
            reasons.append("counter_trend")

        # Indicator gaps
        if not trade.obv_confirm:
            reasons.append("no_obv")
        if not trade.ad_confirm:
            reasons.append("no_ad")

        # Reversal analysis
        if trade.exit_reason == FnOExitReason.SIGNAL_REVERSAL:
            if trade.post_exit_move > 0:
                reasons.append("reversal_correct")
            elif trade.post_exit_move < 0:
                reasons.append("reversal_wrong")
            else:
                reasons.append("reversal_flat")

        # Score
        good_flags = {"profit_taken", "strong_signal", "trend_aligned", "reversal_correct"}
        bad_flags = {"sl_hit", "weak_signal", "counter_trend", "no_obv", "no_ad", "reversal_wrong"}

        good_count = sum(1 for r in reasons if r in good_flags)
        bad_count = sum(1 for r in reasons if r in bad_flags)

        if good_count > bad_count:
            quality = "good"
        elif bad_count > good_count:
            quality = "bad"
        else:
            quality = "neutral"

        return quality, ", ".join(reasons)

    @staticmethod
    def _get_iv(vix_series: pd.Series, current_date: dt.date) -> float:
        """Look up IV from VIX series. Forward-fills missing dates.

        Returns IV as decimal (e.g. 0.13 for 13%).
        Falls back to 0.15 (15%) if VIX data unavailable.
        """
        DEFAULT_IV = 0.15

        if vix_series is None or vix_series.empty:
            return DEFAULT_IV

        if current_date in vix_series.index:
            val = vix_series[current_date]
            if pd.notna(val) and val > 0:
                return val / 100.0

        past = vix_series[vix_series.index <= current_date]
        if not past.empty:
            val = past.iloc[-1]
            if pd.notna(val) and val > 0:
                return val / 100.0

        return DEFAULT_IV


# ---------------------------------------------------------------------------
# Multi-strike x multi-expiry analysis
# ---------------------------------------------------------------------------

def run_fno_analysis(
    signals_df: pd.DataFrame,
    vix_series: pd.Series,
    num_otm: int = 4,
    num_expiries: int = 4,
    **engine_kwargs,
) -> pd.DataFrame:
    """Run F&O backtest across all strike x expiry combinations."""
    engine = FnOEngine(**engine_kwargs)
    rows: list[dict] = []

    for strike_offset in range(num_otm + 1):
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


# ---------------------------------------------------------------------------
# Trade log builder
# ---------------------------------------------------------------------------

def build_fno_trade_log(result: FnOResult) -> pd.DataFrame:
    """Build a detailed trade log with risk levels, indicators, and quality."""
    rows = []
    for t in result.trades:
        row = {
            # Trade basics
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "contract": t.contract_label,
            "expiry": t.expiry,
            "entry_spot": round(t.entry_spot, 2),
            "exit_spot": round(t.exit_spot, 2),
            # Premium & PnL
            "premium_in": t.premium_entry,
            "premium_out": t.premium_exit,
            "premium_high": round(t.premium_high, 2),
            "pnl_per_lot": round(t.pnl_per_lot, 2),
            "pnl_gross": round(t.pnl_gross, 2),
            "txn_cost": round(t.txn_cost, 2),
            "pnl_net": round(t.pnl, 2),
            "pnl_pct": round(t.pnl_pct, 1),
            "hold_days": t.hold_days,
            # Risk levels
            "sl_points": t.sl_points,
            "tp_points": t.tp_points,
            "rr_ratio": t.rr_ratio,
            "sl_level": t.sl_level,
            "tp_level": t.tp_level,
            "breakeven_level": t.breakeven_level,
            "tsl_level_at_exit": round(t.tsl_current_level, 2),
            "trail_milestone_reached": t.active_milestone,
            # Exit
            "exit_reason": t.exit_reason.value if t.exit_reason else "",
            "exit_pnl_points": t.exit_pnl_points,
            # Signal & indicators
            "signal_strength": t.signal_strength,
            "inner_band": t.inner_band_touch,
            "outer_band": t.outer_band_touch,
            "obv_confirm": t.obv_confirm,
            "ad_confirm": t.ad_confirm,
            # Trend
            "ema_value": t.ema_value,
            "trend_aligned": t.trend_aligned,
            "trend_distance_pct": t.trend_distance_pct,
            # Greeks at entry
            "delta_in": round(t.greeks_entry.get("delta", 0), 4),
            "gamma_in": round(t.greeks_entry.get("gamma", 0), 6),
            "theta_in": round(t.greeks_entry.get("theta", 0), 2),
            "vega_in": round(t.greeks_entry.get("vega", 0), 2),
            "iv_in": round(t.greeks_entry.get("iv", 0), 1),
            # Greeks at exit
            "delta_out": round(t.greeks_exit.get("delta", 0), 4),
            "theta_out": round(t.greeks_exit.get("theta", 0), 2),
            "iv_out": round(t.greeks_exit.get("iv", 0), 1),
            # PnL decomposition
            "delta_pnl": t.delta_pnl,
            "theta_pnl": t.theta_pnl,
            "vega_pnl": t.vega_pnl,
            # Post-exit
            "post_exit_move": t.post_exit_move,
            # Indicators at entry
            "vwap_in": t.indicators_entry.get("vwap", ""),
            "close_vs_vwap_in": t.indicators_entry.get("close_vs_vwap", ""),
            "obv_in": t.indicators_entry.get("obv", ""),
            "ad_in": t.indicators_entry.get("ad_line", ""),
            # Indicators at exit
            "vwap_out": t.indicators_exit.get("vwap", ""),
            "close_vs_vwap_out": t.indicators_exit.get("close_vs_vwap", ""),
            "obv_out": t.indicators_exit.get("obv", ""),
            "ad_out": t.indicators_exit.get("ad_line", ""),
            # Quality
            "trade_quality": t.trade_quality,
            "quality_reasons": t.quality_reasons,
        }
        rows.append(row)
    return pd.DataFrame(rows)
