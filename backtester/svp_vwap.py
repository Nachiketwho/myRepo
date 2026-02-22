"""Session Volume Profile + VWAP Combined Strategy.

The strategy combines:
1. Session Volume Profile (SVP): Identifies price levels with highest traded
   volume within a session. Key levels: POC (Point of Control), VAH (Value
   Area High), VAL (Value Area Low).
2. VWAP with bands: Anchored VWAP with standard-deviation bands as dynamic
   support/resistance.

Trade signals:
- LONG:  Price bounces off VAL/VWAP lower band with volume confirmation,
         or breaks above POC with momentum.
- SHORT: Price rejects from VAH/VWAP upper band with volume confirmation,
         or breaks below POC with momentum.

Supports:
- SL strategies: ATR, Swing, Percentage, Fixed
- Trailing SL: NONE, BREAKEVEN, STEPPED
- Position sizing: Fixed, Risk-based, Kelly
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from backtester.ema_crossover import (
    SLStrategy,
    PositionSizing,
    TrailingMode,
    TRAIL_MILESTONES,
    SLCalculator,
    PositionSizer,
    atr,
    swing_high,
    swing_low,
)


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class SVPSignalType(Enum):
    """Signal types from SVP+VWAP strategy."""
    # Basic signals (V1)
    VAL_BOUNCE = "val_bounce"         # Price bounces off Value Area Low
    VAH_REJECT = "vah_reject"         # Price rejects from Value Area High
    POC_BREAK_UP = "poc_break_up"     # Bullish breakout above POC
    POC_BREAK_DOWN = "poc_break_down" # Bearish breakdown below POC
    VWAP_BOUNCE = "vwap_bounce"       # Bounce off VWAP lower band
    VWAP_REJECT = "vwap_reject"       # Rejection at VWAP upper band
    # Advanced signals (V2)
    VAH_ACCEPTANCE = "vah_acceptance" # Sustained trading above VAH (bullish)
    VAL_ACCEPTANCE = "val_acceptance" # Sustained trading below VAL (bearish)
    POC_MIGRATE_UP = "poc_migrate_up" # POC shifting upward (bullish)
    POC_MIGRATE_DN = "poc_migrate_dn" # POC shifting downward (bearish)
    LVN_BREAKOUT = "lvn_breakout"     # Low Volume Node breakout (explosion)


class TradeInstrument(Enum):
    """Recommended trade instrument based on signal."""
    CALL = "call"     # Buy call options
    PUT = "put"       # Buy put options
    FUTURES = "futures"  # Trade futures (neutral)


# Value area covers 70% of session volume (standard TPO/VP definition)
VALUE_AREA_PCT = 0.70

# Number of price bins for volume profile
DEFAULT_NUM_BINS = 50

# LVN threshold: bins with volume below this fraction of avg are LVNs
LVN_THRESHOLD = 0.3

# Default config
SVP_DEFAULT_CONFIG = {
    "atr_period": 14,
    "atr_multiplier": 2.0,
    "swing_lookback": 5,
    "sl_pct": 1.0,
    "sl_fixed_points": 50.0,
    "tp_rr_ratio": 2.0,
    "capital": 100000.0,
    "risk_per_trade_pct": 1.0,
    "lot_size": 25,
    "trailing_mode": TrailingMode.NONE,
    "trail_milestones": TRAIL_MILESTONES,
    # SVP-specific
    "num_bins": DEFAULT_NUM_BINS,
    "value_area_pct": VALUE_AREA_PCT,
    "session_lookback": 75,  # bars per session (5min: 75, 15min: 25)
    "vwap_band_inner": 1.0,
    "vwap_band_outer": 2.0,
    "volume_confirm_mult": 1.2,  # volume must be 1.2x avg for confirmation
    "poc_break_bars": 3,  # bars above/below POC to confirm breakout
    # Advanced V2 features
    "acceptance_bars": 3,  # bars above VAH or below VAL for acceptance
    "poc_migrate_sessions": 3,  # sessions to track POC migration
    "poc_migrate_threshold": 0.002,  # min % POC shift to trigger migration signal
    "lvn_threshold": LVN_THRESHOLD,  # bins below this fraction of avg = LVN
    "lvn_volume_surge": 1.5,  # volume surge multiplier for LVN breakout
}


# ---------------------------------------------------------------------------
# Session Volume Profile Calculator
# ---------------------------------------------------------------------------

@dataclass
class VolumeProfileLevel:
    """Single volume profile computation for a session."""
    poc: float = 0.0          # Point of Control (highest volume price)
    vah: float = 0.0          # Value Area High
    val: float = 0.0          # Value Area Low
    total_volume: float = 0.0
    bin_edges: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_volumes: np.ndarray = field(default_factory=lambda: np.array([]))
    # LVN (Low Volume Node) data
    lvn_zones: list = field(default_factory=list)  # List of (low, high) price zones
    hvn_zones: list = field(default_factory=list)  # High Volume Nodes for reference


def compute_volume_profile(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    num_bins: int = DEFAULT_NUM_BINS,
    value_area_pct: float = VALUE_AREA_PCT,
) -> VolumeProfileLevel:
    """Compute volume profile for a price/volume array.

    Distributes volume across price bins between session low and high,
    then finds POC, VAH, and VAL.
    """
    if len(closes) == 0 or np.sum(volumes) == 0:
        return VolumeProfileLevel()

    price_min = float(np.min(lows))
    price_max = float(np.max(highs))

    if price_max <= price_min:
        return VolumeProfileLevel(
            poc=price_min, vah=price_min, val=price_min,
            total_volume=float(np.sum(volumes)),
        )

    # Create price bins
    bin_edges = np.linspace(price_min, price_max, num_bins + 1)
    bin_volumes = np.zeros(num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Distribute each bar's volume across the bins it spans
    for i in range(len(closes)):
        bar_low = lows[i]
        bar_high = highs[i]
        bar_vol = volumes[i]

        if bar_vol <= 0 or bar_high <= bar_low:
            continue

        # Find bins this bar overlaps
        for b in range(num_bins):
            bin_lo = bin_edges[b]
            bin_hi = bin_edges[b + 1]

            # Overlap between [bar_low, bar_high] and [bin_lo, bin_hi]
            overlap_lo = max(bar_low, bin_lo)
            overlap_hi = min(bar_high, bin_hi)

            if overlap_hi > overlap_lo:
                overlap_frac = (overlap_hi - overlap_lo) / (bar_high - bar_low)
                bin_volumes[b] += bar_vol * overlap_frac

    total_volume = float(np.sum(bin_volumes))
    if total_volume == 0:
        return VolumeProfileLevel(total_volume=0.0)

    # POC: bin with highest volume
    poc_idx = int(np.argmax(bin_volumes))
    poc = float(bin_centers[poc_idx])

    # Value Area: expand from POC until 70% of volume is covered
    va_volume = bin_volumes[poc_idx]
    va_low_idx = poc_idx
    va_high_idx = poc_idx

    while va_volume / total_volume < value_area_pct:
        # Expand toward the side with more volume
        can_go_low = va_low_idx > 0
        can_go_high = va_high_idx < num_bins - 1

        if not can_go_low and not can_go_high:
            break

        low_vol = bin_volumes[va_low_idx - 1] if can_go_low else -1
        high_vol = bin_volumes[va_high_idx + 1] if can_go_high else -1

        if low_vol >= high_vol:
            va_low_idx -= 1
            va_volume += bin_volumes[va_low_idx]
        else:
            va_high_idx += 1
            va_volume += bin_volumes[va_high_idx]

    val = float(bin_edges[va_low_idx])
    vah = float(bin_edges[va_high_idx + 1])

    # Detect LVN and HVN zones
    avg_vol = float(np.mean(bin_volumes[bin_volumes > 0])) if np.any(bin_volumes > 0) else 0
    lvn_zones = []
    hvn_zones = []
    lvn_threshold = 0.3  # bins with < 30% avg volume

    for b in range(num_bins):
        bin_lo = float(bin_edges[b])
        bin_hi = float(bin_edges[b + 1])
        if avg_vol > 0:
            if bin_volumes[b] < avg_vol * lvn_threshold:
                lvn_zones.append((bin_lo, bin_hi))
            elif bin_volumes[b] > avg_vol * 1.5:
                hvn_zones.append((bin_lo, bin_hi))

    return VolumeProfileLevel(
        poc=poc, vah=vah, val=val,
        total_volume=total_volume,
        bin_edges=bin_edges, bin_volumes=bin_volumes,
        lvn_zones=lvn_zones, hvn_zones=hvn_zones,
    )


# ---------------------------------------------------------------------------
# SVP+VWAP Signal Generator
# ---------------------------------------------------------------------------

@dataclass
class SVPVWAPSignals:
    """Generate signals from Session Volume Profile + VWAP.

    V2 additions:
    - POC migration tracking (bullish: POC shifting up; bearish: POC down)
    - VAH/VAL acceptance (sustained trading above/below)
    - LVN breakout detection (explosion trades with volume surge)
    - Trade instrument recommendation (calls/puts/futures)
    """

    session_lookback: int = 75
    num_bins: int = DEFAULT_NUM_BINS
    value_area_pct: float = VALUE_AREA_PCT
    vwap_band_inner: float = 1.0
    vwap_band_outer: float = 2.0
    volume_confirm_mult: float = 1.2
    poc_break_bars: int = 3
    # V2 advanced features
    acceptance_bars: int = 3
    poc_migrate_sessions: int = 3
    poc_migrate_threshold: float = 0.002
    lvn_threshold: float = 0.3
    lvn_volume_surge: float = 1.5

    def _compute_rolling_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute VWAP and bands with session reset."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"].replace(0, 1)
        tp_volume = typical_price * vol

        window = self.session_lookback

        rolling_tp_vol = tp_volume.rolling(window, min_periods=1).sum()
        rolling_vol = vol.rolling(window, min_periods=1).sum()
        vwap_line = rolling_tp_vol / rolling_vol

        sq_diff_vol = ((typical_price - vwap_line) ** 2) * vol
        rolling_sq = sq_diff_vol.rolling(window, min_periods=1).sum()
        variance = rolling_sq / rolling_vol
        std = variance ** 0.5

        result = pd.DataFrame(index=df.index)
        result["vwap"] = vwap_line
        result["vwap_upper_inner"] = vwap_line + self.vwap_band_inner * std
        result["vwap_upper_outer"] = vwap_line + self.vwap_band_outer * std
        result["vwap_lower_inner"] = vwap_line - self.vwap_band_inner * std
        result["vwap_lower_outer"] = vwap_line - self.vwap_band_outer * std
        return result

    def _compute_session_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling session volume profile levels with LVN detection."""
        n = len(df)
        poc = np.full(n, np.nan)
        vah = np.full(n, np.nan)
        val = np.full(n, np.nan)
        # Store LVN zones per bar (as string for DataFrame compatibility)
        lvn_zones_list = [None] * n
        # Store profiles for POC migration tracking
        profiles = [None] * n

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        volumes = df["volume"].values

        for i in range(self.session_lookback, n):
            start = i - self.session_lookback
            profile = compute_volume_profile(
                highs[start:i],
                lows[start:i],
                closes[start:i],
                volumes[start:i],
                self.num_bins,
                self.value_area_pct,
            )
            poc[i] = profile.poc
            vah[i] = profile.vah
            val[i] = profile.val
            lvn_zones_list[i] = profile.lvn_zones
            profiles[i] = profile

        result = pd.DataFrame(index=df.index)
        result["poc"] = poc
        result["vah"] = vah
        result["val"] = val

        # POC migration: track shift over multiple sessions
        poc_shift = np.zeros(n)
        poc_migrate_dir = np.zeros(n)  # 1 = up, -1 = down, 0 = flat
        session_step = max(1, self.session_lookback // 2)

        for i in range(self.session_lookback * self.poc_migrate_sessions, n):
            # Compare current POC with POC from previous sessions
            prev_idx = i - session_step * (self.poc_migrate_sessions - 1)
            if prev_idx >= self.session_lookback and not np.isnan(poc[prev_idx]):
                pct_shift = (poc[i] - poc[prev_idx]) / poc[prev_idx]
                poc_shift[i] = pct_shift
                if pct_shift > self.poc_migrate_threshold:
                    poc_migrate_dir[i] = 1  # POC migrating up
                elif pct_shift < -self.poc_migrate_threshold:
                    poc_migrate_dir[i] = -1  # POC migrating down

        result["poc_shift"] = poc_shift
        result["poc_migrate_dir"] = poc_migrate_dir.astype(int)

        # Store LVN data for breakout detection
        result["_lvn_zones"] = lvn_zones_list
        result["_profiles"] = profiles

        return result

    def _check_lvn_breakout(
        self, close: float, prev_close: float, lvn_zones: list, vol_surge: bool,
    ) -> tuple[bool, bool]:
        """Check if price broke through an LVN zone with volume surge.

        Returns: (lvn_break_up, lvn_break_down)
        """
        if not lvn_zones or not vol_surge:
            return False, False

        lvn_break_up = False
        lvn_break_down = False

        for lvn_lo, lvn_hi in lvn_zones:
            # Breakout up through LVN
            if prev_close < lvn_lo and close > lvn_hi:
                lvn_break_up = True
            # Breakdown through LVN
            elif prev_close > lvn_hi and close < lvn_lo:
                lvn_break_down = True

        return lvn_break_up, lvn_break_down

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate SVP+VWAP signals with V2 advanced features.

        Returns DataFrame with:
        - vwap, vwap bands, poc, vah, val, poc_shift, poc_migrate_dir
        - signal: 1 (long), -1 (short), 0 (no signal)
        - signal_type: SVPSignalType value
        - signal_strength: 1-4 (higher = more confluent)
        - trade_instrument: call/put/futures recommendation

        V2 Signal Types:
        - vah_acceptance: Sustained trading above VAH → LONG (trade calls)
        - val_acceptance: Sustained trading below VAL → SHORT (trade puts)
        - poc_migrate_up: POC shifting higher → LONG (trade calls)
        - poc_migrate_dn: POC shifting lower → SHORT (trade puts)
        - lvn_breakout: Price breaks through LVN with volume → explosion trade
        """
        result = df.copy()

        # VWAP and bands
        vwap_data = self._compute_rolling_vwap(df)
        for col in vwap_data.columns:
            result[col] = vwap_data[col]

        # Session volume profile (includes POC migration and LVN)
        svp_data = self._compute_session_profiles(df)
        for col in svp_data.columns:
            result[col] = svp_data[col]

        # Volume average for confirmation
        result["vol_avg"] = df["volume"].rolling(
            self.session_lookback, min_periods=1
        ).mean()
        result["vol_confirm"] = df["volume"] > (
            result["vol_avg"] * self.volume_confirm_mult
        )
        # Volume surge for LVN breakouts (higher threshold)
        result["vol_surge"] = df["volume"] > (
            result["vol_avg"] * self.lvn_volume_surge
        )

        # Price relative to POC (bars above/below)
        above_poc = (df["close"] > result["poc"]).astype(int)
        result["bars_above_poc"] = above_poc.rolling(
            self.poc_break_bars, min_periods=1
        ).sum()
        below_poc = (df["close"] < result["poc"]).astype(int)
        result["bars_below_poc"] = below_poc.rolling(
            self.poc_break_bars, min_periods=1
        ).sum()

        # VAH/VAL acceptance: sustained bars above/below
        above_vah = (df["close"] > result["vah"]).astype(int)
        result["bars_above_vah"] = above_vah.rolling(
            self.acceptance_bars, min_periods=1
        ).sum()
        below_val = (df["close"] < result["val"]).astype(int)
        result["bars_below_val"] = below_val.rolling(
            self.acceptance_bars, min_periods=1
        ).sum()

        # Generate signals
        n = len(df)
        signal = np.zeros(n)
        signal_type = [""] * n
        signal_strength = np.zeros(n)
        trade_instrument = [""] * n

        for i in range(self.session_lookback + 1, n):
            row = result.iloc[i]
            prev = result.iloc[i - 1]

            if pd.isna(row["poc"]):
                continue

            close = row["close"]
            low = row["low"]
            high = row["high"]
            prev_close = prev["close"]
            vol_ok = row["vol_confirm"]
            vol_surge = row["vol_surge"]

            strength = 0
            sig = 0
            sig_type = ""
            instrument = ""

            # Get LVN zones for this bar
            lvn_zones = row.get("_lvn_zones", None) or []

            # --- ADVANCED SIGNALS (V2) - higher priority ---

            # 1. VAH Acceptance: sustained trading above VAH (bullish)
            vah_acceptance = row["bars_above_vah"] >= self.acceptance_bars

            # 2. VAL Acceptance: sustained trading below VAL (bearish)
            val_acceptance = row["bars_below_val"] >= self.acceptance_bars

            # 3. POC Migration Up/Down
            poc_migrate_up = row["poc_migrate_dir"] == 1
            poc_migrate_dn = row["poc_migrate_dir"] == -1

            # 4. LVN Breakout (explosion trade)
            lvn_break_up, lvn_break_down = self._check_lvn_breakout(
                close, prev_close, lvn_zones, vol_surge,
            )

            # --- BULLISH SIGNALS ---

            # V2: VAH acceptance with POC migrating up → strong bullish
            if vah_acceptance and poc_migrate_up:
                sig = 1
                sig_type = SVPSignalType.VAH_ACCEPTANCE.value
                instrument = TradeInstrument.CALL.value
                strength = 4  # Maximum strength
            # V2: LVN breakout up → explosion trade
            elif lvn_break_up:
                sig = 1
                sig_type = SVPSignalType.LVN_BREAKOUT.value
                instrument = TradeInstrument.CALL.value
                strength = 4
            # V2: VAH acceptance only
            elif vah_acceptance and close > row["vah"]:
                sig = 1
                sig_type = SVPSignalType.VAH_ACCEPTANCE.value
                instrument = TradeInstrument.CALL.value
                strength = 3
            # V2: POC migrating up
            elif poc_migrate_up and vol_ok:
                sig = 1
                sig_type = SVPSignalType.POC_MIGRATE_UP.value
                instrument = TradeInstrument.CALL.value
                strength = 2

            # V1 signals (lower priority if no V2 signal)
            if sig == 0:
                val_bounce = (low <= row["val"]) and (close > row["val"])
                vwap_bounce = (low <= row["vwap_lower_inner"]) and (
                    close > row["vwap_lower_inner"]
                )
                poc_break_up = (
                    row["bars_above_poc"] >= self.poc_break_bars
                    and prev_close <= prev["poc"]
                ) if not pd.isna(prev["poc"]) else False

                if val_bounce or vwap_bounce or poc_break_up:
                    sig = 1
                    instrument = TradeInstrument.CALL.value
                    if val_bounce:
                        sig_type = SVPSignalType.VAL_BOUNCE.value
                        strength += 1
                    if vwap_bounce:
                        sig_type = sig_type or SVPSignalType.VWAP_BOUNCE.value
                        strength += 1
                    if poc_break_up:
                        sig_type = sig_type or SVPSignalType.POC_BREAK_UP.value
                        strength += 1
                    if vol_ok:
                        strength += 1

            # --- BEARISH SIGNALS ---

            if sig == 0:
                # V2: VAL acceptance with POC migrating down → strong bearish
                if val_acceptance and poc_migrate_dn:
                    sig = -1
                    sig_type = SVPSignalType.VAL_ACCEPTANCE.value
                    instrument = TradeInstrument.PUT.value
                    strength = 4
                # V2: LVN breakdown → explosion trade
                elif lvn_break_down:
                    sig = -1
                    sig_type = SVPSignalType.LVN_BREAKOUT.value
                    instrument = TradeInstrument.PUT.value
                    strength = 4
                # V2: VAL acceptance only
                elif val_acceptance and close < row["val"]:
                    sig = -1
                    sig_type = SVPSignalType.VAL_ACCEPTANCE.value
                    instrument = TradeInstrument.PUT.value
                    strength = 3
                # V2: POC migrating down
                elif poc_migrate_dn and vol_ok:
                    sig = -1
                    sig_type = SVPSignalType.POC_MIGRATE_DN.value
                    instrument = TradeInstrument.PUT.value
                    strength = 2
                else:
                    # V1 signals
                    vah_reject = (high >= row["vah"]) and (close < row["vah"])
                    vwap_reject = (high >= row["vwap_upper_inner"]) and (
                        close < row["vwap_upper_inner"]
                    )
                    poc_break_down = (
                        row["bars_below_poc"] >= self.poc_break_bars
                        and prev_close >= prev["poc"]
                    ) if not pd.isna(prev["poc"]) else False

                    if vah_reject or vwap_reject or poc_break_down:
                        sig = -1
                        instrument = TradeInstrument.PUT.value
                        if vah_reject:
                            sig_type = SVPSignalType.VAH_REJECT.value
                            strength += 1
                        if vwap_reject:
                            sig_type = sig_type or SVPSignalType.VWAP_REJECT.value
                            strength += 1
                        if poc_break_down:
                            sig_type = sig_type or SVPSignalType.POC_BREAK_DOWN.value
                            strength += 1
                        if vol_ok:
                            strength += 1

            signal[i] = sig
            signal_type[i] = sig_type
            signal_strength[i] = min(strength, 4)
            trade_instrument[i] = instrument

        result["signal"] = signal.astype(int)
        result["signal_type"] = signal_type
        result["signal_strength"] = signal_strength.astype(int)
        result["trade_instrument"] = trade_instrument

        # Clean up internal columns
        result.drop(columns=["_lvn_zones", "_profiles"], inplace=True, errors="ignore")

        return result


# ---------------------------------------------------------------------------
# SVP+VWAP Trade
# ---------------------------------------------------------------------------

@dataclass
class SVPVWAPTrade:
    """Single SVP+VWAP trade."""

    entry_date: pd.Timestamp
    entry_price: float
    direction: int  # 1 = long, -1 = short
    signal_type: str
    signal_strength: int
    trade_instrument: str = ""  # call, put, or futures

    # SVP levels at entry
    poc_at_entry: float = 0.0
    vah_at_entry: float = 0.0
    val_at_entry: float = 0.0
    vwap_at_entry: float = 0.0
    poc_migrate_dir: int = 0  # 1=up, -1=down, 0=flat

    # Risk management
    sl_level: float = 0.0
    sl_points: float = 0.0
    sl_strategy: str = ""
    tp_level: float = 0.0
    tp_points: float = 0.0
    rr_ratio: float = 2.0

    # Trailing SL
    trailing_mode: str = "none"
    initial_sl_level: float = 0.0
    tsl_level: float = 0.0
    active_milestone: int = -1
    trail_events: list = field(default_factory=list)

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
    mae_points: float = 0.0
    mfe_points: float = 0.0
    bars_held: int = 0

    @property
    def is_winner(self) -> bool:
        return self.pnl_points > 0

    @property
    def r_multiple(self) -> float:
        if self.sl_points > 0:
            return self.pnl_points / self.sl_points
        return 0.0


# ---------------------------------------------------------------------------
# SVP+VWAP Backtest Result
# ---------------------------------------------------------------------------

@dataclass
class SVPVWAPResult:
    """Backtest result for one SVP+VWAP configuration."""

    sl_strategy: str
    position_sizing: str
    trailing_mode: str
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
        return float(np.mean([t.pnl_amount for t in self.winners]))

    @property
    def avg_loss(self) -> float:
        if not self.losers:
            return 0.0
        return abs(float(np.mean([t.pnl_amount for t in self.losers])))

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
        return float(np.mean([t.r_multiple for t in self.trades]))

    @property
    def expectancy(self) -> float:
        if self.num_trades == 0:
            return 0.0
        wr = len(self.winners) / self.num_trades
        avg_win_r = float(np.mean([t.r_multiple for t in self.winners])) if self.winners else 0
        avg_loss_r = abs(float(np.mean([t.r_multiple for t in self.losers]))) if self.losers else 0
        return (wr * avg_win_r) - ((1 - wr) * avg_loss_r)

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        equity = [0.0]
        for t in self.trades:
            equity.append(equity[-1] + t.pnl_amount)
        equity_arr = np.array(equity)
        peaks = np.maximum.accumulate(equity_arr)
        drawdowns = peaks - equity_arr
        return float(np.max(drawdowns))

    @property
    def max_drawdown_pct(self) -> float:
        if not self.trades or "capital" not in self.params:
            return 0.0
        return (self.max_drawdown / self.params["capital"]) * 100

    @property
    def sharpe_ratio(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        returns = [t.pnl_pct for t in self.trades]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0.0
        annualization = np.sqrt(250 / max(1, len(self.trades)))
        return float((mean_ret / std_ret) * annualization)

    @property
    def calmar_ratio(self) -> float:
        if self.max_drawdown == 0:
            return float("inf") if self.total_pnl > 0 else 0.0
        return self.total_pnl / self.max_drawdown

    @property
    def avg_bars_held(self) -> float:
        if not self.trades:
            return 0.0
        return float(np.mean([t.bars_held for t in self.trades]))

    def to_dict(self) -> dict:
        return {
            "strategy": "SVP+VWAP",
            "sl_strategy": self.sl_strategy,
            "trailing_mode": self.trailing_mode,
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
# SVP+VWAP Backtest Engine
# ---------------------------------------------------------------------------

class SVPVWAPEngine:
    """Backtest engine for Session Volume Profile + VWAP strategy.

    Signal logic:
      LONG when:
        - Price bounces off VAL (session low-volume support)
        - Price bounces off VWAP lower band
        - Price breaks above POC with sustained bars
      SHORT when:
        - Price rejects from VAH (session high-volume resistance)
        - Price rejects from VWAP upper band
        - Price breaks below POC with sustained bars

    Trailing SL modes: NONE, BREAKEVEN, STEPPED.
    """

    def __init__(
        self,
        sl_strategy: SLStrategy = SLStrategy.ATR,
        position_sizing: PositionSizing = PositionSizing.RISK_BASED,
        tp_rr_ratio: float = 2.0,
        trailing_mode: TrailingMode = TrailingMode.NONE,
        **config,
    ):
        self.sl_strategy = sl_strategy
        self.position_sizing = position_sizing
        self.tp_rr_ratio = tp_rr_ratio
        self.trailing_mode = trailing_mode

        self.config = {**SVP_DEFAULT_CONFIG, **config}
        self.config["trailing_mode"] = trailing_mode
        self.trail_milestones = self.config.get("trail_milestones", TRAIL_MILESTONES)

        self.signal_gen = SVPVWAPSignals(
            session_lookback=self.config["session_lookback"],
            num_bins=self.config["num_bins"],
            value_area_pct=self.config["value_area_pct"],
            vwap_band_inner=self.config["vwap_band_inner"],
            vwap_band_outer=self.config["vwap_band_outer"],
            volume_confirm_mult=self.config["volume_confirm_mult"],
            poc_break_bars=self.config["poc_break_bars"],
            # V2 advanced features
            acceptance_bars=self.config["acceptance_bars"],
            poc_migrate_sessions=self.config["poc_migrate_sessions"],
            poc_migrate_threshold=self.config["poc_migrate_threshold"],
            lvn_threshold=self.config["lvn_threshold"],
            lvn_volume_surge=self.config["lvn_volume_surge"],
        )
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

    def _build_trail_thresholds(
        self, entry_price: float, sl_points: float, direction: int,
    ) -> list[tuple[float, float]]:
        thresholds = []
        for profit_mult, lock_mult in self.trail_milestones:
            if direction == 1:
                threshold_price = entry_price + sl_points * profit_mult
                new_sl = entry_price + sl_points * lock_mult
            else:
                threshold_price = entry_price - sl_points * profit_mult
                new_sl = entry_price - sl_points * lock_mult
            thresholds.append((threshold_price, new_sl))
        return thresholds

    def _update_trailing_sl(
        self, position: SVPVWAPTrade, high: float, low: float, timestamp,
    ) -> None:
        if self.trailing_mode == TrailingMode.NONE:
            return

        thresholds = self._build_trail_thresholds(
            position.entry_price, position.sl_points, position.direction,
        )

        if self.trailing_mode == TrailingMode.BREAKEVEN:
            thresholds = thresholds[:1] if thresholds else []

        for i, (threshold_price, new_sl) in enumerate(thresholds):
            if i <= position.active_milestone:
                continue

            triggered = False
            if position.direction == 1:
                triggered = high >= threshold_price
                if triggered and new_sl > position.tsl_level:
                    position.tsl_level = new_sl
            else:
                triggered = low <= threshold_price
                if triggered and new_sl < position.tsl_level:
                    position.tsl_level = new_sl

            if triggered:
                position.active_milestone = i
                position.trail_events.append({
                    "timestamp": timestamp,
                    "milestone": i,
                    "price_trigger": threshold_price,
                    "new_sl": new_sl,
                })

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.signal_gen.generate(df)
        result["atr"] = atr(df, self.config["atr_period"])
        result["swing_high"] = swing_high(df, self.config["swing_lookback"])
        result["swing_low"] = swing_low(df, self.config["swing_lookback"])
        return result

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "NIFTY",
        timeframe: str = "5min",
    ) -> SVPVWAPResult:
        """Run backtest on OHLCV data."""

        data = self._prepare_data(df)
        trades: list[SVPVWAPTrade] = []
        position: Optional[SVPVWAPTrade] = None

        start_idx = self.config["session_lookback"] + 1

        for i in range(start_idx, len(data)):
            row = data.iloc[i]
            signal = int(row["signal"])

            # --- Check exits ---
            if position is not None:
                exit_reason = None
                exit_price = None

                active_sl = (
                    position.tsl_level if position.tsl_level != 0
                    else position.sl_level
                )

                if position.direction == 1:
                    if row["low"] <= active_sl:
                        exit_reason = (
                            "trailing_sl" if position.active_milestone >= 0
                            else "sl_hit"
                        )
                        exit_price = active_sl
                    elif row["high"] >= position.tp_level:
                        exit_reason = "tp_hit"
                        exit_price = position.tp_level
                else:
                    if row["high"] >= active_sl:
                        exit_reason = (
                            "trailing_sl" if position.active_milestone >= 0
                            else "sl_hit"
                        )
                        exit_price = active_sl
                    elif row["low"] <= position.tp_level:
                        exit_reason = "tp_hit"
                        exit_price = position.tp_level

                # Signal reversal
                if exit_reason is None and signal != 0 and signal != position.direction:
                    exit_reason = "signal_reversal"
                    exit_price = row["close"]

                # MAE / MFE
                if position.direction == 1:
                    adverse = position.entry_price - row["low"]
                    favorable = row["high"] - position.entry_price
                else:
                    adverse = row["high"] - position.entry_price
                    favorable = position.entry_price - row["low"]
                position.mae_points = max(position.mae_points, adverse)
                position.mfe_points = max(position.mfe_points, favorable)
                position.bars_held += 1

                # Update trailing SL
                if exit_reason is None:
                    self._update_trailing_sl(
                        position, row["high"], row["low"], row.name,
                    )

                # Close trade
                if exit_reason:
                    position.exit_date = row.name
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason

                    if position.direction == 1:
                        position.pnl_points = exit_price - position.entry_price
                    else:
                        position.pnl_points = position.entry_price - exit_price

                    position.pnl_amount = position.pnl_points * position.qty
                    position.pnl_pct = (
                        (position.pnl_points / position.entry_price) * 100
                    )
                    trades.append(position)
                    position = None

            # --- Open new position ---
            if position is None and signal != 0:
                entry_price = row["close"]
                sl_info = self.sl_calc.calculate(data, signal, i)
                tp_points = sl_info["sl_points"] * self.tp_rr_ratio
                if signal == 1:
                    tp_level = entry_price + tp_points
                else:
                    tp_level = entry_price - tp_points

                pos_info = self.pos_sizer.calculate(entry_price, sl_info["sl_points"])
                initial_sl = sl_info["sl_level"]

                position = SVPVWAPTrade(
                    entry_date=row.name,
                    entry_price=entry_price,
                    direction=signal,
                    signal_type=row.get("signal_type", ""),
                    signal_strength=int(row.get("signal_strength", 0)),
                    trade_instrument=row.get("trade_instrument", ""),
                    poc_at_entry=row.get("poc", 0.0) if not pd.isna(row.get("poc", np.nan)) else 0.0,
                    vah_at_entry=row.get("vah", 0.0) if not pd.isna(row.get("vah", np.nan)) else 0.0,
                    val_at_entry=row.get("val", 0.0) if not pd.isna(row.get("val", np.nan)) else 0.0,
                    vwap_at_entry=row.get("vwap", 0.0) if not pd.isna(row.get("vwap", np.nan)) else 0.0,
                    poc_migrate_dir=int(row.get("poc_migrate_dir", 0)),
                    sl_level=sl_info["sl_level"],
                    sl_points=sl_info["sl_points"],
                    sl_strategy=sl_info["sl_strategy"],
                    tp_level=tp_level,
                    tp_points=tp_points,
                    rr_ratio=self.tp_rr_ratio,
                    trailing_mode=self.trailing_mode.value,
                    initial_sl_level=initial_sl,
                    tsl_level=initial_sl,
                    active_milestone=-1,
                    trail_events=[],
                    num_lots=pos_info["num_lots"],
                    qty=pos_info["qty"],
                    position_value=pos_info["position_value"],
                    risk_amount=pos_info["risk_amount"],
                )

        # Close open position at end
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

        return SVPVWAPResult(
            sl_strategy=self.sl_strategy.value,
            position_sizing=self.position_sizing.value,
            trailing_mode=self.trailing_mode.value,
            timeframe=timeframe,
            symbol=symbol,
            trades=trades,
            params=self.config,
        )


# ---------------------------------------------------------------------------
# Grid Runner
# ---------------------------------------------------------------------------

def run_svp_vwap_grid(
    df: pd.DataFrame,
    symbol: str = "NIFTY",
    timeframe: str = "5min",
    sl_strategies: list[SLStrategy] = None,
    trailing_modes: list[TrailingMode] = None,
    tp_rr_ratio: float = 2.0,
    **config,
) -> pd.DataFrame:
    """Run grid of SVP+VWAP backtests across SL and trailing configs."""
    if sl_strategies is None:
        sl_strategies = list(SLStrategy)
    if trailing_modes is None:
        trailing_modes = list(TrailingMode)

    results = []
    for sl_strat in sl_strategies:
        for trail_mode in trailing_modes:
            engine = SVPVWAPEngine(
                sl_strategy=sl_strat,
                tp_rr_ratio=tp_rr_ratio,
                trailing_mode=trail_mode,
                **config,
            )
            result = engine.run(df, symbol, timeframe)
            results.append(result.to_dict())

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.sort_values("total_pnl", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)
    return results_df


def build_svp_trade_log(result: SVPVWAPResult) -> pd.DataFrame:
    """Build detailed trade log from SVP+VWAP result."""
    if not result.trades:
        return pd.DataFrame()

    rows = []
    for t in result.trades:
        rows.append({
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "signal_type": t.signal_type,
            "signal_strength": t.signal_strength,
            "trade_instrument": t.trade_instrument,  # call/put/futures
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2),
            "poc_at_entry": round(t.poc_at_entry, 2),
            "vah_at_entry": round(t.vah_at_entry, 2),
            "val_at_entry": round(t.val_at_entry, 2),
            "vwap_at_entry": round(t.vwap_at_entry, 2),
            "poc_migrate_dir": t.poc_migrate_dir,  # 1=up, -1=down, 0=flat
            "sl_level": round(t.sl_level, 2),
            "tp_level": round(t.tp_level, 2),
            "sl_points": round(t.sl_points, 1),
            "sl_strategy": t.sl_strategy,
            "trailing_mode": t.trailing_mode,
            "tsl_level": round(t.tsl_level, 2) if t.tsl_level else t.sl_level,
            "trail_milestones_hit": t.active_milestone + 1 if t.active_milestone >= 0 else 0,
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
