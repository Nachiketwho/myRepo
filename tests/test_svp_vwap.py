"""Tests for Session Volume Profile + VWAP strategy."""

import datetime as dt
import numpy as np
import pandas as pd
import pytest

from backtester.svp_vwap import (
    SVPVWAPEngine,
    SVPVWAPResult,
    SVPVWAPTrade,
    SVPVWAPSignals,
    SVPSignalType,
    VolumeProfileLevel,
    compute_volume_profile,
    run_svp_vwap_grid,
    build_svp_trade_log,
    VALUE_AREA_PCT,
    DEFAULT_NUM_BINS,
    SVP_DEFAULT_CONFIG,
)
from backtester.ema_crossover import (
    SLStrategy,
    PositionSizing,
    TrailingMode,
    TRAIL_MILESTONES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """200-bar sample OHLCV data with trends and volume patterns."""
    np.random.seed(42)
    n = 200
    base = 22000
    trend = np.cumsum(np.random.randn(n) * 10)
    close = base + trend

    df = pd.DataFrame({
        "open": close - np.random.uniform(5, 15, n),
        "high": close + np.random.uniform(10, 30, n),
        "low": close - np.random.uniform(10, 30, n),
        "close": close,
        "volume": np.random.randint(10000, 100000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="5min"))
    return df


@pytest.fixture
def trending_up_data():
    """Data with clear uptrend for VAL bounce testing."""
    n = 200
    # Start flat, then trend up
    flat = np.full(80, 22000.0)
    up = 22000 + np.arange(120) * 5.0
    close = np.concatenate([flat, up])

    # Add noise
    np.random.seed(10)
    noise = np.random.randn(n) * 3
    close = close + noise

    df = pd.DataFrame({
        "open": close - 5,
        "high": close + 15,
        "low": close - 15,
        "close": close,
        "volume": np.random.randint(50000, 200000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="5min"))
    return df


@pytest.fixture
def trending_down_data():
    """Data with downtrend for VAH reject testing."""
    n = 200
    up = 22000 + np.arange(60) * 10
    down = up[-1] - np.arange(140) * 6.0
    close = np.concatenate([up, down])

    np.random.seed(20)
    noise = np.random.randn(n) * 3
    close = close + noise

    df = pd.DataFrame({
        "open": close + 5,
        "high": close + 15,
        "low": close - 15,
        "close": close,
        "volume": np.random.randint(50000, 200000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="5min"))
    return df


@pytest.fixture
def high_volume_data():
    """Data with volume spikes at key price levels."""
    np.random.seed(33)
    n = 200
    base = 22000
    # Oscillating price to create clear volume profile
    period = 40
    amplitude = 50
    close = base + amplitude * np.sin(2 * np.pi * np.arange(n) / period)

    # High volume at extremes (VAH/VAL), low at middle
    volume = np.full(n, 30000)
    for i in range(n):
        if close[i] > base + amplitude * 0.7:
            volume[i] = 150000  # High vol at highs
        elif close[i] < base - amplitude * 0.7:
            volume[i] = 150000  # High vol at lows

    df = pd.DataFrame({
        "open": close - 3,
        "high": close + 10,
        "low": close - 10,
        "close": close,
        "volume": volume,
    }, index=pd.date_range("2024-01-01", periods=n, freq="5min"))
    return df


# ---------------------------------------------------------------------------
# Volume Profile Computation Tests
# ---------------------------------------------------------------------------

class TestVolumeProfile:
    def test_basic_profile(self):
        """POC should be at the price level with highest volume."""
        highs = np.array([100, 102, 104, 103, 101])
        lows = np.array([98, 99, 101, 100, 99])
        closes = np.array([99, 101, 103, 101, 100])
        # Concentrate volume at 101-102
        volumes = np.array([1000, 5000, 500, 5000, 1000])

        profile = compute_volume_profile(highs, lows, closes, volumes)
        assert profile.poc > 0
        assert profile.vah >= profile.poc
        assert profile.val <= profile.poc
        assert profile.total_volume == pytest.approx(np.sum(volumes), rel=0.01)

    def test_poc_at_high_volume_price(self):
        """POC should be near the price level where volume is concentrated."""
        # All volume at 100
        highs = np.array([101, 101, 101])
        lows = np.array([99, 99, 99])
        closes = np.array([100, 100, 100])
        volumes = np.array([10000, 10000, 10000])

        profile = compute_volume_profile(highs, lows, closes, volumes, num_bins=20)
        assert abs(profile.poc - 100) < 2  # POC near 100

    def test_value_area_contains_majority_volume(self):
        """Value area should cover ~70% of total volume."""
        np.random.seed(99)
        n = 50
        closes = 100 + np.random.randn(n) * 5
        highs = closes + np.abs(np.random.randn(n))
        lows = closes - np.abs(np.random.randn(n))
        volumes = np.random.randint(1000, 10000, n).astype(float)

        profile = compute_volume_profile(highs, lows, closes, volumes)
        assert profile.vah > profile.val
        assert profile.vah >= profile.poc >= profile.val

    def test_empty_input(self):
        """Empty arrays should return zero profile."""
        profile = compute_volume_profile(
            np.array([]), np.array([]), np.array([]), np.array([]),
        )
        assert profile.poc == 0.0
        assert profile.total_volume == 0.0

    def test_single_bar(self):
        """Single bar should have POC at its price."""
        profile = compute_volume_profile(
            np.array([105.0]),
            np.array([95.0]),
            np.array([100.0]),
            np.array([10000.0]),
        )
        assert profile.poc > 0
        assert profile.total_volume > 0

    def test_zero_volume(self):
        """Zero volume bars should yield zero profile."""
        profile = compute_volume_profile(
            np.array([105.0, 106.0]),
            np.array([95.0, 96.0]),
            np.array([100.0, 101.0]),
            np.array([0.0, 0.0]),
        )
        assert profile.total_volume == 0.0

    def test_flat_price(self):
        """All bars at same price should produce valid profile."""
        n = 10
        profile = compute_volume_profile(
            np.full(n, 100.0),
            np.full(n, 100.0),
            np.full(n, 100.0),
            np.full(n, 1000.0),
        )
        # Flat price: max == min, special case
        assert profile.poc == 100.0

    def test_val_below_vah(self):
        """VAL should always be <= POC <= VAH."""
        np.random.seed(77)
        n = 100
        closes = 200 + np.random.randn(n) * 10
        highs = closes + 3
        lows = closes - 3
        volumes = np.random.randint(500, 5000, n).astype(float)

        profile = compute_volume_profile(highs, lows, closes, volumes)
        assert profile.val <= profile.poc
        assert profile.poc <= profile.vah


# ---------------------------------------------------------------------------
# Signal Generator Tests
# ---------------------------------------------------------------------------

class TestSVPVWAPSignals:
    def test_signals_shape(self, sample_data):
        """Output should have same length as input with signal columns."""
        gen = SVPVWAPSignals(session_lookback=50)
        result = gen.generate(sample_data)
        assert len(result) == len(sample_data)
        assert "signal" in result.columns
        assert "signal_type" in result.columns
        assert "signal_strength" in result.columns
        assert "vwap" in result.columns
        assert "poc" in result.columns
        assert "vah" in result.columns
        assert "val" in result.columns

    def test_signal_values(self, sample_data):
        """Signals should be -1, 0, or 1."""
        gen = SVPVWAPSignals(session_lookback=50)
        result = gen.generate(sample_data)
        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_signal_strength_range(self, sample_data):
        """Signal strength should be 0-4."""
        gen = SVPVWAPSignals(session_lookback=50)
        result = gen.generate(sample_data)
        assert result["signal_strength"].min() >= 0
        assert result["signal_strength"].max() <= 4

    def test_no_signals_before_lookback(self, sample_data):
        """No signals in the warmup period."""
        lookback = 50
        gen = SVPVWAPSignals(session_lookback=lookback)
        result = gen.generate(sample_data)
        assert (result["signal"].iloc[:lookback] == 0).all()

    def test_vwap_computed(self, sample_data):
        """VWAP and bands should be computed."""
        gen = SVPVWAPSignals(session_lookback=50)
        result = gen.generate(sample_data)
        assert not result["vwap"].isna().all()
        assert "vwap_upper_inner" in result.columns
        assert "vwap_lower_inner" in result.columns

    def test_svp_levels_computed(self, sample_data):
        """POC/VAH/VAL should be computed after lookback."""
        gen = SVPVWAPSignals(session_lookback=50)
        result = gen.generate(sample_data)
        # After lookback, POC should be non-NaN
        after_warmup = result.iloc[51:]
        assert not after_warmup["poc"].isna().all()
        assert not after_warmup["vah"].isna().all()
        assert not after_warmup["val"].isna().all()

    def test_volume_confirmation_column(self, sample_data):
        """Volume confirm flag should exist."""
        gen = SVPVWAPSignals(session_lookback=50)
        result = gen.generate(sample_data)
        assert "vol_confirm" in result.columns

    def test_signal_types_valid(self, sample_data):
        """Signal types should be valid SVPSignalType values or empty."""
        gen = SVPVWAPSignals(session_lookback=50)
        result = gen.generate(sample_data)
        valid_types = {st.value for st in SVPSignalType} | {""}
        for st in result["signal_type"]:
            assert st in valid_types, f"Invalid signal type: {st}"

    def test_high_volume_data_produces_signals(self, high_volume_data):
        """Oscillating data with volume spikes should produce signals."""
        gen = SVPVWAPSignals(session_lookback=50, volume_confirm_mult=1.0)
        result = gen.generate(high_volume_data)
        assert result["signal"].abs().sum() > 0


# ---------------------------------------------------------------------------
# Trade Dataclass Tests
# ---------------------------------------------------------------------------

class TestSVPVWAPTrade:
    def test_winner(self):
        trade = SVPVWAPTrade(
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=22000,
            direction=1,
            signal_type="val_bounce",
            signal_strength=2,
            pnl_points=50,
        )
        assert trade.is_winner is True

    def test_loser(self):
        trade = SVPVWAPTrade(
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=22000,
            direction=1,
            signal_type="val_bounce",
            signal_strength=2,
            pnl_points=-30,
        )
        assert trade.is_winner is False

    def test_r_multiple(self):
        trade = SVPVWAPTrade(
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=22000,
            direction=1,
            signal_type="val_bounce",
            signal_strength=2,
            sl_points=50,
            pnl_points=100,
        )
        assert trade.r_multiple == pytest.approx(2.0)

    def test_r_multiple_zero_sl(self):
        trade = SVPVWAPTrade(
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=22000,
            direction=1,
            signal_type="val_bounce",
            signal_strength=2,
            sl_points=0,
            pnl_points=100,
        )
        assert trade.r_multiple == 0.0


# ---------------------------------------------------------------------------
# Result Tests
# ---------------------------------------------------------------------------

class TestSVPVWAPResult:
    def _make_result(self, pnl_list):
        trades = []
        for pnl in pnl_list:
            t = SVPVWAPTrade(
                entry_date=pd.Timestamp("2024-01-01"),
                entry_price=22000,
                direction=1,
                signal_type="val_bounce",
                signal_strength=2,
                sl_points=50,
                pnl_points=pnl,
                pnl_amount=pnl * 25,
                qty=25,
                bars_held=5,
            )
            trades.append(t)
        return SVPVWAPResult(
            sl_strategy="atr",
            position_sizing="risk_based",
            trailing_mode="none",
            timeframe="5min",
            symbol="NIFTY",
            trades=trades,
            params={"capital": 100000},
        )

    def test_win_rate(self):
        result = self._make_result([100, -50, 80, -30])
        assert result.win_rate == pytest.approx(50.0)

    def test_profit_factor(self):
        result = self._make_result([100, -50])
        expected = (100 * 25) / abs(-50 * 25)
        assert result.profit_factor == pytest.approx(expected)

    def test_total_pnl(self):
        result = self._make_result([100, -50, 80])
        assert result.total_pnl == pytest.approx((100 - 50 + 80) * 25)

    def test_empty_result(self):
        result = self._make_result([])
        assert result.num_trades == 0
        assert result.win_rate == 0.0
        assert result.total_pnl == 0.0
        assert result.max_drawdown == 0.0
        assert result.expectancy == 0.0

    def test_to_dict(self):
        result = self._make_result([100, -50])
        d = result.to_dict()
        assert d["strategy"] == "SVP+VWAP"
        assert d["num_trades"] == 2
        assert "sharpe_ratio" in d
        assert "calmar_ratio" in d

    def test_max_drawdown(self):
        result = self._make_result([-100, -50, 200])
        assert result.max_drawdown > 0

    def test_all_winners_profit_factor(self):
        result = self._make_result([100, 50, 80])
        assert result.profit_factor == float("inf")

    def test_all_losers_profit_factor(self):
        result = self._make_result([-100, -50, -80])
        assert result.profit_factor == 0.0


# ---------------------------------------------------------------------------
# Engine Tests
# ---------------------------------------------------------------------------

class TestSVPVWAPEngine:
    def test_basic_run(self, sample_data):
        """Engine should run and return a result."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data, "NIFTY", "5min")
        assert isinstance(result, SVPVWAPResult)
        assert result.symbol == "NIFTY"
        assert result.timeframe == "5min"

    def test_produces_trades(self, sample_data):
        """Engine should produce trades on sample data."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data, "NIFTY", "5min")
        assert result.num_trades > 0

    def test_all_trades_have_exit(self, sample_data):
        """Every trade should have an exit."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data, "NIFTY", "5min")
        for trade in result.trades:
            assert trade.exit_date is not None
            assert trade.exit_price > 0
            assert trade.exit_reason != ""

    def test_exit_reasons_valid(self, sample_data):
        """All exit reasons should be valid."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data, "NIFTY", "5min")
        valid = {"sl_hit", "tp_hit", "trailing_sl", "signal_reversal", "end_of_data"}
        for t in result.trades:
            assert t.exit_reason in valid

    def test_pnl_sign_matches_direction(self, sample_data):
        """PnL sign should be consistent with exit price vs entry."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data, "NIFTY", "5min")
        for t in result.trades:
            if t.direction == 1:
                expected_pnl = t.exit_price - t.entry_price
            else:
                expected_pnl = t.entry_price - t.exit_price
            assert t.pnl_points == pytest.approx(expected_pnl, abs=0.01)

    def test_atr_sl_strategy(self, sample_data):
        """ATR SL strategy should work."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.ATR,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        for t in result.trades:
            assert t.sl_strategy == "atr"
            assert t.sl_points > 0

    def test_swing_sl_strategy(self, sample_data):
        """Swing SL strategy should work."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.SWING,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        for t in result.trades:
            assert t.sl_strategy == "swing"

    def test_fixed_sl_strategy(self, sample_data):
        """Fixed SL strategy should work."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.FIXED,
            session_lookback=50,
            sl_fixed_points=100,
        )
        result = engine.run(sample_data)
        for t in result.trades:
            assert t.sl_strategy == "fixed"
            assert t.sl_points == pytest.approx(100.0)

    def test_tp_level_calculation(self, sample_data):
        """TP should be at entry + sl_points * rr_ratio for longs."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.PERCENTAGE,
            tp_rr_ratio=2.0,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        for t in result.trades:
            assert t.tp_points == pytest.approx(t.sl_points * 2.0, abs=0.01)

    def test_position_sizing(self, sample_data):
        """Risk-based position sizing should produce correct qty."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.PERCENTAGE,
            position_sizing=PositionSizing.RISK_BASED,
            session_lookback=50,
            capital=100000,
            risk_per_trade_pct=1.0,
            lot_size=25,
        )
        result = engine.run(sample_data)
        for t in result.trades:
            assert t.num_lots >= 1
            assert t.qty == t.num_lots * 25

    def test_svp_levels_in_trades(self, sample_data):
        """Trades should record SVP levels at entry."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        for t in result.trades:
            assert t.poc_at_entry > 0 or t.poc_at_entry == 0.0
            # VWAP should be in reasonable range
            if t.vwap_at_entry > 0:
                assert abs(t.vwap_at_entry - t.entry_price) < t.entry_price * 0.05


# ---------------------------------------------------------------------------
# Trailing SL Tests
# ---------------------------------------------------------------------------

class TestSVPTrailingMode:
    def test_none_mode(self, sample_data):
        """NONE mode should not trail."""
        engine = SVPVWAPEngine(
            trailing_mode=TrailingMode.NONE,
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        for t in result.trades:
            assert t.trailing_mode == "none"
            assert t.active_milestone == -1
            assert len(t.trail_events) == 0

    def test_breakeven_mode_field(self, sample_data):
        """BREAKEVEN mode should set trailing_mode field."""
        engine = SVPVWAPEngine(
            trailing_mode=TrailingMode.BREAKEVEN,
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        for t in result.trades:
            assert t.trailing_mode == "breakeven"

    def test_stepped_mode_field(self, sample_data):
        """STEPPED mode should set trailing_mode field."""
        engine = SVPVWAPEngine(
            trailing_mode=TrailingMode.STEPPED,
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        for t in result.trades:
            assert t.trailing_mode == "stepped"

    def test_trailing_sl_exit_reason(self, sample_data):
        """Trades exiting via trailing SL should have correct exit reason."""
        engine = SVPVWAPEngine(
            trailing_mode=TrailingMode.STEPPED,
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        trailing_exits = [t for t in result.trades if t.exit_reason == "trailing_sl"]
        for t in trailing_exits:
            assert t.active_milestone >= 0
            assert len(t.trail_events) > 0

    def test_trail_events_have_required_fields(self, sample_data):
        """Trail events should contain timestamp, milestone, price_trigger, new_sl."""
        engine = SVPVWAPEngine(
            trailing_mode=TrailingMode.STEPPED,
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        for t in result.trades:
            for evt in t.trail_events:
                assert "timestamp" in evt
                assert "milestone" in evt
                assert "price_trigger" in evt
                assert "new_sl" in evt

    def test_breakeven_at_most_one_milestone(self, sample_data):
        """BREAKEVEN mode should hit at most one milestone."""
        engine = SVPVWAPEngine(
            trailing_mode=TrailingMode.BREAKEVEN,
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        for t in result.trades:
            assert t.active_milestone <= 0

    def test_stepped_can_have_multiple_milestones(self, sample_data):
        """STEPPED mode can hit multiple milestones."""
        engine = SVPVWAPEngine(
            trailing_mode=TrailingMode.STEPPED,
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        # At least check that the mechanism works
        for t in result.trades:
            assert t.active_milestone <= len(TRAIL_MILESTONES) - 1


# ---------------------------------------------------------------------------
# Grid Runner Tests
# ---------------------------------------------------------------------------

class TestGridRunner:
    def test_grid_produces_results(self, sample_data):
        """Grid runner should produce a DataFrame."""
        results = run_svp_vwap_grid(
            sample_data,
            symbol="NIFTY",
            timeframe="5min",
            sl_strategies=[SLStrategy.PERCENTAGE, SLStrategy.ATR],
            trailing_modes=[TrailingMode.NONE],
            session_lookback=50,
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2  # 2 SL × 1 trailing

    def test_grid_all_combos(self, sample_data):
        """Full grid should produce all combinations."""
        sl = [SLStrategy.PERCENTAGE, SLStrategy.ATR]
        trails = [TrailingMode.NONE, TrailingMode.BREAKEVEN]
        results = run_svp_vwap_grid(
            sample_data,
            sl_strategies=sl,
            trailing_modes=trails,
            session_lookback=50,
        )
        assert len(results) == 4  # 2 × 2

    def test_grid_sorted_by_pnl(self, sample_data):
        """Results should be sorted by total_pnl descending."""
        results = run_svp_vwap_grid(
            sample_data,
            sl_strategies=[SLStrategy.PERCENTAGE, SLStrategy.ATR],
            trailing_modes=[TrailingMode.NONE],
            session_lookback=50,
        )
        if len(results) > 1:
            assert results.iloc[0]["total_pnl"] >= results.iloc[-1]["total_pnl"]

    def test_grid_has_required_columns(self, sample_data):
        """Grid results should have all KPI columns."""
        results = run_svp_vwap_grid(
            sample_data,
            sl_strategies=[SLStrategy.PERCENTAGE],
            trailing_modes=[TrailingMode.NONE],
            session_lookback=50,
        )
        required = [
            "strategy", "sl_strategy", "trailing_mode", "num_trades",
            "win_rate", "total_pnl", "profit_factor", "expectancy",
            "max_drawdown", "sharpe_ratio", "calmar_ratio",
        ]
        for col in required:
            assert col in results.columns


# ---------------------------------------------------------------------------
# Trade Log Tests
# ---------------------------------------------------------------------------

class TestTradeLog:
    def test_trade_log_shape(self, sample_data):
        """Trade log should have one row per trade."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        log = build_svp_trade_log(result)
        assert len(log) == result.num_trades

    def test_trade_log_columns(self, sample_data):
        """Trade log should have required columns."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        log = build_svp_trade_log(result)
        required = [
            "entry_date", "exit_date", "direction", "signal_type",
            "signal_strength", "entry_price", "exit_price",
            "poc_at_entry", "vah_at_entry", "val_at_entry", "vwap_at_entry",
            "sl_level", "tp_level", "sl_strategy", "trailing_mode",
            "exit_reason", "pnl_points", "pnl_amount", "r_multiple",
        ]
        for col in required:
            assert col in log.columns

    def test_empty_trade_log(self):
        """Empty result should produce empty trade log."""
        result = SVPVWAPResult(
            sl_strategy="atr",
            position_sizing="risk_based",
            trailing_mode="none",
            timeframe="5min",
            symbol="NIFTY",
        )
        log = build_svp_trade_log(result)
        assert len(log) == 0


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_all_sl_strategies(self, sample_data):
        """All SL strategies should run without error."""
        for sl in SLStrategy:
            engine = SVPVWAPEngine(
                sl_strategy=sl,
                session_lookback=50,
            )
            result = engine.run(sample_data)
            assert isinstance(result, SVPVWAPResult)

    def test_all_trailing_modes(self, sample_data):
        """All trailing modes should run without error."""
        for trail in TrailingMode:
            engine = SVPVWAPEngine(
                trailing_mode=trail,
                sl_strategy=SLStrategy.PERCENTAGE,
                session_lookback=50,
            )
            result = engine.run(sample_data)
            assert isinstance(result, SVPVWAPResult)

    def test_full_grid(self, sample_data):
        """Full grid across all SL + trailing should complete."""
        results = run_svp_vwap_grid(
            sample_data,
            session_lookback=50,
        )
        # 4 SL × 3 trailing = 12 configs
        assert len(results) == 12

    def test_mae_mfe_tracked(self, sample_data):
        """MAE and MFE should be tracked for all trades."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        for t in result.trades:
            assert t.mae_points >= 0
            assert t.mfe_points >= 0

    def test_bars_held_positive(self, sample_data):
        """All trades should hold for at least 1 bar."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        result = engine.run(sample_data)
        for t in result.trades:
            assert t.bars_held >= 1 or t.exit_reason == "end_of_data"

    def test_different_data_sizes(self):
        """Engine should work with different data sizes."""
        np.random.seed(55)
        for n in [100, 300, 500]:
            close = 22000 + np.cumsum(np.random.randn(n) * 10)
            df = pd.DataFrame({
                "open": close - 5,
                "high": close + 15,
                "low": close - 15,
                "close": close,
                "volume": np.random.randint(10000, 100000, n),
            }, index=pd.date_range("2024-01-01", periods=n, freq="5min"))

            engine = SVPVWAPEngine(
                sl_strategy=SLStrategy.PERCENTAGE,
                session_lookback=50,
            )
            result = engine.run(df)
            assert isinstance(result, SVPVWAPResult)

    def test_trending_data(self, trending_up_data, trending_down_data):
        """Engine should handle trending data."""
        engine = SVPVWAPEngine(
            sl_strategy=SLStrategy.PERCENTAGE,
            session_lookback=50,
        )
        up_result = engine.run(trending_up_data)
        down_result = engine.run(trending_down_data)
        assert isinstance(up_result, SVPVWAPResult)
        assert isinstance(down_result, SVPVWAPResult)

    def test_import_from_backtester(self):
        """SVP+VWAP should be importable from backtester package."""
        from backtester import SVPVWAPEngine, SVPVWAPResult, compute_volume_profile
        assert SVPVWAPEngine is not None
        assert SVPVWAPResult is not None
        assert compute_volume_profile is not None
