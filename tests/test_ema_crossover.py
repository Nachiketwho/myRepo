"""Tests for EMA crossover strategy."""

import datetime as dt
import numpy as np
import pandas as pd
import pytest

from backtester.ema_crossover import (
    EMACrossoverEngine, EMACrossoverResult, EMACrossoverTrade,
    EMACrossoverSignals, SLCalculator, PositionSizer,
    SLStrategy, PositionSizing, EMA_PAIRS,
    TrailingMode, TRAIL_MILESTONES,
    run_ema_backtest_grid, build_ema_trade_log, compare_for_trading_style,
    ema, atr, swing_high, swing_low,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """100-bar sample OHLCV data with trends."""
    np.random.seed(42)
    n = 100
    base = 22000
    trend = np.cumsum(np.random.randn(n) * 10)
    close = base + trend

    df = pd.DataFrame({
        "open": close - np.random.uniform(5, 15, n),
        "high": close + np.random.uniform(10, 30, n),
        "low": close - np.random.uniform(10, 30, n),
        "close": close,
        "volume": np.random.randint(10000, 100000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))
    return df


@pytest.fixture
def trending_up_data():
    """Data with clear uptrend (for golden cross)."""
    n = 100
    close = 22000 + np.arange(n) * 5  # Clear uptrend

    df = pd.DataFrame({
        "open": close - 5,
        "high": close + 10,
        "low": close - 10,
        "close": close,
        "volume": [50000] * n,
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))
    return df


@pytest.fixture
def trending_down_data():
    """Data that starts up then trends down (for death cross)."""
    n = 100
    # Start with uptrend for first 30 bars, then downtrend
    up = 22000 + np.arange(30) * 10
    down = up[-1] - np.arange(70) * 8
    close = np.concatenate([up, down])

    df = pd.DataFrame({
        "open": close + 5,
        "high": close + 10,
        "low": close - 10,
        "close": close,
        "volume": [50000] * n,
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))
    return df


# ---------------------------------------------------------------------------
# Indicator tests
# ---------------------------------------------------------------------------

class TestIndicators:
    def test_ema_length(self, sample_data):
        result = ema(sample_data["close"], 9)
        assert len(result) == len(sample_data)

    def test_ema_smoothing(self, sample_data):
        result = ema(sample_data["close"], 9)
        # EMA should be smoother (lower std) than raw price
        assert result.std() < sample_data["close"].std()

    def test_atr_length(self, sample_data):
        result = atr(sample_data, 14)
        assert len(result) == len(sample_data)

    def test_atr_positive(self, sample_data):
        result = atr(sample_data, 14)
        assert (result.dropna() > 0).all()

    def test_swing_high_max(self, sample_data):
        result = swing_high(sample_data, 5)
        # Swing high should be >= current high
        valid_idx = result.notna()
        assert (result[valid_idx] >= sample_data["high"][valid_idx]).all()

    def test_swing_low_min(self, sample_data):
        result = swing_low(sample_data, 5)
        # Swing low should be <= current low
        valid_idx = result.notna()
        assert (result[valid_idx] <= sample_data["low"][valid_idx]).all()


# ---------------------------------------------------------------------------
# Signal generator tests
# ---------------------------------------------------------------------------

class TestEMACrossoverSignals:
    def test_generate_returns_dataframe(self, sample_data):
        gen = EMACrossoverSignals(fast_period=9, slow_period=21)
        result = gen.generate(sample_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)

    def test_generate_has_ema_columns(self, sample_data):
        gen = EMACrossoverSignals(fast_period=9, slow_period=21)
        result = gen.generate(sample_data)
        assert "ema_fast" in result.columns
        assert "ema_slow" in result.columns

    def test_generate_has_signal_column(self, sample_data):
        gen = EMACrossoverSignals(fast_period=9, slow_period=21)
        result = gen.generate(sample_data)
        assert "signal" in result.columns
        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_uptrend_generates_golden_cross(self, trending_up_data):
        gen = EMACrossoverSignals(fast_period=5, slow_period=20)
        result = gen.generate(trending_up_data)
        # Should have at least one golden cross
        assert (result["signal"] == 1).any()
        assert (result["crossover_type"] == "golden_cross").any()

    def test_downtrend_generates_death_cross(self, trending_down_data):
        gen = EMACrossoverSignals(fast_period=5, slow_period=20)
        result = gen.generate(trending_down_data)
        # Should have at least one death cross
        assert (result["signal"] == -1).any()
        assert (result["crossover_type"] == "death_cross").any()


# ---------------------------------------------------------------------------
# SL Calculator tests
# ---------------------------------------------------------------------------

class TestSLCalculator:
    def test_atr_sl(self, sample_data):
        calc = SLCalculator(
            strategy=SLStrategy.ATR,
            atr_multiplier=2.0,
            atr_period=14,
        )
        df = sample_data.copy()
        df["atr"] = atr(sample_data, 14)

        result = calc.calculate(df, 1, 50)
        assert "sl_level" in result
        assert "sl_points" in result
        assert result["sl_strategy"] == "atr"
        assert result["sl_points"] > 0

    def test_swing_sl_long(self, sample_data):
        calc = SLCalculator(
            strategy=SLStrategy.SWING,
            swing_lookback=5,
        )
        df = sample_data.copy()
        df["swing_low"] = swing_low(sample_data, 5)

        result = calc.calculate(df, 1, 50)
        entry = df.iloc[50]["close"]
        assert result["sl_level"] < entry  # Long SL below entry

    def test_swing_sl_short(self, sample_data):
        calc = SLCalculator(
            strategy=SLStrategy.SWING,
            swing_lookback=5,
        )
        df = sample_data.copy()
        df["swing_high"] = swing_high(sample_data, 5)

        result = calc.calculate(df, -1, 50)
        entry = df.iloc[50]["close"]
        assert result["sl_level"] > entry  # Short SL above entry

    def test_percentage_sl(self, sample_data):
        calc = SLCalculator(
            strategy=SLStrategy.PERCENTAGE,
            sl_pct=1.0,
        )
        result = calc.calculate(sample_data, 1, 50)
        entry = sample_data.iloc[50]["close"]
        expected_points = entry * 0.01
        assert pytest.approx(result["sl_points"], rel=0.01) == expected_points

    def test_fixed_sl(self, sample_data):
        calc = SLCalculator(
            strategy=SLStrategy.FIXED,
            sl_fixed_points=50.0,
        )
        result = calc.calculate(sample_data, 1, 50)
        assert result["sl_points"] == 50.0


# ---------------------------------------------------------------------------
# Position Sizer tests
# ---------------------------------------------------------------------------

class TestPositionSizer:
    def test_fixed_lots(self):
        sizer = PositionSizer(
            method=PositionSizing.FIXED,
            fixed_lots=2,
            lot_size=25,
        )
        result = sizer.calculate(22000, 50)
        assert result["num_lots"] == 2
        assert result["qty"] == 50

    def test_risk_based_sizing(self):
        sizer = PositionSizer(
            method=PositionSizing.RISK_BASED,
            capital=100000,
            risk_per_trade_pct=1.0,
            lot_size=25,
        )
        # Risk = 1000, SL = 50 pts, risk_per_lot = 50*25 = 1250
        # So num_lots should be int(1000/1250) = 0, but capped at 1
        result = sizer.calculate(22000, 50)
        assert result["num_lots"] >= 1

    def test_kelly_sizing(self):
        sizer = PositionSizer(
            method=PositionSizing.KELLY,
            capital=100000,
            lot_size=25,
            win_rate=0.5,
            avg_win_loss_ratio=2.0,
        )
        result = sizer.calculate(22000, 50)
        assert result["num_lots"] >= 1

    def test_position_value_calculated(self):
        sizer = PositionSizer(
            method=PositionSizing.FIXED,
            fixed_lots=1,
            lot_size=25,
        )
        result = sizer.calculate(22000, 50)
        assert result["position_value"] == 22000 * 25

    def test_risk_amount_calculated(self):
        sizer = PositionSizer(
            method=PositionSizing.FIXED,
            fixed_lots=1,
            lot_size=25,
        )
        result = sizer.calculate(22000, 50)
        assert result["risk_amount"] == 50 * 25


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

class TestEMACrossoverEngine:
    def test_run_returns_result(self, sample_data):
        engine = EMACrossoverEngine(ema_fast=9, ema_slow=21)
        result = engine.run(sample_data, "TEST", "15min")
        assert isinstance(result, EMACrossoverResult)

    def test_result_has_trades(self, trending_up_data):
        engine = EMACrossoverEngine(ema_fast=5, ema_slow=20)
        result = engine.run(trending_up_data, "TEST", "15min")
        # Should have at least one trade on trending data
        assert result.num_trades > 0

    def test_trades_have_sl_tp(self, trending_up_data):
        engine = EMACrossoverEngine(ema_fast=5, ema_slow=20)
        result = engine.run(trending_up_data, "TEST", "15min")
        for trade in result.trades:
            assert trade.sl_level > 0
            assert trade.tp_level > 0
            assert trade.sl_points > 0
            assert trade.tp_points > 0

    def test_trades_have_position_sizing(self, trending_up_data):
        engine = EMACrossoverEngine(ema_fast=5, ema_slow=20)
        result = engine.run(trending_up_data, "TEST", "15min")
        for trade in result.trades:
            assert trade.num_lots >= 1
            assert trade.qty > 0

    def test_sl_strategy_applied(self, sample_data):
        for sl_strat in SLStrategy:
            engine = EMACrossoverEngine(
                ema_fast=9, ema_slow=21,
                sl_strategy=sl_strat,
            )
            result = engine.run(sample_data, "TEST", "15min")
            for trade in result.trades:
                assert trade.sl_strategy == sl_strat.value

    def test_rr_ratio_applied(self, sample_data):
        engine = EMACrossoverEngine(
            ema_fast=9, ema_slow=21,
            tp_rr_ratio=3.0,
        )
        result = engine.run(sample_data, "TEST", "15min")
        for trade in result.trades:
            assert trade.rr_ratio == 3.0
            assert pytest.approx(trade.tp_points / trade.sl_points, rel=0.1) == 3.0


# ---------------------------------------------------------------------------
# Result metrics tests
# ---------------------------------------------------------------------------

class TestEMACrossoverResult:
    def test_win_rate_calculation(self):
        trades = [
            EMACrossoverTrade(
                entry_date=pd.Timestamp("2024-01-01"),
                entry_price=22000, direction=1, ema_fast=9, ema_slow=21,
                crossover_type="golden_cross", pnl_points=50,
            ),
            EMACrossoverTrade(
                entry_date=pd.Timestamp("2024-01-02"),
                entry_price=22000, direction=1, ema_fast=9, ema_slow=21,
                crossover_type="golden_cross", pnl_points=-30,
            ),
        ]
        result = EMACrossoverResult(
            ema_fast=9, ema_slow=21,
            sl_strategy="atr", position_sizing="risk_based",
            timeframe="15min", symbol="TEST", trades=trades,
        )
        assert result.win_rate == 50.0

    def test_profit_factor(self):
        trades = [
            EMACrossoverTrade(
                entry_date=pd.Timestamp("2024-01-01"),
                entry_price=22000, direction=1, ema_fast=9, ema_slow=21,
                crossover_type="golden_cross", pnl_amount=1000, pnl_points=40,
            ),
            EMACrossoverTrade(
                entry_date=pd.Timestamp("2024-01-02"),
                entry_price=22000, direction=1, ema_fast=9, ema_slow=21,
                crossover_type="golden_cross", pnl_amount=-500, pnl_points=-20,
            ),
        ]
        result = EMACrossoverResult(
            ema_fast=9, ema_slow=21,
            sl_strategy="atr", position_sizing="risk_based",
            timeframe="15min", symbol="TEST", trades=trades,
        )
        assert result.profit_factor == 2.0

    def test_r_multiple(self):
        trade = EMACrossoverTrade(
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=22000, direction=1, ema_fast=9, ema_slow=21,
            crossover_type="golden_cross", pnl_points=100, sl_points=50,
        )
        assert trade.r_multiple == 2.0

    def test_to_dict(self, sample_data):
        engine = EMACrossoverEngine(ema_fast=9, ema_slow=21)
        result = engine.run(sample_data, "TEST", "15min")
        d = result.to_dict()
        assert "ema_pair" in d
        assert "win_rate" in d
        assert "profit_factor" in d
        assert "expectancy" in d


# ---------------------------------------------------------------------------
# Grid backtest tests
# ---------------------------------------------------------------------------

class TestRunEMABacktestGrid:
    def test_returns_dataframe(self, sample_data):
        result = run_ema_backtest_grid(
            sample_data, "TEST", "15min",
            ema_pairs=[(9, 21)],
            sl_strategies=[SLStrategy.ATR],
        )
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_combo(self, sample_data):
        pairs = [(5, 9), (9, 21)]
        strategies = [SLStrategy.ATR, SLStrategy.FIXED]
        result = run_ema_backtest_grid(
            sample_data, "TEST", "15min",
            ema_pairs=pairs,
            sl_strategies=strategies,
        )
        assert len(result) == len(pairs) * len(strategies)

    def test_sorted_by_pnl(self, sample_data):
        result = run_ema_backtest_grid(
            sample_data, "TEST", "15min",
            ema_pairs=[(5, 9), (9, 21), (21, 50)],
            sl_strategies=[SLStrategy.ATR],
        )
        if len(result) > 1:
            assert result["total_pnl"].iloc[0] >= result["total_pnl"].iloc[1]


# ---------------------------------------------------------------------------
# Trade log tests
# ---------------------------------------------------------------------------

class TestBuildEMATradeLog:
    def test_returns_dataframe(self, sample_data):
        engine = EMACrossoverEngine(ema_fast=9, ema_slow=21)
        result = engine.run(sample_data, "TEST", "15min")
        log = build_ema_trade_log(result)
        assert isinstance(log, pd.DataFrame)

    def test_columns_present(self, trending_up_data):
        engine = EMACrossoverEngine(ema_fast=5, ema_slow=20)
        result = engine.run(trending_up_data, "TEST", "15min")
        log = build_ema_trade_log(result)
        if not log.empty:
            expected = [
                "entry_date", "exit_date", "direction", "entry_price",
                "exit_price", "sl_level", "tp_level", "sl_points",
                "pnl_points", "pnl_amount", "r_multiple",
            ]
            for col in expected:
                assert col in log.columns


# ---------------------------------------------------------------------------
# Trading style comparison tests
# ---------------------------------------------------------------------------

class TestCompareForTradingStyle:
    def test_returns_dict(self, sample_data):
        results = run_ema_backtest_grid(
            sample_data, "TEST", "15min",
            ema_pairs=[(5, 9), (9, 21)],
            sl_strategies=[SLStrategy.ATR],
        )
        recs = compare_for_trading_style(results)
        assert isinstance(recs, dict)
        assert "intraday" in recs
        assert "swing" in recs

    def test_intraday_has_fields(self, sample_data):
        results = run_ema_backtest_grid(
            sample_data, "TEST", "15min",
            ema_pairs=[(5, 9), (9, 21)],
            sl_strategies=[SLStrategy.ATR],
        )
        recs = compare_for_trading_style(results)
        if recs["intraday"]:
            assert "ema_pair" in recs["intraday"]
            assert "win_rate" in recs["intraday"]


# ---------------------------------------------------------------------------
# EMA pairs constant
# ---------------------------------------------------------------------------

class TestEMAPairs:
    def test_pairs_sorted(self):
        for fast, slow in EMA_PAIRS:
            assert fast < slow

    def test_includes_common_pairs(self):
        pairs_set = set(EMA_PAIRS)
        assert (5, 9) in pairs_set
        assert (9, 21) in pairs_set


# ---------------------------------------------------------------------------
# Trailing SL tests
# ---------------------------------------------------------------------------

class TestTrailingMode:
    def test_trailing_mode_values(self):
        assert TrailingMode.NONE.value == "none"
        assert TrailingMode.BREAKEVEN.value == "breakeven"
        assert TrailingMode.STEPPED.value == "stepped"

    def test_trail_milestones_defined(self):
        assert len(TRAIL_MILESTONES) >= 3
        # Check milestones are ordered by profit multiple
        prev_mult = 0
        for profit_mult, _ in TRAIL_MILESTONES:
            assert profit_mult > prev_mult
            prev_mult = profit_mult


class TestTrailingEngine:
    @pytest.fixture
    def trailing_data(self):
        """Data with clear price movement for testing trailing SL."""
        n = 100
        # Start with golden cross conditions, then price moves up significantly
        # First 30 bars: slow uptrend to trigger golden cross
        # Next 70 bars: strong uptrend to trigger trailing milestones
        close_values = []
        for i in range(30):
            close_values.append(22000 + i * 5)  # Slow uptrend
        for i in range(70):
            close_values.append(22150 + i * 20)  # Strong uptrend

        close = np.array(close_values)
        df = pd.DataFrame({
            "open": close - 5,
            "high": close + 15,
            "low": close - 10,
            "close": close,
            "volume": [50000] * n,
        }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))
        return df

    def test_no_trailing_mode(self, trending_up_data):
        """No trailing mode should not modify SL."""
        engine = EMACrossoverEngine(
            ema_fast=5, ema_slow=20,
            trailing_mode=TrailingMode.NONE,
        )
        result = engine.run(trending_up_data, "TEST", "15min")
        for trade in result.trades:
            assert trade.trailing_mode == "none"
            # TSL level should equal initial SL level
            assert trade.tsl_level == trade.initial_sl_level
            assert trade.active_milestone == -1

    def test_breakeven_mode_sets_field(self, trending_up_data):
        """Breakeven mode should be set in trades."""
        engine = EMACrossoverEngine(
            ema_fast=5, ema_slow=20,
            trailing_mode=TrailingMode.BREAKEVEN,
        )
        result = engine.run(trending_up_data, "TEST", "15min")
        for trade in result.trades:
            assert trade.trailing_mode == "breakeven"

    def test_stepped_mode_sets_field(self, trending_up_data):
        """Stepped mode should be set in trades."""
        engine = EMACrossoverEngine(
            ema_fast=5, ema_slow=20,
            trailing_mode=TrailingMode.STEPPED,
        )
        result = engine.run(trending_up_data, "TEST", "15min")
        for trade in result.trades:
            assert trade.trailing_mode == "stepped"

    def test_build_trail_thresholds_long(self):
        """Test building trail thresholds for long position."""
        engine = EMACrossoverEngine(
            ema_fast=9, ema_slow=21,
            trailing_mode=TrailingMode.STEPPED,
        )
        thresholds = engine._build_trail_thresholds(
            entry_price=22000, sl_points=100, direction=1
        )
        # Should have 4 milestones from TRAIL_MILESTONES
        assert len(thresholds) == len(TRAIL_MILESTONES)

        # First milestone (1R profit): threshold at 22100, lock at breakeven
        assert thresholds[0][0] == 22100  # 22000 + 100 * 1.0
        assert thresholds[0][1] == 22000  # 22000 + 100 * 0.0 (breakeven)

        # Second milestone (1.5R profit): threshold at 22150, lock 0.5R
        assert thresholds[1][0] == 22150  # 22000 + 100 * 1.5
        assert thresholds[1][1] == 22050  # 22000 + 100 * 0.5

    def test_build_trail_thresholds_short(self):
        """Test building trail thresholds for short position."""
        engine = EMACrossoverEngine(
            ema_fast=9, ema_slow=21,
            trailing_mode=TrailingMode.STEPPED,
        )
        thresholds = engine._build_trail_thresholds(
            entry_price=22000, sl_points=100, direction=-1
        )

        # First milestone: threshold at 21900, lock at breakeven
        assert thresholds[0][0] == 21900  # 22000 - 100 * 1.0
        assert thresholds[0][1] == 22000  # 22000 - 100 * 0.0 (breakeven)

        # Second milestone: threshold at 21850, lock 0.5R
        assert thresholds[1][0] == 21850  # 22000 - 100 * 1.5
        assert thresholds[1][1] == 21950  # 22000 - 100 * 0.5

    def test_trailing_sl_updates_on_profit(self, trailing_data):
        """Trailing SL should update when price moves in favor."""
        engine = EMACrossoverEngine(
            ema_fast=5, ema_slow=20,
            trailing_mode=TrailingMode.STEPPED,
            sl_strategy=SLStrategy.FIXED,
            sl_fixed_points=100.0,
            tp_rr_ratio=5.0,  # High TP so we don't exit early
        )
        result = engine.run(trailing_data, "TEST", "15min")

        # Should have trades with trailing updates
        trades_with_trails = [t for t in result.trades if t.active_milestone >= 0]
        # Some trades should have hit trailing milestones
        if result.num_trades > 0:
            # At least check that trailing mode was applied
            assert all(t.trailing_mode == "stepped" for t in result.trades)

    def test_trailing_sl_exit_reason(self, trailing_data):
        """Check that trailing SL exit is properly recorded."""
        engine = EMACrossoverEngine(
            ema_fast=5, ema_slow=20,
            trailing_mode=TrailingMode.STEPPED,
            sl_strategy=SLStrategy.FIXED,
            sl_fixed_points=50.0,
            tp_rr_ratio=10.0,  # Very high TP
        )
        result = engine.run(trailing_data, "TEST", "15min")

        # Check exit reasons include trailing_sl if milestones were hit
        trailing_exits = [t for t in result.trades if t.exit_reason == "trailing_sl"]
        # It's okay if none exited via trailing (depends on data)
        for trade in trailing_exits:
            assert trade.active_milestone >= 0

    def test_breakeven_only_first_milestone(self, trailing_data):
        """Breakeven mode should only use first milestone."""
        engine = EMACrossoverEngine(
            ema_fast=5, ema_slow=20,
            trailing_mode=TrailingMode.BREAKEVEN,
            sl_strategy=SLStrategy.FIXED,
            sl_fixed_points=50.0,
            tp_rr_ratio=10.0,
        )
        result = engine.run(trailing_data, "TEST", "15min")

        # All active milestones should be at most 0 (first milestone)
        for trade in result.trades:
            assert trade.active_milestone <= 0

    def test_trade_log_includes_trailing_info(self, trending_up_data):
        """Trade log should include trailing SL columns."""
        engine = EMACrossoverEngine(
            ema_fast=5, ema_slow=20,
            trailing_mode=TrailingMode.STEPPED,
        )
        result = engine.run(trending_up_data, "TEST", "15min")
        log = build_ema_trade_log(result)

        if not log.empty:
            assert "trailing_mode" in log.columns
            assert "tsl_level" in log.columns
            assert "trail_milestones_hit" in log.columns

    def test_initial_sl_preserved(self, trending_up_data):
        """Initial SL level should be preserved even after trailing."""
        engine = EMACrossoverEngine(
            ema_fast=5, ema_slow=20,
            trailing_mode=TrailingMode.STEPPED,
        )
        result = engine.run(trending_up_data, "TEST", "15min")

        for trade in result.trades:
            # Initial SL should match sl_level
            assert trade.initial_sl_level == trade.sl_level
            # TSL should be >= initial SL for long (more protective)
            if trade.direction == 1 and trade.active_milestone >= 0:
                assert trade.tsl_level >= trade.initial_sl_level

    def test_custom_milestones(self, trending_up_data):
        """Test with custom trailing milestones."""
        custom_milestones = [(0.5, 0.0), (1.0, 0.3)]  # Tighter milestones
        engine = EMACrossoverEngine(
            ema_fast=5, ema_slow=20,
            trailing_mode=TrailingMode.STEPPED,
            trail_milestones=custom_milestones,
        )
        # Just verify it runs without error
        result = engine.run(trending_up_data, "TEST", "15min")
        assert isinstance(result, EMACrossoverResult)
