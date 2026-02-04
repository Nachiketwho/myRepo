"""Tests for F&O backtesting engine."""

import datetime as dt
import numpy as np
import pandas as pd
import pytest

from backtester.fno_engine import (
    FnOTrade, FnOResult, FnOEngine, FnOExitReason,
    FNO_DEFAULTS, ADAPTIVE_SL_MAP, TRAIL_MILESTONES,
    run_fno_analysis, build_fno_trade_log,
    _indicator_flags, _capture_indicators,
)
from backtester.greeks import bs_price
from backtester.strikes import LOT_SIZE, STRIKE_INTERVAL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vix_series():
    """VIX series at 13% for 60 days."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    return pd.Series(13.0, index=dates.date)


@pytest.fixture
def signals_buy_only():
    """20-bar data with a single buy signal on bar 5."""
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    df = pd.DataFrame({
        "open":   [22000 + i * 10 for i in range(20)],
        "high":   [22010 + i * 10 for i in range(20)],
        "low":    [21990 + i * 10 for i in range(20)],
        "close":  [22000 + i * 10 for i in range(20)],
        "volume": [1000] * 20,
        "signal": [0] * 20,
    }, index=dates)
    df.iloc[5, df.columns.get_loc("signal")] = 1
    return df


@pytest.fixture
def signals_sell_only():
    """20-bar data with a single sell signal on bar 5."""
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    df = pd.DataFrame({
        "open":   [22000 - i * 10 for i in range(20)],
        "high":   [22010 - i * 10 for i in range(20)],
        "low":    [21990 - i * 10 for i in range(20)],
        "close":  [22000 - i * 10 for i in range(20)],
        "volume": [1000] * 20,
        "signal": [0] * 20,
    }, index=dates)
    df.iloc[5, df.columns.get_loc("signal")] = -1
    return df


@pytest.fixture
def signals_no_signals():
    """20-bar data with no signals at all."""
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    df = pd.DataFrame({
        "open":   [22000] * 20,
        "high":   [22050] * 20,
        "low":    [21950] * 20,
        "close":  [22000] * 20,
        "volume": [1000] * 20,
        "signal": [0] * 20,
    }, index=dates)
    return df


def _make_row(close=22000, high=22050, low=21950, signal=0, **extra):
    """Helper to build a pd.Series mimicking a signals_df row."""
    data = {
        "open": close, "high": high, "low": low,
        "close": close, "volume": 1000, "signal": signal,
    }
    data.update(extra)
    return pd.Series(data)


# ---------------------------------------------------------------------------
# FnOTrade dataclass
# ---------------------------------------------------------------------------

class TestFnOTrade:
    def test_pnl_per_lot(self):
        trade = FnOTrade(
            entry_date=dt.date(2024, 1, 1), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=1,
            greeks_entry={}, premium_exit=250.0,
        )
        assert trade.pnl_per_lot == 50.0

    def test_pnl_gross(self):
        trade = FnOTrade(
            entry_date=dt.date(2024, 1, 1), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=2,
            greeks_entry={}, premium_exit=250.0,
        )
        assert trade.pnl_gross == 50.0 * 25 * 2

    def test_pnl_net_deducts_txn_cost(self):
        trade = FnOTrade(
            entry_date=dt.date(2024, 1, 1), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=1,
            greeks_entry={}, premium_exit=250.0, txn_cost=100.0,
        )
        assert trade.pnl == trade.pnl_gross - 100.0

    def test_pnl_pct(self):
        trade = FnOTrade(
            entry_date=dt.date(2024, 1, 1), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=1,
            greeks_entry={}, premium_exit=250.0,
        )
        assert pytest.approx(trade.pnl_pct) == 25.0  # 50/200 * 100

    def test_pnl_pct_zero_entry(self):
        trade = FnOTrade(
            entry_date=dt.date(2024, 1, 1), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=0.0, lot_size=25, num_lots=1,
            greeks_entry={}, premium_exit=10.0,
        )
        assert trade.pnl_pct == 0.0

    def test_hold_days(self):
        trade = FnOTrade(
            entry_date=pd.Timestamp("2024-01-01"), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=1,
            greeks_entry={},
            exit_date=pd.Timestamp("2024-01-05"),
        )
        assert trade.hold_days == 4

    def test_hold_days_none_when_no_exit(self):
        trade = FnOTrade(
            entry_date=pd.Timestamp("2024-01-01"), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=1,
            greeks_entry={},
        )
        assert trade.hold_days is None

    def test_contract_label(self):
        trade = FnOTrade(
            entry_date=dt.date(2024, 1, 1), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=1,
            greeks_entry={},
        )
        assert "22000CE" in trade.contract_label
        assert "ATM" in trade.contract_label

    def test_strike_label_otm(self):
        trade = FnOTrade(
            entry_date=dt.date(2024, 1, 1), entry_spot=22000,
            strike=22100, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=150.0, lot_size=25, num_lots=1,
            greeks_entry={},
        )
        assert trade.strike_label == "2-OTM"

    def test_losing_trade(self):
        trade = FnOTrade(
            entry_date=dt.date(2024, 1, 1), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=1,
            greeks_entry={}, premium_exit=150.0,
        )
        assert trade.pnl_per_lot == -50.0
        assert trade.pnl_gross < 0

    def test_risk_level_fields(self):
        trade = FnOTrade(
            entry_date=dt.date(2024, 1, 1), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=1,
            greeks_entry={},
            sl_points=30.0, tp_points=60.0, rr_ratio=2.0,
            sl_level=170.0, tp_level=260.0, breakeven_level=202.5,
        )
        assert trade.sl_points == 30.0
        assert trade.tp_points == 60.0
        assert trade.rr_ratio == 2.0
        assert trade.sl_level == 170.0
        assert trade.tp_level == 260.0

    def test_indicator_flag_fields(self):
        trade = FnOTrade(
            entry_date=dt.date(2024, 1, 1), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=1,
            greeks_entry={},
            inner_band_touch=True, outer_band_touch=False,
            obv_confirm=True, ad_confirm=False, signal_strength=2,
        )
        assert trade.inner_band_touch is True
        assert trade.outer_band_touch is False
        assert trade.signal_strength == 2


# ---------------------------------------------------------------------------
# FnOResult
# ---------------------------------------------------------------------------

class TestFnOResult:
    def _make_result(self, pnls):
        trades = []
        for pnl_val in pnls:
            t = FnOTrade(
                entry_date=pd.Timestamp("2024-01-01"), entry_spot=22000,
                strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
                premium_entry=200.0, lot_size=25, num_lots=1,
                greeks_entry={},
                exit_date=pd.Timestamp("2024-01-05"),
                premium_exit=200.0 + pnl_val,
                exit_reason=FnOExitReason.TP_POINTS if pnl_val > 0 else FnOExitReason.SL_POINTS,
            )
            trades.append(t)
        return FnOResult(trades=trades, strike=0, option_type_used="CE", expiry_week=1)

    def test_num_trades(self):
        result = self._make_result([50, -30, 20])
        assert result.num_trades == 3

    def test_total_pnl(self):
        result = self._make_result([50, -30, 20])
        expected = (50 + (-30) + 20) * 25 * 1  # pnl_per_lot * lot_size * num_lots
        assert pytest.approx(result.total_pnl) == expected

    def test_win_rate(self):
        result = self._make_result([50, -30, 20])
        assert pytest.approx(result.win_rate, abs=0.1) == 66.7

    def test_avg_pnl(self):
        result = self._make_result([50, -30, 20])
        assert result.avg_pnl != 0

    def test_avg_hold_days(self):
        result = self._make_result([50])
        assert result.avg_hold_days == 4.0

    def test_exit_reasons(self):
        result = self._make_result([50, -30])
        reasons = result.exit_reasons
        assert reasons.get("tp_points", 0) == 1
        assert reasons.get("sl_points", 0) == 1

    def test_empty_result(self):
        result = FnOResult(trades=[], strike=0, option_type_used="CE", expiry_week=1)
        assert result.num_trades == 0
        assert result.total_pnl == 0
        assert result.win_rate == 0.0
        assert result.avg_pnl == 0.0
        assert result.avg_hold_days == 0.0


# ---------------------------------------------------------------------------
# FnOEngine
# ---------------------------------------------------------------------------

class TestFnOEngine:
    def test_buy_signal_creates_ce(self, signals_buy_only, vix_series):
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, expiry_week=2)
        assert result.num_trades >= 1
        assert result.trades[0].option_type == "CE"

    def test_sell_signal_creates_pe(self, signals_sell_only, vix_series):
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_sell_only, vix_series, expiry_week=2)
        assert result.num_trades >= 1
        assert result.trades[0].option_type == "PE"

    def test_no_signals_no_trades(self, signals_no_signals, vix_series):
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_no_signals, vix_series)
        assert result.num_trades == 0

    def test_strike_offset_atm(self, signals_buy_only, vix_series):
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, strike_offset=0, expiry_week=2)
        if result.trades:
            trade = result.trades[0]
            expected_strike = round(trade.entry_spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
            assert trade.strike == expected_strike

    def test_strike_offset_otm_ce(self, signals_buy_only, vix_series):
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, strike_offset=2, expiry_week=2)
        if result.trades:
            trade = result.trades[0]
            atm = round(trade.entry_spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
            assert trade.strike == atm + 2 * STRIKE_INTERVAL

    def test_strike_offset_otm_pe(self, signals_sell_only, vix_series):
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_sell_only, vix_series, strike_offset=2, expiry_week=2)
        if result.trades:
            trade = result.trades[0]
            atm = round(trade.entry_spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
            assert trade.strike == atm - 2 * STRIKE_INTERVAL

    def test_expiry_week_selection(self, signals_buy_only, vix_series):
        engine = FnOEngine(min_signal_strength=0)
        r1 = engine.run(signals_buy_only, vix_series, expiry_week=2)
        r2 = engine.run(signals_buy_only, vix_series, expiry_week=3)
        if r1.trades and r2.trades:
            assert r2.trades[0].expiry > r1.trades[0].expiry

    def test_end_of_data_exit(self, vix_series):
        """Open position at end of data should be closed with END_OF_DATA."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({
            "open":   [22000] * 5,
            "high":   [22050] * 5,
            "low":    [21950] * 5,
            "close":  [22000] * 5,
            "volume": [1000] * 5,
            "signal": [1, 0, 0, 0, 0],
        }, index=dates)
        engine = FnOEngine(sl_points=9999, tp_points=9999, min_signal_strength=0)
        result = engine.run(df, vix_series, expiry_week=4)
        assert result.num_trades >= 1
        last_trade = result.trades[-1]
        assert last_trade.exit_reason == FnOExitReason.END_OF_DATA

    def test_signal_reversal_exit(self, vix_series):
        """Buy CE on bar 3, sell signal on bar 10 -> reversal exit."""
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        df = pd.DataFrame({
            "open":   [22000] * 20,
            "high":   [22050] * 20,
            "low":    [21950] * 20,
            "close":  [22000] * 20,
            "volume": [1000] * 20,
            "signal": [0] * 20,
        }, index=dates)
        df.iloc[3, df.columns.get_loc("signal")] = 1   # buy CE
        df.iloc[10, df.columns.get_loc("signal")] = -1  # sell -> reversal
        engine = FnOEngine(sl_points=9999, tp_points=9999, min_signal_strength=0)
        result = engine.run(df, vix_series, expiry_week=4)
        has_reversal = any(
            t.exit_reason == FnOExitReason.SIGNAL_REVERSAL
            for t in result.trades
        )
        assert has_reversal

    def test_greeks_populated(self, signals_buy_only, vix_series):
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, expiry_week=2)
        if result.trades:
            trade = result.trades[0]
            assert "delta" in trade.greeks_entry
            assert "theta" in trade.greeks_entry
            assert "vega" in trade.greeks_entry
            assert "iv" in trade.greeks_entry

    def test_premium_entry_positive(self, signals_buy_only, vix_series):
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, expiry_week=2)
        for trade in result.trades:
            assert trade.premium_entry > 0

    def test_custom_lot_size(self, signals_buy_only, vix_series):
        engine = FnOEngine(lot_size=50, num_lots=2, min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, expiry_week=2)
        if result.trades:
            assert result.trades[0].lot_size == 50
            assert result.trades[0].num_lots == 2

    def test_result_params_stored(self, signals_buy_only, vix_series):
        engine = FnOEngine(sl_points=40, tp_points=60, min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, strike_offset=1, expiry_week=2)
        assert result.params["sl_points"] == 40
        assert result.params["tp_points"] == 60
        assert result.params["strike_offset"] == 1
        assert result.params["expiry_week"] == 2

    def test_multiple_trades(self, vix_series):
        """Multiple signals -> multiple trades."""
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        df = pd.DataFrame({
            "open":   [22000] * 40,
            "high":   [22050] * 40,
            "low":    [21950] * 40,
            "close":  [22000] * 40,
            "volume": [1000] * 40,
            "signal": [0] * 40,
        }, index=dates)
        df.iloc[2, df.columns.get_loc("signal")] = 1
        df.iloc[15, df.columns.get_loc("signal")] = -1
        df.iloc[30, df.columns.get_loc("signal")] = 1

        engine = FnOEngine(sl_points=9999, tp_points=9999, min_signal_strength=0)
        result = engine.run(df, vix_series, expiry_week=2)
        assert result.num_trades >= 2

    def test_min_signal_strength_filters(self, signals_buy_only, vix_series):
        """High min_signal_strength should filter out weak signals."""
        engine_strict = FnOEngine(min_signal_strength=4)
        result = engine_strict.run(signals_buy_only, vix_series, expiry_week=2)
        # Test data has no indicator columns -> strength=0 -> all filtered
        assert result.num_trades == 0

    def test_adaptive_sl_applied(self, signals_buy_only, vix_series):
        """Trade should have SL based on signal strength (0 -> 40pts default)."""
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, expiry_week=2)
        if result.trades:
            trade = result.trades[0]
            # Strength=0 for test data without indicator columns -> SL=40
            assert trade.sl_points == ADAPTIVE_SL_MAP.get(0, 30.0)

    def test_rr_based_tp(self, signals_buy_only, vix_series):
        """TP should be SL * rr_ratio."""
        engine = FnOEngine(min_signal_strength=0, rr_ratio=2.0)
        result = engine.run(signals_buy_only, vix_series, expiry_week=2)
        if result.trades:
            trade = result.trades[0]
            assert pytest.approx(trade.tp_points) == trade.sl_points * 2.0

    def test_backward_compat_fixed_sl_tp(self, signals_buy_only, vix_series):
        """sl_points/tp_points override adaptive SL and RR TP."""
        engine = FnOEngine(sl_points=45, tp_points=90, min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, expiry_week=2)
        if result.trades:
            trade = result.trades[0]
            assert trade.sl_points == 45
            assert trade.tp_points == 90

    def test_trend_fields_populated(self, signals_buy_only, vix_series):
        """Trend fields should be set on each trade."""
        engine = FnOEngine(min_signal_strength=0, ema_period=5)
        result = engine.run(signals_buy_only, vix_series, expiry_week=2)
        if result.trades:
            trade = result.trades[0]
            assert trade.ema_value > 0
            assert trade.trend_aligned in (True, False)

    def test_txn_cost_computed(self, signals_buy_only, vix_series):
        """Trades should have transaction costs computed."""
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, expiry_week=2)
        for trade in result.trades:
            assert trade.txn_cost >= 0

    def test_trade_quality_assigned(self, signals_buy_only, vix_series):
        """Trades should have quality classification."""
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, expiry_week=2)
        for trade in result.trades:
            assert trade.trade_quality in ("good", "neutral", "bad")
            assert len(trade.quality_reasons) > 0


# ---------------------------------------------------------------------------
# Adaptive SL / RR helpers
# ---------------------------------------------------------------------------

class TestAdaptiveSL:
    def test_get_sl_default_map(self):
        engine = FnOEngine()
        assert engine._get_sl(4) == 20.0
        assert engine._get_sl(3) == 25.0
        assert engine._get_sl(2) == 30.0
        assert engine._get_sl(1) == 35.0
        assert engine._get_sl(0) == 40.0

    def test_get_sl_custom_map(self):
        custom = {4: 10.0, 3: 15.0, 2: 20.0, 1: 25.0, 0: 30.0}
        engine = FnOEngine(sl_map=custom)
        assert engine._get_sl(4) == 10.0
        assert engine._get_sl(0) == 30.0

    def test_get_sl_fixed_override(self):
        engine = FnOEngine(sl_points=50)
        assert engine._get_sl(4) == 50
        assert engine._get_sl(0) == 50

    def test_get_tp_from_rr(self):
        engine = FnOEngine(rr_ratio=2.5)
        assert engine._get_tp(30.0) == 75.0

    def test_get_tp_fixed_override(self):
        engine = FnOEngine(tp_points=100)
        assert engine._get_tp(30.0) == 100

    def test_build_trail_milestones(self):
        engine = FnOEngine()
        ms = engine._build_trail_milestones(
            entry_premium=200.0, sl=30.0, be_points=2.5,
        )
        assert len(ms) == len(TRAIL_MILESTONES)
        # First milestone: 200 + 30*1.5 = 245, floor = 200 + 2.5 = 202.5
        assert ms[0] == (245.0, 202.5)
        # Second: 200 + 30*2.0 = 260, floor = 200 + 30*1.0 = 230
        assert ms[1] == (260.0, 230.0)
        # Third: 200 + 30*2.5 = 275, floor = 200 + 30*1.5 = 245
        assert ms[2] == (275.0, 245.0)


# ---------------------------------------------------------------------------
# IV lookup
# ---------------------------------------------------------------------------

class TestGetIV:
    def test_exact_date_match(self):
        dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
        vix = pd.Series([13.0, 14.0], index=dates)
        iv = FnOEngine._get_iv(vix, dt.date(2024, 1, 2))
        assert pytest.approx(iv) == 0.13

    def test_forward_fill(self):
        dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 4)]
        vix = pd.Series([13.0, 15.0], index=dates)
        # Jan 3 not in index -> forward fill from Jan 2
        iv = FnOEngine._get_iv(vix, dt.date(2024, 1, 3))
        assert pytest.approx(iv) == 0.13

    def test_empty_series_default(self):
        iv = FnOEngine._get_iv(pd.Series(dtype=float), dt.date(2024, 1, 1))
        assert iv == 0.15

    def test_none_series_default(self):
        iv = FnOEngine._get_iv(None, dt.date(2024, 1, 1))
        assert iv == 0.15

    def test_converts_percent_to_decimal(self):
        vix = pd.Series([20.0], index=[dt.date(2024, 1, 1)])
        iv = FnOEngine._get_iv(vix, dt.date(2024, 1, 1))
        assert pytest.approx(iv) == 0.20


# ---------------------------------------------------------------------------
# Exit conditions (using pd.Series rows)
# ---------------------------------------------------------------------------

class TestCheckExit:
    """Test _check_exit directly for specific exit scenarios."""

    def _make_position(self, premium_entry=200.0, option_type="CE",
                       expiry_days_ahead=7, sl_points=50.0, tp_points=100.0):
        current = dt.date(2024, 1, 10)
        expiry = current + dt.timedelta(days=expiry_days_ahead)
        return FnOTrade(
            entry_date=current, entry_spot=22000,
            strike=22000, option_type=option_type, expiry=expiry,
            premium_entry=premium_entry, lot_size=25, num_lots=1,
            greeks_entry={}, premium_high=premium_entry,
            sl_points=sl_points, tp_points=tp_points,
            sl_level=premium_entry - sl_points,
            tp_level=premium_entry + tp_points,
        )

    def test_expiry_eve_exit(self):
        """Position should exit when close to expiry."""
        pos = self._make_position(expiry_days_ahead=1)
        engine = FnOEngine(exit_before_expiry_days=1)
        row = _make_row(close=22000, high=22050, low=21950)
        _, reason = engine._check_exit(
            pos, row, dt.date(2024, 1, 10), 0.13, 0
        )
        assert reason == FnOExitReason.EXPIRY_EVE

    def test_sl_exit_intra_bar(self):
        """Premium drops below SL level intra-bar -> SL exit."""
        pos = self._make_position(premium_entry=200.0, expiry_days_ahead=14,
                                  sl_points=50.0)
        engine = FnOEngine(sl_points=50)
        # Big move down: low=21500 should drop CE premium well below SL
        row = _make_row(close=21500, high=22000, low=21500)
        _, reason = engine._check_exit(
            pos, row, dt.date(2024, 1, 11), 0.13, 0
        )
        assert reason == FnOExitReason.SL_POINTS

    def test_signal_reversal_ce_with_sell(self):
        """CE position + sell signal -> reversal exit."""
        pos = self._make_position(option_type="CE", expiry_days_ahead=14,
                                  sl_points=9999, tp_points=9999)
        pos.sl_level = pos.premium_entry - 9999
        pos.tp_level = pos.premium_entry + 9999
        engine = FnOEngine(sl_points=9999, tp_points=9999)
        row = _make_row(close=22000, high=22050, low=21950)
        _, reason = engine._check_exit(
            pos, row, dt.date(2024, 1, 11), 0.13, -1
        )
        assert reason == FnOExitReason.SIGNAL_REVERSAL

    def test_signal_reversal_pe_with_buy(self):
        """PE position + buy signal -> reversal exit."""
        pos = self._make_position(option_type="PE", expiry_days_ahead=14,
                                  sl_points=9999, tp_points=9999)
        pos.sl_level = pos.premium_entry - 9999
        pos.tp_level = pos.premium_entry + 9999
        engine = FnOEngine(sl_points=9999, tp_points=9999)
        row = _make_row(close=22000, high=22050, low=21950)
        _, reason = engine._check_exit(
            pos, row, dt.date(2024, 1, 11), 0.13, 1
        )
        assert reason == FnOExitReason.SIGNAL_REVERSAL

    def test_no_exit_when_within_bounds(self):
        """No exit when premium change is within SL/TP bounds."""
        pos = self._make_position(premium_entry=200.0, expiry_days_ahead=14,
                                  sl_points=500, tp_points=500)
        engine = FnOEngine(sl_points=500, tp_points=500)
        row = _make_row(close=22010, high=22020, low=22000)
        _, reason = engine._check_exit(
            pos, row, dt.date(2024, 1, 11), 0.13, 0
        )
        assert reason is None

    def test_same_direction_signal_no_exit(self):
        """CE position + buy signal -> no exit (same direction)."""
        pos = self._make_position(option_type="CE", expiry_days_ahead=14,
                                  sl_points=500, tp_points=500)
        engine = FnOEngine(sl_points=500, tp_points=500)
        row = _make_row(close=22000, high=22050, low=21950)
        _, reason = engine._check_exit(
            pos, row, dt.date(2024, 1, 11), 0.13, 1
        )
        assert reason is None


# ---------------------------------------------------------------------------
# Indicator flags
# ---------------------------------------------------------------------------

class TestIndicatorFlags:
    def test_no_signal_returns_zero_strength(self):
        row = _make_row(signal=0)
        df = pd.DataFrame([row])
        flags = _indicator_flags(row, df, 0)
        assert flags["strength"] == 0
        assert not flags["inner_band"]

    def test_buy_with_band_touch(self):
        row = _make_row(
            signal=1, low=21900,
            vwap_lower_inner=21950, vwap_lower_outer=21850,
        )
        df = pd.DataFrame([row])
        flags = _indicator_flags(row, df, 0)
        assert flags["inner_band"] is True
        assert flags["outer_band"] is False  # low > outer

    def test_sell_with_band_touch(self):
        row = _make_row(
            signal=-1, high=22100,
            vwap_upper_inner=22050, vwap_upper_outer=22150,
        )
        df = pd.DataFrame([row])
        flags = _indicator_flags(row, df, 0)
        assert flags["inner_band"] is True
        assert flags["outer_band"] is False


# ---------------------------------------------------------------------------
# Trade classification
# ---------------------------------------------------------------------------

class TestClassifyTrade:
    def test_profitable_strong_signal_good(self):
        trade = FnOTrade(
            entry_date=dt.date(2024, 1, 1), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=1,
            greeks_entry={}, premium_exit=260.0,
            exit_reason=FnOExitReason.TP_POINTS,
            signal_strength=3, trend_aligned=True,
            obv_confirm=True, ad_confirm=True,
        )
        quality, reasons = FnOEngine._classify_trade(trade)
        assert quality == "good"
        assert "profit_taken" in reasons
        assert "strong_signal" in reasons

    def test_sl_hit_weak_signal_bad(self):
        trade = FnOTrade(
            entry_date=dt.date(2024, 1, 1), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=1,
            greeks_entry={}, premium_exit=150.0,
            exit_reason=FnOExitReason.SL_POINTS,
            signal_strength=1, trend_aligned=False,
            obv_confirm=False, ad_confirm=False,
        )
        quality, reasons = FnOEngine._classify_trade(trade)
        assert quality == "bad"
        assert "sl_hit" in reasons
        assert "weak_signal" in reasons


# ---------------------------------------------------------------------------
# PnL decomposition
# ---------------------------------------------------------------------------

class TestDecomposePnl:
    def test_basic_decomposition(self):
        trade = FnOTrade(
            entry_date=pd.Timestamp("2024-01-01"), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=1,
            greeks_entry={"delta": 0.5, "theta": -5.0, "vega": 10.0, "iv": 13.0},
            exit_date=pd.Timestamp("2024-01-03"), exit_spot=22100,
            premium_exit=240.0,
        )
        d, th, v = FnOEngine._decompose_pnl(trade, 0.14)
        # delta_pnl = 0.5 * 100 = 50
        assert d == 50.0
        # theta_pnl = -5.0 * 2 = -10
        assert th == -10.0
        # vega_pnl = 10.0 * (0.14 - 0.13)*100 = 10
        assert v == 10.0


# ---------------------------------------------------------------------------
# run_fno_analysis
# ---------------------------------------------------------------------------

class TestRunFnOAnalysis:
    def test_returns_dataframe(self, signals_buy_only, vix_series):
        df = run_fno_analysis(signals_buy_only, vix_series,
                              num_otm=0, num_expiries=2, min_signal_strength=0)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self, signals_buy_only, vix_series):
        df = run_fno_analysis(signals_buy_only, vix_series,
                              num_otm=0, num_expiries=2, min_signal_strength=0)
        if not df.empty:
            assert "strike_type" in df.columns
            assert "total_pnl" in df.columns
            assert "win_rate" in df.columns

    def test_sorted_by_pnl(self, signals_buy_only, vix_series):
        df = run_fno_analysis(signals_buy_only, vix_series,
                              num_otm=1, num_expiries=3, min_signal_strength=0)
        if len(df) > 1:
            assert df["total_pnl"].iloc[0] >= df["total_pnl"].iloc[1]

    def test_no_signals_empty_result(self, signals_no_signals, vix_series):
        df = run_fno_analysis(signals_no_signals, vix_series, min_signal_strength=0)
        assert df.empty

    def test_strike_labels(self, signals_buy_only, vix_series):
        df = run_fno_analysis(signals_buy_only, vix_series,
                              num_otm=2, num_expiries=3, min_signal_strength=0)
        if not df.empty:
            labels = set(df["strike_type"].values)
            assert "ATM" in labels or any("OTM" in l for l in labels)


# ---------------------------------------------------------------------------
# build_fno_trade_log
# ---------------------------------------------------------------------------

class TestBuildFnOTradeLog:
    def test_returns_dataframe(self, signals_buy_only, vix_series):
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, expiry_week=2)
        log = build_fno_trade_log(result)
        assert isinstance(log, pd.DataFrame)

    def test_log_columns(self, signals_buy_only, vix_series):
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, expiry_week=2)
        log = build_fno_trade_log(result)
        if not log.empty:
            expected_cols = [
                "entry_date", "exit_date", "contract", "expiry",
                "premium_in", "premium_out", "pnl_per_lot", "pnl_net",
                "exit_reason", "sl_points", "tp_points", "rr_ratio",
                "signal_strength", "trade_quality",
            ]
            for col in expected_cols:
                assert col in log.columns, f"Missing column: {col}"

    def test_empty_result_empty_log(self):
        result = FnOResult(trades=[], strike=0, option_type_used="CE", expiry_week=1)
        log = build_fno_trade_log(result)
        assert log.empty

    def test_log_row_count(self, signals_buy_only, vix_series):
        engine = FnOEngine(min_signal_strength=0)
        result = engine.run(signals_buy_only, vix_series, expiry_week=2)
        log = build_fno_trade_log(result)
        assert len(log) == result.num_trades


# ---------------------------------------------------------------------------
# FNO_DEFAULTS
# ---------------------------------------------------------------------------

class TestFNODefaults:
    def test_defaults_keys(self):
        expected_keys = {
            "lot_size", "num_lots", "base_sl", "rr_ratio",
            "ema_period", "slippage", "strike_interval",
            "risk_free_rate", "exit_before_expiry_days",
            "num_otm_strikes", "num_expiries", "min_signal_strength",
        }
        assert set(FNO_DEFAULTS.keys()) == expected_keys

    def test_lot_size_matches(self):
        assert FNO_DEFAULTS["lot_size"] == LOT_SIZE

    def test_strike_interval_matches(self):
        assert FNO_DEFAULTS["strike_interval"] == STRIKE_INTERVAL

    def test_base_sl_default(self):
        assert FNO_DEFAULTS["base_sl"] == 30.0

    def test_rr_ratio_default(self):
        assert FNO_DEFAULTS["rr_ratio"] == 2.0

    def test_min_signal_strength_default(self):
        assert FNO_DEFAULTS["min_signal_strength"] == 2

    def test_adaptive_sl_map_covers_all_strengths(self):
        for strength in range(5):
            assert strength in ADAPTIVE_SL_MAP

    def test_trail_milestones_ordered(self):
        prev_threshold = 0
        for rise_mult, lock_mult in TRAIL_MILESTONES:
            assert rise_mult > prev_threshold
            prev_threshold = rise_mult
