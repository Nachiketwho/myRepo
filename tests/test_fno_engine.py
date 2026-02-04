"""Tests for F&O backtesting engine."""

import datetime as dt
import numpy as np
import pandas as pd
import pytest

from backtester.fno_engine import (
    FnOTrade, FnOResult, FnOEngine, FnOExitReason,
    FNO_DEFAULTS, run_fno_analysis, build_fno_trade_log,
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
def signals_reversal():
    """20-bar data with buy on bar 3, sell on bar 10 (signal reversal)."""
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
    df.iloc[10, df.columns.get_loc("signal")] = -1  # sell → reversal
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

    def test_pnl_total(self):
        trade = FnOTrade(
            entry_date=dt.date(2024, 1, 1), entry_spot=22000,
            strike=22000, option_type="CE", expiry=dt.date(2024, 1, 11),
            premium_entry=200.0, lot_size=25, num_lots=2,
            greeks_entry={}, premium_exit=250.0,
        )
        assert trade.pnl == 50.0 * 25 * 2

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
        assert trade.pnl < 0


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
        engine = FnOEngine()
        result = engine.run(signals_buy_only, vix_series)
        assert result.num_trades >= 1
        assert result.trades[0].option_type == "CE"

    def test_sell_signal_creates_pe(self, signals_sell_only, vix_series):
        engine = FnOEngine()
        result = engine.run(signals_sell_only, vix_series)
        assert result.num_trades >= 1
        assert result.trades[0].option_type == "PE"

    def test_no_signals_no_trades(self, signals_no_signals, vix_series):
        engine = FnOEngine()
        result = engine.run(signals_no_signals, vix_series)
        assert result.num_trades == 0

    def test_strike_offset_atm(self, signals_buy_only, vix_series):
        engine = FnOEngine()
        result = engine.run(signals_buy_only, vix_series, strike_offset=0)
        if result.trades:
            trade = result.trades[0]
            expected_strike = round(trade.entry_spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
            assert trade.strike == expected_strike

    def test_strike_offset_otm_ce(self, signals_buy_only, vix_series):
        engine = FnOEngine()
        result = engine.run(signals_buy_only, vix_series, strike_offset=2)
        if result.trades:
            trade = result.trades[0]
            atm = round(trade.entry_spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
            assert trade.strike == atm + 2 * STRIKE_INTERVAL

    def test_strike_offset_otm_pe(self, signals_sell_only, vix_series):
        engine = FnOEngine()
        result = engine.run(signals_sell_only, vix_series, strike_offset=2)
        if result.trades:
            trade = result.trades[0]
            atm = round(trade.entry_spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
            assert trade.strike == atm - 2 * STRIKE_INTERVAL

    def test_expiry_week_selection(self, signals_buy_only, vix_series):
        engine = FnOEngine()
        r1 = engine.run(signals_buy_only, vix_series, expiry_week=1)
        r2 = engine.run(signals_buy_only, vix_series, expiry_week=3)
        if r1.trades and r2.trades:
            assert r2.trades[0].expiry > r1.trades[0].expiry

    def test_end_of_data_exit(self, vix_series):
        """Open position at end of data should be closed with END_OF_DATA."""
        # Use short data (5 bars) with signal on bar 0 and far expiry (week 4)
        # so expiry_eve doesn't trigger first
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({
            "open":   [22000] * 5,
            "high":   [22050] * 5,
            "low":    [21950] * 5,
            "close":  [22000] * 5,
            "volume": [1000] * 5,
            "signal": [1, 0, 0, 0, 0],
        }, index=dates)
        engine = FnOEngine(sl_points=9999, tp_points=9999)
        result = engine.run(df, vix_series, expiry_week=4)  # ~4 weeks out
        assert result.num_trades >= 1
        last_trade = result.trades[-1]
        assert last_trade.exit_reason == FnOExitReason.END_OF_DATA

    def test_signal_reversal_exit(self, vix_series):
        """Buy CE on bar 3, sell signal on bar 10 → reversal exit."""
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
        df.iloc[10, df.columns.get_loc("signal")] = -1  # sell → reversal
        engine = FnOEngine(sl_points=9999, tp_points=9999)
        result = engine.run(df, vix_series, expiry_week=4)  # far expiry
        has_reversal = any(
            t.exit_reason == FnOExitReason.SIGNAL_REVERSAL
            for t in result.trades
        )
        assert has_reversal

    def test_greeks_populated(self, signals_buy_only, vix_series):
        engine = FnOEngine()
        result = engine.run(signals_buy_only, vix_series)
        if result.trades:
            trade = result.trades[0]
            assert "delta" in trade.greeks_entry
            assert "theta" in trade.greeks_entry
            assert "vega" in trade.greeks_entry
            assert "iv" in trade.greeks_entry

    def test_premium_entry_positive(self, signals_buy_only, vix_series):
        engine = FnOEngine()
        result = engine.run(signals_buy_only, vix_series)
        for trade in result.trades:
            assert trade.premium_entry > 0

    def test_custom_lot_size(self, signals_buy_only, vix_series):
        engine = FnOEngine(lot_size=50, num_lots=2)
        result = engine.run(signals_buy_only, vix_series)
        if result.trades:
            assert result.trades[0].lot_size == 50
            assert result.trades[0].num_lots == 2

    def test_result_params_stored(self, signals_buy_only, vix_series):
        engine = FnOEngine(sl_points=40, tp_points=60)
        result = engine.run(signals_buy_only, vix_series, strike_offset=1, expiry_week=2)
        assert result.params["sl_points"] == 40
        assert result.params["tp_points"] == 60
        assert result.params["strike_offset"] == 1
        assert result.params["expiry_week"] == 2

    def test_multiple_trades(self, vix_series):
        """Multiple signals → multiple trades."""
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        df = pd.DataFrame({
            "open":   [22000] * 40,
            "high":   [22050] * 40,
            "low":    [21950] * 40,
            "close":  [22000] * 40,
            "volume": [1000] * 40,
            "signal": [0] * 40,
        }, index=dates)
        # Three buy signals spaced apart
        df.iloc[2, df.columns.get_loc("signal")] = 1
        df.iloc[15, df.columns.get_loc("signal")] = -1  # reversal closes first, opens PE
        df.iloc[30, df.columns.get_loc("signal")] = 1   # reversal closes second, opens CE

        engine = FnOEngine(sl_points=9999, tp_points=9999)
        result = engine.run(df, vix_series)
        assert result.num_trades >= 2


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
        # Jan 3 not in index → forward fill from Jan 2
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
# Exit conditions
# ---------------------------------------------------------------------------

class TestCheckExit:
    """Test _check_exit directly for specific exit scenarios."""

    def _make_position(self, premium_entry=200.0, option_type="CE",
                       expiry_days_ahead=7):
        current = dt.date(2024, 1, 10)
        expiry = current + dt.timedelta(days=expiry_days_ahead)
        return FnOTrade(
            entry_date=current, entry_spot=22000,
            strike=22000, option_type=option_type, expiry=expiry,
            premium_entry=premium_entry, lot_size=25, num_lots=1,
            greeks_entry={}, premium_high=premium_entry,
        )

    def test_expiry_eve_exit(self, vix_series):
        """Position should exit when close to expiry."""
        pos = self._make_position(expiry_days_ahead=1)
        engine = FnOEngine(exit_before_expiry_days=1)
        _, reason = engine._check_exit(
            pos, 22000, dt.date(2024, 1, 10), 0.13, 0
        )
        assert reason == FnOExitReason.EXPIRY_EVE

    def test_sl_exit(self):
        """Premium drops by sl_points → SL exit."""
        pos = self._make_position(premium_entry=200.0, expiry_days_ahead=14)
        engine = FnOEngine(sl_points=50)
        # Simulate a big spot move down so CE premium drops
        # We need to find a spot where BS price gives premium ~150 or less
        # For a CE, moving spot down significantly should drop premium
        _, reason = engine._check_exit(
            pos, 21500, dt.date(2024, 1, 11), 0.13, 0
        )
        assert reason == FnOExitReason.SL_POINTS

    def test_signal_reversal_ce_with_sell(self):
        """CE position + sell signal → reversal exit."""
        pos = self._make_position(option_type="CE", expiry_days_ahead=14)
        engine = FnOEngine()
        _, reason = engine._check_exit(
            pos, 22000, dt.date(2024, 1, 11), 0.13, -1
        )
        assert reason == FnOExitReason.SIGNAL_REVERSAL

    def test_signal_reversal_pe_with_buy(self):
        """PE position + buy signal → reversal exit."""
        pos = self._make_position(option_type="PE", expiry_days_ahead=14)
        engine = FnOEngine()
        _, reason = engine._check_exit(
            pos, 22000, dt.date(2024, 1, 11), 0.13, 1
        )
        assert reason == FnOExitReason.SIGNAL_REVERSAL

    def test_no_exit_when_within_bounds(self):
        """No exit when premium change is within SL/TP bounds."""
        pos = self._make_position(premium_entry=200.0, expiry_days_ahead=14)
        engine = FnOEngine(sl_points=500, tp_points=500)
        _, reason = engine._check_exit(
            pos, 22010, dt.date(2024, 1, 11), 0.13, 0
        )
        assert reason is None

    def test_same_direction_signal_no_exit(self):
        """CE position + buy signal → no exit (same direction)."""
        pos = self._make_position(option_type="CE", expiry_days_ahead=14)
        engine = FnOEngine(sl_points=500, tp_points=500)
        _, reason = engine._check_exit(
            pos, 22000, dt.date(2024, 1, 11), 0.13, 1
        )
        assert reason is None


# ---------------------------------------------------------------------------
# run_fno_analysis
# ---------------------------------------------------------------------------

class TestRunFnOAnalysis:
    def test_returns_dataframe(self, signals_buy_only, vix_series):
        df = run_fno_analysis(signals_buy_only, vix_series, num_otm=1, num_expiries=1)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self, signals_buy_only, vix_series):
        df = run_fno_analysis(signals_buy_only, vix_series, num_otm=1, num_expiries=1)
        if not df.empty:
            assert "strike_type" in df.columns
            assert "total_pnl" in df.columns
            assert "win_rate" in df.columns

    def test_sorted_by_pnl(self, signals_buy_only, vix_series):
        df = run_fno_analysis(signals_buy_only, vix_series, num_otm=2, num_expiries=2)
        if len(df) > 1:
            assert df["total_pnl"].iloc[0] >= df["total_pnl"].iloc[1]

    def test_no_signals_empty_result(self, signals_no_signals, vix_series):
        df = run_fno_analysis(signals_no_signals, vix_series)
        assert df.empty

    def test_strike_labels(self, signals_buy_only, vix_series):
        df = run_fno_analysis(signals_buy_only, vix_series, num_otm=2, num_expiries=1)
        if not df.empty:
            labels = set(df["strike_type"].values)
            assert "ATM" in labels or any("OTM" in l for l in labels)


# ---------------------------------------------------------------------------
# build_fno_trade_log
# ---------------------------------------------------------------------------

class TestBuildFnOTradeLog:
    def test_returns_dataframe(self, signals_buy_only, vix_series):
        engine = FnOEngine()
        result = engine.run(signals_buy_only, vix_series)
        log = build_fno_trade_log(result)
        assert isinstance(log, pd.DataFrame)

    def test_log_columns(self, signals_buy_only, vix_series):
        engine = FnOEngine()
        result = engine.run(signals_buy_only, vix_series)
        log = build_fno_trade_log(result)
        if not log.empty:
            expected_cols = [
                "entry_date", "exit_date", "contract", "expiry",
                "premium_in", "premium_out", "pnl_per_lot", "pnl_total",
                "exit_reason",
            ]
            for col in expected_cols:
                assert col in log.columns

    def test_empty_result_empty_log(self):
        result = FnOResult(trades=[], strike=0, option_type_used="CE", expiry_week=1)
        log = build_fno_trade_log(result)
        assert log.empty

    def test_log_row_count(self, signals_buy_only, vix_series):
        engine = FnOEngine()
        result = engine.run(signals_buy_only, vix_series)
        log = build_fno_trade_log(result)
        assert len(log) == result.num_trades


# ---------------------------------------------------------------------------
# FNO_DEFAULTS
# ---------------------------------------------------------------------------

class TestFNODefaults:
    def test_defaults_keys(self):
        expected_keys = {
            "lot_size", "num_lots", "sl_points", "tp_points",
            "tsl_points", "tsl_activation", "strike_interval",
            "risk_free_rate", "exit_before_expiry_days",
            "num_otm_strikes", "num_expiries",
        }
        assert set(FNO_DEFAULTS.keys()) == expected_keys

    def test_lot_size_matches(self):
        assert FNO_DEFAULTS["lot_size"] == LOT_SIZE

    def test_strike_interval_matches(self):
        assert FNO_DEFAULTS["strike_interval"] == STRIKE_INTERVAL
