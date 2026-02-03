import pandas as pd
import numpy as np
import pytest

from backtester.engine import BacktestEngine, BacktestResult, Trade, ExitReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signals(prices, signals, highs=None, lows=None):
    """Build a minimal signals DataFrame for engine testing."""
    n = len(prices)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "open": prices,
            "high": highs if highs is not None else [p + 2 for p in prices],
            "low": lows if lows is not None else [p - 2 for p in prices],
            "close": prices,
            "signal": signals,
        },
        index=dates,
    )
    return df


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------

class TestTrade:
    def test_pnl_long(self):
        t = Trade(entry_date=None, entry_price=100, direction=1,
                  exit_price=110, exit_reason=ExitReason.TAKE_PROFIT)
        assert t.pnl == 10

    def test_pnl_short(self):
        t = Trade(entry_date=None, entry_price=100, direction=-1,
                  exit_price=90, exit_reason=ExitReason.TAKE_PROFIT)
        assert t.pnl == 10

    def test_pnl_pct(self):
        t = Trade(entry_date=None, entry_price=200, direction=1,
                  exit_price=210)
        assert pytest.approx(t.pnl_pct) == 5.0

    def test_pnl_pct_zero_entry(self):
        t = Trade(entry_date=None, entry_price=0, direction=1, exit_price=10)
        assert t.pnl_pct == 0.0

    def test_duration(self):
        t = Trade(
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=100,
            direction=1,
            exit_date=pd.Timestamp("2024-01-11"),
            exit_price=110,
        )
        assert t.duration == 10


# ---------------------------------------------------------------------------
# Engine — entry / exit mechanics
# ---------------------------------------------------------------------------

class TestEngineBasics:
    def test_no_signals_no_trades(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df["signal"] = 0
        engine = BacktestEngine()
        result = engine.run(df)
        assert result.num_trades == 0

    def test_buy_and_hold_to_end(self):
        """Single buy signal, held to end of data."""
        df = _make_signals(
            prices=[100, 102, 104, 106, 108],
            signals=[1, 0, 0, 0, 0],
            highs=[101, 103, 105, 107, 109],
            lows=[99, 101, 103, 105, 107],
        )
        engine = BacktestEngine(sl_pct=50, tsl_pct=50, tp_pct=50, ttp_pct=0)
        result = engine.run(df)
        assert result.num_trades == 1
        assert result.trades[0].exit_reason == ExitReason.END_OF_DATA
        assert result.trades[0].pnl == 8  # 108 - 100

    def test_signal_reversal_exit(self):
        """Sell signal while long → exit via SIGNAL."""
        df = _make_signals(
            prices=[100, 102, 104, 103, 101],
            signals=[1, 0, 0, -1, 0],
            highs=[101, 103, 105, 104, 102],
            lows=[99, 101, 103, 102, 100],
        )
        engine = BacktestEngine(sl_pct=50, tsl_pct=50, tp_pct=50, ttp_pct=0)
        result = engine.run(df)
        assert result.num_trades >= 1
        first = result.trades[0]
        assert first.exit_reason == ExitReason.SIGNAL
        assert first.pnl == 3  # exit at close=103


class TestStopLoss:
    def test_fixed_stop_loss_long(self):
        """Price drops below SL level → exit at SL price."""
        df = _make_signals(
            prices=[100, 97, 95, 93, 90],
            signals=[1, 0, 0, 0, 0],
            highs=[101, 98, 96, 94, 91],
            lows=[99, 95, 93, 91, 88],
        )
        engine = BacktestEngine(sl_pct=3.0, tsl_pct=50, tp_pct=50, ttp_pct=0)
        result = engine.run(df)
        assert result.num_trades == 1
        assert result.trades[0].exit_reason == ExitReason.STOP_LOSS
        assert pytest.approx(result.trades[0].exit_price) == 97.0  # 100*(1-0.03)

    def test_fixed_stop_loss_short(self):
        df = _make_signals(
            prices=[100, 103, 106, 110, 115],
            signals=[-1, 0, 0, 0, 0],
            highs=[101, 104, 107, 111, 116],
            lows=[99, 102, 105, 109, 114],
        )
        engine = BacktestEngine(sl_pct=3.0, tsl_pct=50, tp_pct=50, ttp_pct=0)
        result = engine.run(df)
        assert result.num_trades == 1
        assert result.trades[0].exit_reason == ExitReason.STOP_LOSS


class TestTrailingStop:
    def test_trailing_stop_after_rise(self):
        """Price rises then reverses — trailing stop triggers."""
        df = _make_signals(
            prices=[100, 105, 110, 115, 110, 105],
            signals=[1, 0, 0, 0, 0, 0],
            highs=[101, 106, 111, 116, 111, 106],
            lows=[99, 104, 109, 114, 108, 103],
        )
        # SL at 50% (won't trigger), TSL at 5%
        engine = BacktestEngine(sl_pct=50, tsl_pct=5.0, tp_pct=50, ttp_pct=0)
        result = engine.run(df)
        assert result.num_trades == 1
        t = result.trades[0]
        assert t.exit_reason == ExitReason.TRAILING_STOP
        # Highest was 116, TSL = 116 * 0.95 = 110.2, bar 4 low=108 < 110.2
        assert pytest.approx(t.exit_price, rel=1e-2) == 116 * 0.95


class TestTakeProfit:
    def test_take_profit_no_trailing(self):
        """TP hit with ttp_pct=0 → immediate exit."""
        df = _make_signals(
            prices=[100, 104, 108, 112, 116],
            signals=[1, 0, 0, 0, 0],
            highs=[101, 105, 109, 113, 117],
            lows=[99, 103, 107, 111, 115],
        )
        engine = BacktestEngine(sl_pct=50, tsl_pct=50, tp_pct=5.0, ttp_pct=0)
        result = engine.run(df)
        assert result.num_trades == 1
        assert result.trades[0].exit_reason == ExitReason.TAKE_PROFIT
        assert pytest.approx(result.trades[0].exit_price) == 105.0  # 100*1.05

    def test_trailing_take_profit(self):
        """TP hit → switch to trailing mode → exit when price drops."""
        df = _make_signals(
            prices=[100, 106, 112, 118, 115, 108],
            signals=[1, 0, 0, 0, 0, 0],
            highs=[101, 107, 113, 119, 116, 109],
            lows=[99, 105, 111, 117, 113, 106],
        )
        # TP at 5% (hit at bar1 high=107 > 105), then trail at 3%
        engine = BacktestEngine(sl_pct=50, tsl_pct=50, tp_pct=5.0, ttp_pct=3.0)
        result = engine.run(df)
        assert result.num_trades >= 1
        t = result.trades[0]
        assert t.exit_reason == ExitReason.TRAILING_TP


# ---------------------------------------------------------------------------
# BacktestResult stats
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_win_rate(self):
        trades = [
            Trade(None, 100, 1, exit_price=110, exit_reason=ExitReason.TAKE_PROFIT),
            Trade(None, 100, 1, exit_price=95, exit_reason=ExitReason.STOP_LOSS),
        ]
        r = BacktestResult(trades=trades, equity_curve=pd.Series([100000, 100010]))
        assert pytest.approx(r.win_rate) == 50.0

    def test_profit_factor(self):
        trades = [
            Trade(None, 100, 1, exit_price=120, exit_reason=ExitReason.TAKE_PROFIT),
            Trade(None, 100, 1, exit_price=90, exit_reason=ExitReason.STOP_LOSS),
        ]
        r = BacktestResult(trades=trades, equity_curve=pd.Series([100000]))
        assert pytest.approx(r.profit_factor) == 2.0  # 20 / 10

    def test_profit_factor_no_losses(self):
        trades = [
            Trade(None, 100, 1, exit_price=120, exit_reason=ExitReason.TAKE_PROFIT),
        ]
        r = BacktestResult(trades=trades, equity_curve=pd.Series([100000]))
        assert r.profit_factor == float("inf")

    def test_max_drawdown(self):
        eq = pd.Series([100, 110, 105, 95, 100])
        r = BacktestResult(trades=[], equity_curve=eq)
        # Peak 110, trough 95 → dd = (95-110)/110 = -13.6%
        assert pytest.approx(r.max_drawdown, rel=0.01) == -13.636

    def test_empty_trades(self):
        r = BacktestResult(trades=[], equity_curve=pd.Series(dtype=float))
        assert r.num_trades == 0
        assert r.win_rate == 0.0
        assert r.total_pnl == 0.0
        assert r.avg_pnl == 0.0

    def test_equity_curve_length(self, signals_with_buy):
        engine = BacktestEngine()
        result = engine.run(signals_with_buy)
        assert len(result.equity_curve) == len(signals_with_buy)
