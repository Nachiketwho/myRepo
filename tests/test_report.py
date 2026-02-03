import pandas as pd
import pytest

from backtester.engine import BacktestEngine, BacktestResult, Trade, ExitReason
from backtester.report import build_trade_log, build_summary, format_report, export_csv


@pytest.fixture
def sample_result():
    trades = [
        Trade(
            entry_date=pd.Timestamp("2024-01-02"),
            entry_price=100,
            direction=1,
            exit_date=pd.Timestamp("2024-01-10"),
            exit_price=110,
            exit_reason=ExitReason.TAKE_PROFIT,
        ),
        Trade(
            entry_date=pd.Timestamp("2024-01-11"),
            entry_price=108,
            direction=1,
            exit_date=pd.Timestamp("2024-01-15"),
            exit_price=105,
            exit_reason=ExitReason.STOP_LOSS,
        ),
    ]
    equity = pd.Series(
        [100000, 100010, 100007],
        index=pd.date_range("2024-01-02", periods=3),
    )
    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        params={"sl_pct": 2.0, "tp_pct": 5.0},
    )


class TestBuildTradeLog:
    def test_returns_dataframe(self, sample_result):
        log = build_trade_log(sample_result)
        assert isinstance(log, pd.DataFrame)
        assert len(log) == 2

    def test_columns(self, sample_result):
        log = build_trade_log(sample_result)
        expected = {
            "entry_date", "exit_date", "direction", "entry_price",
            "exit_price", "pnl", "pnl_pct", "exit_reason", "duration_days",
        }
        assert expected == set(log.columns)

    def test_direction_labels(self, sample_result):
        log = build_trade_log(sample_result)
        assert (log["direction"] == "long").all()

    def test_pnl_values(self, sample_result):
        log = build_trade_log(sample_result)
        assert log.iloc[0]["pnl"] == 10
        assert log.iloc[1]["pnl"] == -3

    def test_empty_result(self):
        r = BacktestResult(trades=[], equity_curve=pd.Series(dtype=float))
        log = build_trade_log(r)
        assert len(log) == 0


class TestBuildSummary:
    def test_keys(self, sample_result):
        s = build_summary(sample_result)
        assert "total_trades" in s
        assert "win_rate_pct" in s
        assert "profit_factor" in s
        assert "exit_reason_counts" in s
        assert "params" in s

    def test_values(self, sample_result):
        s = build_summary(sample_result)
        assert s["total_trades"] == 2
        assert s["winning_trades"] == 1
        assert s["losing_trades"] == 1
        assert pytest.approx(s["win_rate_pct"]) == 50.0

    def test_exit_counts(self, sample_result):
        s = build_summary(sample_result)
        assert s["exit_reason_counts"]["take_profit"] == 1
        assert s["exit_reason_counts"]["stop_loss"] == 1


class TestFormatReport:
    def test_returns_string(self, sample_result):
        text = format_report(sample_result)
        assert isinstance(text, str)

    def test_contains_key_metrics(self, sample_result):
        text = format_report(sample_result)
        assert "Win rate" in text
        assert "Profit factor" in text
        assert "Max drawdown" in text
        assert "Total trades" in text

    def test_contains_params(self, sample_result):
        text = format_report(sample_result)
        assert "sl_pct" in text
        assert "tp_pct" in text

    def test_empty_trades(self):
        r = BacktestResult(trades=[], equity_curve=pd.Series(dtype=float))
        text = format_report(r)
        assert "Total trades" in text


class TestExportCSV:
    def test_creates_trade_log_csv(self, sample_result, tmp_path):
        tl = tmp_path / "trades.csv"
        sm = tmp_path / "summary.csv"
        export_csv(sample_result, str(tl), str(sm))

        df = pd.read_csv(tl)
        assert len(df) == 2
        assert "entry_price" in df.columns

    def test_creates_summary_csv(self, sample_result, tmp_path):
        tl = tmp_path / "trades.csv"
        sm = tmp_path / "summary.csv"
        export_csv(sample_result, str(tl), str(sm))

        df = pd.read_csv(sm)
        assert len(df) == 1
        assert "total_trades" in df.columns
        assert df.iloc[0]["total_trades"] == 2

    def test_summary_has_flattened_exit_counts(self, sample_result, tmp_path):
        tl = tmp_path / "trades.csv"
        sm = tmp_path / "summary.csv"
        export_csv(sample_result, str(tl), str(sm))

        df = pd.read_csv(sm)
        assert "exits_take_profit" in df.columns
        assert "exits_stop_loss" in df.columns

    def test_summary_has_params(self, sample_result, tmp_path):
        tl = tmp_path / "trades.csv"
        sm = tmp_path / "summary.csv"
        export_csv(sample_result, str(tl), str(sm))

        df = pd.read_csv(sm)
        assert "param_sl_pct" in df.columns
        assert df.iloc[0]["param_sl_pct"] == 2.0

    def test_empty_result(self, tmp_path):
        r = BacktestResult(trades=[], equity_curve=pd.Series(dtype=float))
        tl = tmp_path / "trades.csv"
        sm = tmp_path / "summary.csv"
        export_csv(r, str(tl), str(sm))

        assert tl.exists()
        assert len(pd.read_csv(sm)) == 1
