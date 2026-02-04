"""Tests for transaction cost calculator."""

import pytest

from backtester.costs import TxnCosts, NSE_FNO_COSTS


class TestTxnCosts:
    def test_round_trip_returns_all_components(self):
        costs = TxnCosts()
        result = costs.round_trip(100.0, 120.0, 25, 1)
        assert "brokerage" in result
        assert "stt" in result
        assert "exchange" in result
        assert "gst" in result
        assert "stamp_duty" in result
        assert "sebi" in result
        assert "total" in result

    def test_brokerage_flat_per_side(self):
        costs = TxnCosts(brokerage_per_order=20.0)
        result = costs.round_trip(100.0, 120.0, 25, 1)
        assert result["brokerage"] == 40.0  # 20 * 2

    def test_stt_on_sell_side(self):
        costs = TxnCosts(stt_sell_pct=0.0625)
        result = costs.round_trip(100.0, 120.0, 25, 1)
        sell_turnover = 120.0 * 25 * 1
        expected_stt = round(sell_turnover * 0.0625 / 100, 2)
        assert result["stt"] == expected_stt

    def test_total_is_sum_of_components(self):
        costs = TxnCosts()
        result = costs.round_trip(100.0, 120.0, 25, 1)
        component_sum = (
            result["brokerage"] + result["stt"] + result["exchange"]
            + result["gst"] + result["stamp_duty"] + result["sebi"]
        )
        assert pytest.approx(result["total"], abs=0.01) == component_sum

    def test_total_positive(self):
        costs = TxnCosts()
        result = costs.round_trip(100.0, 80.0, 25, 1)
        assert result["total"] > 0

    def test_multiple_lots(self):
        costs = TxnCosts()
        r1 = costs.round_trip(100.0, 120.0, 25, 1)
        r2 = costs.round_trip(100.0, 120.0, 25, 2)
        # STT, exchange, stamp should scale with lots
        # Brokerage stays same (flat per order)
        assert r2["stt"] > r1["stt"]
        assert r2["brokerage"] == r1["brokerage"]

    def test_zero_premium(self):
        costs = TxnCosts()
        result = costs.round_trip(0.0, 0.0, 25, 1)
        assert result["brokerage"] == 40.0
        assert result["stt"] == 0.0

    def test_breakeven_points(self):
        costs = TxnCosts()
        be = costs.breakeven_points(200.0, 25, 1)
        assert be > 0
        assert isinstance(be, float)

    def test_breakeven_zero_qty(self):
        costs = TxnCosts()
        be = costs.breakeven_points(200.0, 0, 1)
        assert be == 0.0

    def test_nse_fno_costs_defaults(self):
        assert NSE_FNO_COSTS.brokerage_per_order == 20.0
        assert NSE_FNO_COSTS.stt_sell_pct == 0.0625
        assert NSE_FNO_COSTS.gst_pct == 18.0
