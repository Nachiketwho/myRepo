"""Transaction cost calculator for NSE F&O option trades.

Standard discount broker (Zerodha-style) rates for options:
    - Brokerage: Rs 20 per order (flat, each side)
    - STT: 0.0625% on sell-side premium x lot_size
    - Exchange charges: ~0.05% on turnover (both sides)
    - GST: 18% on (brokerage + exchange charges)
    - Stamp duty: 0.003% on buy-side turnover
    - SEBI fees: Rs 10 per crore of turnover (negligible)
"""

from dataclasses import dataclass


@dataclass
class TxnCosts:
    """Transaction cost structure for NSE F&O."""

    brokerage_per_order: float = 20.0   # Rs per order per side
    stt_sell_pct: float = 0.0625        # % on sell-side turnover
    exchange_pct: float = 0.05          # % on total turnover
    gst_pct: float = 18.0              # % on (brokerage + exchange)
    stamp_duty_pct: float = 0.003      # % on buy-side turnover
    sebi_per_crore: float = 10.0       # Rs per crore turnover

    def round_trip(
        self,
        premium_in: float,
        premium_out: float,
        lot_size: int,
        num_lots: int,
    ) -> dict:
        """Calculate all cost components for a round-trip option trade.

        Returns dict with itemised costs and total (all in Rs).
        """
        qty = lot_size * num_lots
        buy_turnover = premium_in * qty
        sell_turnover = premium_out * qty
        total_turnover = buy_turnover + sell_turnover

        brokerage = self.brokerage_per_order * 2  # buy + sell
        stt = sell_turnover * self.stt_sell_pct / 100
        exchange = total_turnover * self.exchange_pct / 100
        gst = (brokerage + exchange) * self.gst_pct / 100
        stamp = buy_turnover * self.stamp_duty_pct / 100
        sebi = total_turnover / 1e7 * self.sebi_per_crore

        total = brokerage + stt + exchange + gst + stamp + sebi
        return {
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "exchange": round(exchange, 2),
            "gst": round(gst, 2),
            "stamp_duty": round(stamp, 2),
            "sebi": round(sebi, 4),
            "total": round(total, 2),
        }

    def breakeven_points(
        self,
        premium_in: float,
        lot_size: int,
        num_lots: int,
    ) -> float:
        """Minimum premium rise (in points) to break even after costs.

        Uses entry premium for both sides as an approximation.
        """
        costs = self.round_trip(premium_in, premium_in, lot_size, num_lots)
        qty = lot_size * num_lots
        if qty == 0:
            return 0.0
        return round(costs["total"] / qty, 2)


# Default NSE F&O cost structure (discount broker)
NSE_FNO_COSTS = TxnCosts()
