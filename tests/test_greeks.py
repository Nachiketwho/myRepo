"""Tests for Black-Scholes option pricing and Greeks."""

import math
import pytest

from backtester.greeks import (
    bs_price, delta, gamma, theta, vega,
    greeks_snapshot, implied_vol,
    _norm_cdf, _norm_pdf, _d1, _d2,
)


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

# Typical Nifty50 parameters
SPOT = 22000.0
ATM_STRIKE = 22000
OTM_CE_STRIKE = 22200
OTM_PE_STRIKE = 21800
T_7D = 7 / 365.0       # 7 calendar days
T_30D = 30 / 365.0      # 30 calendar days
R = 0.07                # 7% risk-free
IV = 0.13               # 13% IV (typical India VIX)


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------

class TestNormHelpers:
    def test_cdf_at_zero(self):
        assert pytest.approx(_norm_cdf(0.0), abs=1e-10) == 0.5

    def test_cdf_symmetry(self):
        assert pytest.approx(_norm_cdf(1.0) + _norm_cdf(-1.0), abs=1e-10) == 1.0

    def test_cdf_large_positive(self):
        assert _norm_cdf(10.0) > 0.9999

    def test_cdf_large_negative(self):
        assert _norm_cdf(-10.0) < 0.0001

    def test_pdf_at_zero(self):
        expected = 1.0 / math.sqrt(2 * math.pi)
        assert pytest.approx(_norm_pdf(0.0), abs=1e-10) == expected

    def test_pdf_symmetry(self):
        assert pytest.approx(_norm_pdf(1.5), abs=1e-10) == _norm_pdf(-1.5)

    def test_pdf_positive(self):
        for x in [-3, -1, 0, 1, 3]:
            assert _norm_pdf(x) > 0


# ---------------------------------------------------------------------------
# d1/d2
# ---------------------------------------------------------------------------

class TestD1D2:
    def test_d1_positive_for_atm(self):
        """ATM with positive rate → d1 > 0."""
        val = _d1(SPOT, ATM_STRIKE, T_30D, R, IV)
        assert val > 0

    def test_d2_less_than_d1(self):
        d1_val = _d1(SPOT, ATM_STRIKE, T_30D, R, IV)
        d2_val = _d2(SPOT, ATM_STRIKE, T_30D, R, IV)
        assert d2_val < d1_val

    def test_d1_d2_difference(self):
        """d1 - d2 = sigma * sqrt(T)."""
        d1_val = _d1(SPOT, ATM_STRIKE, T_30D, R, IV)
        d2_val = _d2(SPOT, ATM_STRIKE, T_30D, R, IV)
        expected_diff = IV * math.sqrt(T_30D)
        assert pytest.approx(d1_val - d2_val, abs=1e-10) == expected_diff

    def test_d1_zero_time(self):
        assert _d1(SPOT, ATM_STRIKE, 0, R, IV) == 0.0

    def test_d1_zero_sigma(self):
        assert _d1(SPOT, ATM_STRIKE, T_30D, R, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Black-Scholes price
# ---------------------------------------------------------------------------

class TestBSPrice:
    def test_ce_positive(self):
        price = bs_price(SPOT, ATM_STRIKE, T_30D, R, IV, "CE")
        assert price > 0

    def test_pe_positive(self):
        price = bs_price(SPOT, ATM_STRIKE, T_30D, R, IV, "PE")
        assert price > 0

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT) (put-call parity)."""
        c = bs_price(SPOT, ATM_STRIKE, T_30D, R, IV, "CE")
        p = bs_price(SPOT, ATM_STRIKE, T_30D, R, IV, "PE")
        parity_rhs = SPOT - ATM_STRIKE * math.exp(-R * T_30D)
        assert pytest.approx(c - p, abs=0.01) == parity_rhs

    def test_ce_increases_with_spot(self):
        low = bs_price(21000, ATM_STRIKE, T_30D, R, IV, "CE")
        high = bs_price(23000, ATM_STRIKE, T_30D, R, IV, "CE")
        assert high > low

    def test_pe_increases_as_spot_falls(self):
        low_spot = bs_price(21000, ATM_STRIKE, T_30D, R, IV, "PE")
        high_spot = bs_price(23000, ATM_STRIKE, T_30D, R, IV, "PE")
        assert low_spot > high_spot

    def test_premium_decreases_with_time(self):
        """Options lose value as time passes (all else equal)."""
        long_t = bs_price(SPOT, ATM_STRIKE, T_30D, R, IV, "CE")
        short_t = bs_price(SPOT, ATM_STRIKE, T_7D, R, IV, "CE")
        assert long_t > short_t

    def test_premium_increases_with_iv(self):
        low_iv = bs_price(SPOT, ATM_STRIKE, T_30D, R, 0.10, "CE")
        high_iv = bs_price(SPOT, ATM_STRIKE, T_30D, R, 0.20, "CE")
        assert high_iv > low_iv

    def test_at_expiry_ce_itm(self):
        """At expiry, CE = max(S-K, 0)."""
        price = bs_price(22500, 22000, 0, R, IV, "CE")
        assert pytest.approx(price) == 500.0

    def test_at_expiry_ce_otm(self):
        price = bs_price(21500, 22000, 0, R, IV, "CE")
        assert price == 0.0

    def test_at_expiry_pe_itm(self):
        price = bs_price(21500, 22000, 0, R, IV, "PE")
        assert pytest.approx(price) == 500.0

    def test_at_expiry_pe_otm(self):
        price = bs_price(22500, 22000, 0, R, IV, "PE")
        assert price == 0.0

    def test_deep_otm_ce_near_zero(self):
        price = bs_price(22000, 25000, T_7D, R, IV, "CE")
        assert price < 1.0

    def test_deep_itm_ce_near_intrinsic(self):
        price = bs_price(25000, 22000, T_7D, R, IV, "CE")
        intrinsic = 25000 - 22000
        assert price >= intrinsic - 1


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

class TestDelta:
    def test_ce_delta_between_0_and_1(self):
        d = delta(SPOT, ATM_STRIKE, T_30D, R, IV, "CE")
        assert 0 < d < 1

    def test_pe_delta_between_neg1_and_0(self):
        d = delta(SPOT, ATM_STRIKE, T_30D, R, IV, "PE")
        assert -1 < d < 0

    def test_atm_ce_delta_near_half(self):
        d = delta(SPOT, ATM_STRIKE, T_30D, R, IV, "CE")
        assert 0.45 < d < 0.60

    def test_deep_itm_ce_delta_near_one(self):
        d = delta(25000, 22000, T_30D, R, IV, "CE")
        assert d > 0.95

    def test_deep_otm_ce_delta_near_zero(self):
        d = delta(22000, 25000, T_30D, R, IV, "CE")
        assert d < 0.05

    def test_ce_pe_delta_relationship(self):
        """delta_CE - delta_PE ≈ 1."""
        d_ce = delta(SPOT, ATM_STRIKE, T_30D, R, IV, "CE")
        d_pe = delta(SPOT, ATM_STRIKE, T_30D, R, IV, "PE")
        assert pytest.approx(d_ce - d_pe, abs=0.01) == 1.0

    def test_delta_at_expiry_ce_itm(self):
        assert delta(22500, 22000, 0, R, IV, "CE") == 1.0

    def test_delta_at_expiry_ce_otm(self):
        assert delta(21500, 22000, 0, R, IV, "CE") == 0.0

    def test_delta_at_expiry_pe_itm(self):
        assert delta(21500, 22000, 0, R, IV, "PE") == -1.0

    def test_delta_at_expiry_pe_otm(self):
        assert delta(22500, 22000, 0, R, IV, "PE") == 0.0


class TestGamma:
    def test_gamma_positive(self):
        g = gamma(SPOT, ATM_STRIKE, T_30D, R, IV)
        assert g > 0

    def test_atm_gamma_highest(self):
        """ATM gamma > deep OTM gamma."""
        g_atm = gamma(SPOT, ATM_STRIKE, T_30D, R, IV)
        g_deep_otm = gamma(SPOT, 23000, T_30D, R, IV)  # 1000pts OTM
        assert g_atm > g_deep_otm

    def test_gamma_increases_near_expiry(self):
        """ATM gamma rises as expiry approaches."""
        g_far = gamma(SPOT, ATM_STRIKE, T_30D, R, IV)
        g_near = gamma(SPOT, ATM_STRIKE, T_7D, R, IV)
        assert g_near > g_far

    def test_gamma_at_expiry_zero(self):
        assert gamma(SPOT, ATM_STRIKE, 0, R, IV) == 0.0

    def test_gamma_zero_sigma(self):
        assert gamma(SPOT, ATM_STRIKE, T_30D, R, 0.0) == 0.0


class TestTheta:
    def test_theta_negative_for_ce(self):
        """Long option theta is negative (time decay)."""
        t = theta(SPOT, ATM_STRIKE, T_30D, R, IV, "CE")
        assert t < 0

    def test_theta_negative_for_pe(self):
        t = theta(SPOT, ATM_STRIKE, T_30D, R, IV, "PE")
        assert t < 0

    def test_atm_theta_larger_magnitude(self):
        """ATM has more time decay than OTM."""
        t_atm = abs(theta(SPOT, ATM_STRIKE, T_30D, R, IV, "CE"))
        t_otm = abs(theta(SPOT, OTM_CE_STRIKE, T_30D, R, IV, "CE"))
        assert t_atm > t_otm

    def test_theta_accelerates_near_expiry(self):
        t_far = abs(theta(SPOT, ATM_STRIKE, T_30D, R, IV, "CE"))
        t_near = abs(theta(SPOT, ATM_STRIKE, T_7D, R, IV, "CE"))
        assert t_near > t_far

    def test_theta_at_expiry_zero(self):
        assert theta(SPOT, ATM_STRIKE, 0, R, IV, "CE") == 0.0


class TestVega:
    def test_vega_positive(self):
        v = vega(SPOT, ATM_STRIKE, T_30D, R, IV)
        assert v > 0

    def test_atm_vega_highest(self):
        v_atm = vega(SPOT, ATM_STRIKE, T_30D, R, IV)
        v_deep_otm = vega(SPOT, 23000, T_30D, R, IV)  # 1000pts OTM
        assert v_atm > v_deep_otm

    def test_vega_decreases_near_expiry(self):
        v_far = vega(SPOT, ATM_STRIKE, T_30D, R, IV)
        v_near = vega(SPOT, ATM_STRIKE, T_7D, R, IV)
        assert v_far > v_near

    def test_vega_at_expiry_zero(self):
        assert vega(SPOT, ATM_STRIKE, 0, R, IV) == 0.0


# ---------------------------------------------------------------------------
# Greeks snapshot
# ---------------------------------------------------------------------------

class TestGreeksSnapshot:
    def test_returns_all_keys(self):
        snap = greeks_snapshot(SPOT, ATM_STRIKE, T_30D, R, IV, "CE")
        assert set(snap.keys()) == {"premium", "delta", "gamma", "theta", "vega", "iv"}

    def test_iv_in_percent(self):
        snap = greeks_snapshot(SPOT, ATM_STRIKE, T_30D, R, IV, "CE")
        assert pytest.approx(snap["iv"]) == 13.0

    def test_premium_matches_bs_price(self):
        snap = greeks_snapshot(SPOT, ATM_STRIKE, T_30D, R, IV, "CE")
        expected = round(bs_price(SPOT, ATM_STRIKE, T_30D, R, IV, "CE"), 2)
        assert snap["premium"] == expected


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

class TestImpliedVol:
    def test_roundtrip_ce(self):
        """Price an option, then recover IV from that price."""
        price = bs_price(SPOT, ATM_STRIKE, T_30D, R, IV, "CE")
        recovered = implied_vol(price, SPOT, ATM_STRIKE, T_30D, R, "CE")
        assert recovered is not None
        assert pytest.approx(recovered, abs=1e-4) == IV

    def test_roundtrip_pe(self):
        price = bs_price(SPOT, ATM_STRIKE, T_30D, R, IV, "PE")
        recovered = implied_vol(price, SPOT, ATM_STRIKE, T_30D, R, "PE")
        assert recovered is not None
        assert pytest.approx(recovered, abs=1e-4) == IV

    def test_zero_price_returns_none(self):
        assert implied_vol(0, SPOT, ATM_STRIKE, T_30D, R) is None

    def test_negative_price_returns_none(self):
        assert implied_vol(-10, SPOT, ATM_STRIKE, T_30D, R) is None

    def test_zero_time_returns_none(self):
        assert implied_vol(100, SPOT, ATM_STRIKE, 0, R) is None

    def test_high_iv_recovery(self):
        high_iv = 0.40
        price = bs_price(SPOT, ATM_STRIKE, T_30D, R, high_iv, "CE")
        recovered = implied_vol(price, SPOT, ATM_STRIKE, T_30D, R, "CE")
        assert recovered is not None
        assert pytest.approx(recovered, abs=1e-3) == high_iv
