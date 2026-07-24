"""
Tests for groupoid algebra of financial events (events/operators.py).
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from events.operators import (
    EventMode, EventOperator,
    stock_split_operator, dividend_operator, earnings_shock_operator,
    analyst_rating_operator, secondary_offering_operator, share_buyback_operator,
    trading_halt_operator, short_squeeze_operator, index_change_operator,
    rate_change_operator, qe_operator, qt_operator, systemic_crisis_operator,
    circuit_breaker_operator, volatility_regime_shift_operator, inflation_shock_operator,
    merger_operator, spinoff_operator, ipo_operator, delisting_operator, bankruptcy_operator,
    compose, event_sequence,
    P, V, L, K, I,
)


class TestEventModes:
    def test_stock_split_mode_local(self):
        op = stock_split_operator(ratio=2.0, n=1)
        assert op.mode == EventMode.LOCAL

    def test_stock_split_log_price_decreases(self):
        """2:1 split halves share price → log_price shift = -log(2)."""
        op = stock_split_operator(ratio=2.0, n=1)
        np.testing.assert_allclose(op.b_w[P], -np.log(2.0), rtol=1e-6)

    def test_stock_split_shares_outstanding_increases(self):
        """2:1 split doubles shares outstanding → log(shares) shift = +log(2)."""
        op = stock_split_operator(ratio=2.0, n=1)
        np.testing.assert_allclose(op.b_w[K], np.log(2.0), rtol=1e-6)

    def test_dividend_mode_local(self):
        op = dividend_operator(div_yield=0.02, n=1)
        assert op.mode == EventMode.LOCAL

    def test_dividend_lowers_price(self):
        """Dividend of q → price drops by -log(1+q) (ex-date adjustment)."""
        op = dividend_operator(div_yield=0.02, n=1)
        np.testing.assert_allclose(op.b_w[P], -np.log(1.02), rtol=1e-5)

    def test_rate_change_mode_global(self):
        op = rate_change_operator(change_bps=25, n=4)
        assert op.mode == EventMode.GLOBAL

    def test_rate_hike_lowers_prices(self):
        """Rate hike (positive change_bps) should lower all log prices."""
        n = 4
        op = rate_change_operator(change_bps=25, n=n)
        for i in range(n):
            assert op.b_w[i * 5 + P] < 0, f"Asset {i} price should decrease after rate hike"

    def test_rate_cut_raises_prices(self):
        """Rate cut (negative change_bps) should raise all log prices."""
        n = 3
        op = rate_change_operator(change_bps=-50, n=n)
        for i in range(n):
            assert op.b_w[i * 5 + P] > 0, f"Asset {i} price should increase after rate cut"

    def test_merger_mode_pairwise(self):
        op = merger_operator(acquirer_idx=0, target_idx=1, n=2)
        assert op.mode == EventMode.PAIRWISE

    def test_merger_reduces_universe(self):
        """Merger of 2 assets → 1 combined. A_w maps 2d → d (n-1=1)."""
        d = 5
        op = merger_operator(acquirer_idx=0, target_idx=1, n=2)
        assert op.A_w.shape == (d, 2 * d)
        assert op.target_dim == 1
        assert op.source_dim == 2

    def test_ipo_mode_pairwise(self):
        op = ipo_operator(ticker='NEWCO', ipo_price=60.0, n=4)
        assert op.mode == EventMode.PAIRWISE

    def test_ipo_increases_universe(self):
        """IPO: n assets → n+1 assets (one new listing added)."""
        n = 4
        op = ipo_operator(ticker='NEWCO', ipo_price=60.0, n=n)
        assert op.target_dim == n + 1
        assert op.source_dim == n

    def test_ipo_sets_log_price(self):
        """IPO new asset gets log(ipo_price) as its price coordinate."""
        ipo_price = 90.0
        op = ipo_operator(ticker='NEWCO', ipo_price=ipo_price, n=2)
        # New asset state is at the end: indices [2*d : 3*d]
        d = 5
        new_asset_start = 2 * d
        np.testing.assert_allclose(op.b_w[new_asset_start + P], np.log(ipo_price), rtol=1e-6)

    def test_systemic_crisis_global(self):
        op = systemic_crisis_operator(severity=0.8, n=3)
        assert op.mode == EventMode.GLOBAL

    def test_short_squeeze_nonidentity_A(self):
        """Short squeeze is the only Mode I op with A_w ≠ I (momentum self-reinforcement)."""
        op = short_squeeze_operator(squeeze_intensity=0.5, n=1)
        assert op.mode == EventMode.LOCAL
        assert op.A_w[P, P] > 1.0, "Short squeeze must have A_pp > 1 (positive feedback)"


class TestApplyOperator:
    def test_apply_stock_split_shape(self):
        op = stock_split_operator(ratio=2.0, n=1)
        s = np.random.randn(5)
        result = op.apply(s, rng=np.random.default_rng(42))
        assert result.shape == (5,)

    def test_apply_deterministic_with_zero_sigma(self):
        """With Sigma=0, operator is fully deterministic."""
        op = stock_split_operator(ratio=2.0, n=1)
        op.Sigma_w = np.zeros((5, 5))
        s = np.ones(5)
        r1 = op.apply(s, rng=np.random.default_rng(1))
        r2 = op.apply(s, rng=np.random.default_rng(99))
        np.testing.assert_allclose(r1, r2)

    def test_apply_rate_change_full_universe(self):
        """Rate hike acts on full n×d state vector and lowers price components."""
        n, d = 3, 5
        op = rate_change_operator(change_bps=50, n=n)
        s = np.zeros(n * d)
        result = op.apply(s, rng=np.random.default_rng(42))
        assert result.shape == (n * d,)
        for i in range(n):
            assert result[i * d + P] < 0, f"Asset {i} log price should decrease after rate hike"

    def test_merger_reduces_state_size(self):
        """Merger of 2-asset universe produces 1-asset state."""
        n = 2
        op = merger_operator(acquirer_idx=0, target_idx=1, n=n)
        s = np.zeros(n * 5)
        s[P] = np.log(100)    # acquirer at $100
        s[5 + P] = np.log(50) # target at $50
        result = op.apply(s, rng=np.random.default_rng(7))
        assert result.shape == (5,)  # 1 combined entity

    def test_ipo_increases_state_size(self):
        """IPO adds one asset to state vector."""
        n = 3
        op = ipo_operator(ticker='X', ipo_price=50.0, n=n)
        s = np.zeros(n * 5)
        result = op.apply(s, rng=np.random.default_rng(13))
        assert result.shape == ((n + 1) * 5,)


class TestGroupoidComposition:
    def test_mode_i_composes_with_mode_i(self):
        """Two Mode I operators with same dimension compose cleanly."""
        op1 = stock_split_operator(ratio=2.0, n=1)
        op2 = stock_split_operator(ratio=3.0, n=1)
        comp = compose(op1, op2)
        assert comp is not None
        # For identity A: b_comp = b2 + b1
        np.testing.assert_allclose(comp.b_w[P], op1.b_w[P] + op2.b_w[P], rtol=1e-6)

    def test_mode_i_composes_with_mode_ii(self):
        """Mode I (n=3) composes with Mode II (n=3) — same state space."""
        op_macro = rate_change_operator(change_bps=25, n=3)
        op_local = earnings_shock_operator(surprise_pct=10.0, asset_idx=0, n=3)
        comp = compose(op_local, op_macro)   # macro first, then local
        assert comp.source_dim == 3
        assert comp.target_dim == 3

    def test_composition_dimension_mismatch_raises(self):
        """Composing operators with incompatible dimensions must raise ValueError."""
        op_merger = merger_operator(acquirer_idx=0, target_idx=1, n=2)  # 2→1 asset
        with pytest.raises(ValueError):
            compose(op_merger, op_merger)  # output (1 asset) can't feed merger (needs 2)

    def test_ipo_then_split_composition(self):
        """After IPO: n+1 asset universe. Split on that n+1 universe should compose cleanly."""
        n_init = 2
        op_ipo = ipo_operator(ticker='NEW', ipo_price=100.0, n=n_init)   # 2→3 assets
        op_split = stock_split_operator(ratio=2.0, asset_idx=0, n=3)     # 3→3 assets
        comp = compose(op_split, op_ipo)   # IPO then split
        assert comp.source_dim == n_init
        assert comp.target_dim == 3

    def test_composed_operator_inherits_source_target(self):
        """Composed operator takes source from inner (op2), target from outer (op1)."""
        op1 = stock_split_operator(ratio=2.0, n=2)
        op1.target_tickers = ['A', 'B']
        op1.source_tickers = ['A', 'B']
        op2 = dividend_operator(div_yield=0.02, asset_idx=0, n=2)
        op2.source_tickers = ['A', 'B']
        op2.target_tickers = ['A', 'B']
        comp = compose(op1, op2)
        assert comp.source_tickers == op2.source_tickers
        assert comp.target_tickers == op1.target_tickers

    def test_event_sequence_dimension_tracking(self):
        """event_sequence correctly tracks universe size through Mode III events."""
        n_init = 3
        d = 5
        s0 = np.zeros(n_init * d)
        s0[P] = np.log(100)

        operators = [
            rate_change_operator(change_bps=25, n=3),   # Mode II, 3→3
            ipo_operator(ticker='NEW', ipo_price=50.0, n=3),  # Mode III, 3→4
            bankruptcy_operator(asset_idx=1, n=4, recovery_rate=0.05),  # Mode III, 4→3
        ]

        final_state, log = event_sequence(operators, s0, rng=np.random.default_rng(2026))
        # 3 → 3 → 4 → 3: net unchanged
        assert final_state.shape == (n_init * d,)
        assert len(log) == len(operators)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
