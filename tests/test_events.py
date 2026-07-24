"""
Tests for groupoid algebra of financial events (events/operators.py).
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from events.operators import (
    EventMode, EventOperator, stock_split_operator, rate_hike_operator,
    merger_operator, ipo_operator, compose,
)


class TestEventModes:
    def test_stock_split_mode_local(self):
        op = stock_split_operator(ratio=2.0, d=5)
        assert op.mode == EventMode.LOCAL

    def test_stock_split_log_price_decreases(self):
        """2:1 split halves price → log_price shift = -log(2)."""
        op = stock_split_operator(ratio=2.0, d=5)
        np.testing.assert_allclose(op.b_w[0], -np.log(2.0), rtol=1e-6)

    def test_stock_split_volume_increases(self):
        """2:1 split doubles volume → log_volume shift = +log(2)."""
        op = stock_split_operator(ratio=2.0, d=5)
        np.testing.assert_allclose(op.b_w[3], np.log(2.0), rtol=1e-6)

    def test_rate_hike_mode_global(self):
        op = rate_hike_operator(hike_bps=25, n_assets=4, d=5)
        assert op.mode == EventMode.GLOBAL

    def test_rate_hike_lowers_first_asset_price(self):
        """Rate hike should lower log prices (b[0] < 0)."""
        op = rate_hike_operator(hike_bps=25, n_assets=4, d=5)
        assert op.b_w[0] < 0, "Rate hike should lower log prices (b[0] must be negative)"

    def test_merger_mode_pairwise(self):
        op = merger_operator(d=5)
        assert op.mode == EventMode.PAIRWISE

    def test_merger_output_smaller_than_input(self):
        """Merger: 2 assets → 1. A_w maps 2d → d."""
        d = 5
        op = merger_operator(d=d)
        assert op.A_w.shape == (d, 2 * d)

    def test_ipo_mode_pairwise(self):
        op = ipo_operator(ticker='NEWCO', d=5, n_assets=4)
        assert op.mode == EventMode.PAIRWISE

    def test_ipo_increases_universe(self):
        """IPO: n assets → n+1 assets in target."""
        n, d = 4, 5
        op = ipo_operator(ticker='NEWCO', d=d, n_assets=n)
        assert len(op.target_tickers) == n + 1
        assert len(op.source_tickers) == n


class TestApplyOperator:
    def test_apply_stock_split_shape(self):
        d = 5
        op = stock_split_operator(ratio=2.0, d=d)
        s = np.random.randn(d)
        result = op.apply(s, rng=np.random.default_rng(42))
        assert result.shape == (d,)

    def test_apply_deterministic_with_zero_sigma(self):
        """With Sigma=0, operator is deterministic."""
        d = 5
        op = stock_split_operator(ratio=2.0, d=d)
        op.Sigma_w = np.zeros((d, d))
        s = np.ones(d)
        r1 = op.apply(s, rng=np.random.default_rng(1))
        r2 = op.apply(s, rng=np.random.default_rng(99))
        np.testing.assert_allclose(r1, r2)

    def test_apply_rate_hike_full_universe(self):
        """Rate hike acts on full n_assets×d state vector."""
        n, d = 3, 5
        op = rate_hike_operator(hike_bps=50, n_assets=n, d=d)
        s = np.zeros(n * d)
        result = op.apply(s, rng=np.random.default_rng(42))
        assert result.shape == (n * d,)
        # Price components (indices 0, 5, 10) should decrease
        for i in range(n):
            assert result[i * d] < 0, f"Asset {i} price should decrease after rate hike"


class TestGroupoidComposition:
    def test_mode_i_composes_with_mode_i_same_dim(self):
        """Two Mode I operators with same dimension can compose."""
        op1 = stock_split_operator(ratio=2.0, d=5)
        op2 = stock_split_operator(ratio=3.0, d=5)
        result = compose(op1, op2)
        assert result is not None
        # Composed shift: b_comp = A1 @ b2 + b1
        # For identity A: b_comp = b2 + b1
        np.testing.assert_allclose(result.b_w[0], op1.b_w[0] + op2.b_w[0], rtol=1e-6)

    def test_composition_dimension_mismatch_raises(self):
        """Composing operators with incompatible dimensions should raise ValueError."""
        d = 5
        op_merger = merger_operator(d=d)  # maps 2d → d
        op_split = stock_split_operator(ratio=2.0, d=d)  # maps d → d
        # op_split ∘ op_merger: output of merger (d) feeds into split (d) — this works
        # op_merger ∘ op_merger: output of merger (d) feeds into merger (2d input) — FAILS
        import pytest
        with pytest.raises(ValueError):
            compose(op_merger, op_merger)

    def test_composed_operator_source_target(self):
        """Composed operator inherits source from op2, target from op1."""
        op1 = stock_split_operator(ratio=2.0, d=5)
        op1.target_tickers = ['A', 'B']
        op2 = stock_split_operator(ratio=3.0, d=5)
        op2.source_tickers = ['A', 'B']
        op2.target_tickers = ['A', 'B']
        op1.source_tickers = ['A', 'B']
        result = compose(op1, op2)
        assert result.source_tickers == op2.source_tickers
        assert result.target_tickers == op1.target_tickers


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
