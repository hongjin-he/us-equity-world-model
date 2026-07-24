"""
Tests for feature engineering (data/features/__init__.py).
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.features import (
    log_returns, rolling_vol, momentum, bipower_variation,
    quadratic_variation, jump_ratio, cross_sectional_zscore,
    cross_sectional_rank, build_feature_matrix,
)


class TestLogReturns:
    def test_shape(self):
        prices = np.random.rand(100, 5) + 1
        ret = log_returns(prices)
        assert ret.shape == (99, 5)

    def test_values(self):
        prices = np.array([[100.0], [110.0], [99.0]])
        ret = log_returns(prices)
        np.testing.assert_allclose(ret[0, 0], np.log(110/100), rtol=1e-6)
        np.testing.assert_allclose(ret[1, 0], np.log(99/110), rtol=1e-6)

    def test_zero_change_gives_zero_return(self):
        prices = np.array([[100.0], [100.0], [100.0]])
        ret = log_returns(prices)
        np.testing.assert_allclose(ret, 0.0, atol=1e-10)


class TestRollingVol:
    def test_shape_preserved(self):
        returns = np.random.randn(200)
        vol = rolling_vol(returns, window=21)
        assert vol.shape == (200,)

    def test_nan_for_initial_window(self):
        returns = np.random.randn(50)
        vol = rolling_vol(returns, window=21)
        assert np.all(np.isnan(vol[:20]))
        assert not np.isnan(vol[20])

    def test_higher_vol_data_gives_higher_vol(self):
        rng = np.random.default_rng(42)
        low_vol = 0.005 * rng.standard_normal(100)
        high_vol = 0.030 * rng.standard_normal(100)
        v_low = np.nanmean(rolling_vol(low_vol, 21))
        v_high = np.nanmean(rolling_vol(high_vol, 21))
        assert v_high > v_low


class TestBiPowerVariation:
    def test_shape(self):
        returns = np.random.randn(100)
        bpv = bipower_variation(returns, window=21)
        assert bpv.shape == (100,)

    def test_bpv_less_than_qv_with_jumps(self):
        rng = np.random.default_rng(42)
        returns = 0.01 * rng.standard_normal(200)
        returns[100] = 0.10  # big jump
        bpv = bipower_variation(returns, window=21)
        qv = quadratic_variation(returns, window=21)
        # At and after jump: QV should exceed BPV
        assert qv[110] > bpv[110]

    def test_jump_ratio_bounded(self):
        returns = np.random.randn(200) * 0.01
        jr = jump_ratio(returns, window=21)
        valid = jr[~np.isnan(jr)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0 + 1e-10)


class TestCrossSectional:
    def test_zscore_zero_mean(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal((50, 10))
        z = cross_sectional_zscore(signal)
        row_means = np.nanmean(z, axis=1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-10)

    def test_rank_bounds(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal((50, 10))
        r = cross_sectional_rank(signal)
        assert np.all(r >= -0.5 - 1e-9)
        assert np.all(r <= 0.5 + 1e-9)

    def test_rank_monotone_within_row(self):
        signal = np.array([[3.0, 1.0, 2.0], [0.5, -1.0, 2.0]])
        r = cross_sectional_rank(signal)
        # Highest value should have highest rank
        assert r[0, 0] > r[0, 2] > r[0, 1]


class TestBuildFeatureMatrix:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        T, N = 100, 5
        prices = 100 * np.exp(np.cumsum(rng.standard_normal((T, N)) * 0.01, axis=0))
        features = build_feature_matrix(prices, window_short=10, window_long=30)
        assert features.shape[0] == T
        assert features.shape[1] == N
        assert features.shape[2] >= 8  # at least 8 feature channels

    def test_no_nan_in_output(self):
        rng = np.random.default_rng(42)
        T, N = 100, 5
        prices = 100 * np.exp(np.cumsum(rng.standard_normal((T, N)) * 0.01, axis=0))
        features = build_feature_matrix(prices, window_short=10, window_long=30)
        assert not np.any(np.isnan(features)), "Feature matrix should not contain NaN"
        assert not np.any(np.isinf(features)), "Feature matrix should not contain Inf"

    def test_single_asset(self):
        """Should handle single-asset case."""
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal((80, 1)) * 0.01, axis=0))
        features = build_feature_matrix(prices, window_short=10, window_long=30)
        assert features.shape[1] == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
