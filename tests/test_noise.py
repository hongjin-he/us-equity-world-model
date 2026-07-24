"""
Tests for dual-noise decomposition (state/noise.py).
Validates BPV estimator, Lee-Mykland detection, and calibration.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from state.noise import DualNoiseCalibrator, DualNoiseParams


class TestBiPowerVariation:
    def test_pure_brownian_bpv_equals_qv(self):
        """Without jumps, BPV ≈ QV (both estimate integrated variance)."""
        rng = np.random.default_rng(42)
        sigma = 0.01
        n = 1000
        returns = sigma * rng.standard_normal(n)

        cal = DualNoiseCalibrator()
        bpv = cal.estimate_bpv(returns)
        qv = np.sum(returns ** 2)

        # BPV should be close to QV for pure Brownian (ratio near 1)
        ratio = bpv / qv
        assert 0.85 < ratio < 1.15, f"BPV/QV ratio {ratio:.3f} outside [0.85, 1.15]"

    def test_bpv_lower_than_qv_with_jumps(self):
        """With jumps, QV > BPV (jumps inflate QV but not BPV)."""
        rng = np.random.default_rng(42)
        sigma = 0.01
        n = 500
        returns = sigma * rng.standard_normal(n)

        # Add large jumps
        jump_indices = rng.choice(n, size=10, replace=False)
        returns[jump_indices] += rng.choice([-0.05, 0.05], size=10)

        cal = DualNoiseCalibrator()
        bpv = cal.estimate_bpv(returns)
        qv = np.sum(returns ** 2)

        assert bpv < qv, f"BPV ({bpv:.6f}) should be < QV ({qv:.6f}) with jumps"

    def test_bpv_empty_returns(self):
        """Edge case: empty or single-element array."""
        cal = DualNoiseCalibrator()
        assert cal.estimate_bpv(np.array([])) == 0.0
        assert cal.estimate_bpv(np.array([0.01])) == 0.0

    def test_bpv_scales_with_variance(self):
        """BPV should scale linearly with σ²."""
        rng = np.random.default_rng(42)
        n = 2000
        r1 = 0.01 * rng.standard_normal(n)
        r2 = 0.02 * rng.standard_normal(n)

        cal = DualNoiseCalibrator()
        bpv1 = cal.estimate_bpv(r1)
        bpv2 = cal.estimate_bpv(r2)

        # BPV2 should be ~4x BPV1 (variance ratio = (0.02/0.01)^2 = 4)
        ratio = bpv2 / bpv1
        assert 3.0 < ratio < 5.0, f"BPV scaling ratio {ratio:.2f} outside [3, 5]"


class TestJumpDetection:
    def test_no_false_positives_in_pure_brownian(self):
        """Jump detector should rarely fire on pure Brownian paths."""
        rng = np.random.default_rng(42)
        sigma = 0.015
        n = 390  # ~1 day of 1-minute returns
        returns = sigma / np.sqrt(390) * rng.standard_normal(n)

        cal = DualNoiseCalibrator(alpha_lm=0.001)
        bpv = cal.estimate_bpv(returns)
        jumps = cal.detect_jumps(returns, bpv, dt=1/390)

        false_positive_rate = jumps.sum() / n
        assert false_positive_rate < 0.05, \
            f"False positive rate {false_positive_rate:.2%} too high (expected < 5%)"

    def test_detects_obvious_jumps(self):
        """Detector should find 10-sigma moves."""
        rng = np.random.default_rng(42)
        sigma = 0.01
        n = 500
        returns = sigma * rng.standard_normal(n)

        # Inject obvious jumps
        jump_indices = [100, 200, 350]
        for idx in jump_indices:
            returns[idx] = 0.15  # ~15σ move

        cal = DualNoiseCalibrator()
        bpv = cal.estimate_bpv(returns)
        jumps = cal.detect_jumps(returns, bpv, dt=1/78)

        detected = sum(jumps[idx] for idx in jump_indices)
        assert detected >= 2, f"Only {detected}/3 obvious jumps detected"


class TestCalibration:
    def test_calibration_returns_valid_params(self):
        """Calibration should return non-negative, finite parameters."""
        rng = np.random.default_rng(42)
        returns = 0.01 * rng.standard_normal(500)
        returns[50] = 0.08  # one jump

        cal = DualNoiseCalibrator()
        params = cal.calibrate(returns, dt=1/78)

        assert isinstance(params, DualNoiseParams)
        assert params.sigma_tau >= 0, "sigma_tau should be non-negative"
        assert np.isfinite(params.sigma_tau), "sigma_tau should be finite"
        assert params.lambda_eta >= 0, "lambda_eta should be non-negative"
        assert params.m2_eta > 0, "m2_eta should be positive"

    def test_tau_t_positive(self):
        """Composite temperature parameter τ_t should always be positive."""
        params = DualNoiseParams(sigma_tau=0.01, lambda_eta=2.0, m2_eta=0.001)
        assert params.tau_t > 0

    def test_higher_vol_gives_higher_sigma_tau(self):
        """Higher volatility data should give higher estimated sigma_tau."""
        rng = np.random.default_rng(42)
        n = 1000

        returns_low = 0.005 * rng.standard_normal(n)
        returns_high = 0.020 * rng.standard_normal(n)

        cal = DualNoiseCalibrator()
        params_low = cal.calibrate(returns_low)
        params_high = cal.calibrate(returns_high)

        assert params_high.sigma_tau > params_low.sigma_tau, \
            "Higher vol should give higher sigma_tau"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
