"""
Dual noise decomposition from §III of the mathematical framework.

dS_t = μ dt + σ_τ dW_t^(τ)  [physical noise — continuous Brownian]
             + dJ_t^(η)       [behavioral noise — pure jump Lévy]

Physical noise: locally Gaussian, continuous paths
Behavioral noise: heavy-tailed jumps (Lévy measure ν^η ~ x^{-α}, α ∈ (1,2))

Cramér-Rao lower bound (Theorem III.1):
  Var(Ŝ_{t+h} - S_{t+h}) ≥ σ_τ² · h  +  λ_η · m₂^η · h

  where λ_η = jump intensity, m₂^η = ∫ z² ν^η(dz)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DualNoiseParams:
    sigma_tau: float        # physical noise volatility (σ_τ)
    lambda_eta: float       # behavioral jump intensity (λ_η)
    m2_eta: float           # second moment of jump size ∫z² ν^η(dz)
    alpha_levy: float = 1.5 # Lévy tail index ∈ (1, 2)

    @property
    def tau_t(self) -> float:
        """Composite temperature parameter from §III.5: τ_t = √(σ_τ² + λ_η · m₂^η)"""
        return np.sqrt(self.sigma_tau**2 + self.lambda_eta * self.m2_eta)

    @property
    def cramer_rao_bound(self, h: float = 1.0) -> float:
        """Prediction variance lower bound for horizon h (Theorem III.1)."""
        return (self.sigma_tau**2 + self.lambda_eta * self.m2_eta) * h


class DualNoiseCalibrator:
    """
    Calibrate (σ_τ, λ_η, m₂^η) from intraday returns using bipower variation.

    BPV estimator (Barndorff-Nielsen & Shephard 2004):
      BPV_t = (π/2) · Σ |r_{t,i}| · |r_{t,i-1}|  →  σ_τ²·h  (a.s., jumps excluded)

    Jump detection: Lee-Mykland test at α = 0.001 on 5-min bars.
    Jump sizes {z_k} → fit Lévy measure via MLE on |z_k| > threshold.
    """

    def __init__(self, alpha_lm: float = 0.001):
        self.alpha_lm = alpha_lm  # Lee-Mykland significance level

    def estimate_bpv(self, returns: np.ndarray) -> float:
        """Bipower variation estimate of integrated physical variance."""
        if len(returns) < 2:
            return 0.0
        bpv = (np.pi / 2) * np.sum(np.abs(returns[1:]) * np.abs(returns[:-1]))
        return bpv

    def detect_jumps(self, returns: np.ndarray, bpv: float, dt: float = 1/78) -> np.ndarray:
        """
        Lee-Mykland (2008) jump detection. Returns boolean mask of jump indices.
        dt = bar interval as fraction of day (e.g. 1/78 for 5-min bars).

        Test statistic: |r_t| / sigma_hat vs c_alpha,
        where sigma_hat = sqrt(BPV/n) is the per-bar volatility estimate.
        """
        if len(returns) == 0:
            return np.zeros(0, dtype=bool)
        sigma_hat = np.sqrt(bpv / len(returns))  # per-bar vol estimate
        critical = sigma_hat * self._lee_mykland_critical(len(returns))
        return np.abs(returns) > critical

    def calibrate(self, intraday_returns: np.ndarray, dt: float = 1/78) -> DualNoiseParams:
        bpv = self.estimate_bpv(intraday_returns)
        jump_mask = self.detect_jumps(intraday_returns, bpv, dt)
        jump_sizes = intraday_returns[jump_mask]

        sigma_tau = np.sqrt(bpv / len(intraday_returns))
        lambda_eta = jump_mask.sum() * dt if dt > 0 else 0.0
        m2_eta = float(np.mean(jump_sizes**2)) if len(jump_sizes) > 0 else 1e-6

        return DualNoiseParams(
            sigma_tau=sigma_tau,
            lambda_eta=lambda_eta,
            m2_eta=m2_eta,
        )

    @staticmethod
    def _lee_mykland_critical(n: int, alpha: float = 0.001) -> float:
        """Approximate critical value for Lee-Mykland test."""
        c = np.sqrt(2 * np.log(n))
        return c + (np.log(np.log(n)) + np.log(4 * np.pi) - 2 * np.log(-np.log(1 - alpha))) / (2 * c)
