"""
Feature engineering for the MicroWorld encoder input.

Transforms raw OHLCV + macro + sentiment data into a normalized feature tensor
suitable for the Transformer VAE encoder E.

Feature groups:
  - Price features: log returns, volatility estimates, momentum signals
  - BPV features: bipower variation, jump ratio, Cramér-Rao bound estimate
  - Cross-sectional features: z-score rank, information ratio, relative strength
  - Regime features: rolling correlation, dispersion, VIX proxy
  - Macro features: rate differential, yield curve slope, FX stress
"""
import numpy as np
from typing import Optional


# ── Price features ─────────────────────────────────────────────────────────

def log_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns: r_t = log(S_t / S_{t-1})."""
    return np.diff(np.log(prices + 1e-10), axis=0)


def rolling_vol(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Annualised rolling volatility (standard deviation of log returns × √252)."""
    n = len(returns)
    vol = np.full(n, np.nan)
    for i in range(window - 1, n):
        vol[i] = np.std(returns[i - window + 1:i + 1]) * np.sqrt(252)
    return vol


def momentum(returns: np.ndarray, window: int = 63) -> np.ndarray:
    """Cumulative log return over window (skip most recent week to avoid short-term reversal)."""
    skip = 5
    n = len(returns)
    mom = np.full(n, np.nan)
    for i in range(window + skip - 1, n):
        mom[i] = np.sum(returns[i - window - skip + 1:i - skip + 1])
    return mom


def short_term_reversal(returns: np.ndarray, window: int = 5) -> np.ndarray:
    """1-week return reversal signal."""
    n = len(returns)
    rev = np.full(n, np.nan)
    for i in range(window - 1, n):
        rev[i] = -np.sum(returns[i - window + 1:i + 1])  # negative sign: reversal
    return rev


# ── BPV / dual-noise features ───────────────────────────────────────────────

def bipower_variation(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Rolling bipower variation (Barndorff-Nielsen & Shephard 2004)."""
    n = len(returns)
    bpv = np.full(n, np.nan)
    for i in range(window, n):
        r = returns[i - window:i]
        bpv[i] = (np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
    return bpv


def quadratic_variation(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Rolling quadratic variation (realized variance)."""
    n = len(returns)
    qv = np.full(n, np.nan)
    for i in range(window, n):
        qv[i] = np.sum(returns[i - window:i] ** 2)
    return qv


def jump_ratio(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Jump share of total variance: max(QV - BPV, 0) / QV."""
    bpv = bipower_variation(returns, window)
    qv = quadratic_variation(returns, window)
    jump_var = np.maximum(qv - bpv, 0.0)
    return np.where(qv > 1e-12, jump_var / qv, 0.0)


def cramer_rao_estimate(returns: np.ndarray, jump_intensity: float,
                         window: int = 21) -> np.ndarray:
    """
    Cramér-Rao lower bound estimate: σ²_τ/Δt + ν_η.
    Here Δt = 1 day, so σ²_τ / Δt = rolling daily variance from BPV.
    """
    bpv = bipower_variation(returns, window)
    sigma_tau_sq = bpv / window  # per-day physical variance estimate
    return sigma_tau_sq + jump_intensity


# ── Cross-sectional features ────────────────────────────────────────────────

def cross_sectional_zscore(signal: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Standardize signal cross-sectionally at each timestep.
    Input: (T, N) array. Output: (T, N) z-scored.
    """
    mu = np.nanmean(signal, axis=1, keepdims=True)
    std = np.nanstd(signal, axis=1, keepdims=True)
    return (signal - mu) / (std + epsilon)


def cross_sectional_rank(signal: np.ndarray) -> np.ndarray:
    """
    Cross-sectional rank normalization to [-0.5, 0.5].
    Input: (T, N). Output: rank-normalized (T, N).
    """
    T, N = signal.shape
    ranks = np.zeros_like(signal)
    for t in range(T):
        row = signal[t]
        valid = ~np.isnan(row)
        if valid.sum() > 0:
            idx = np.argsort(np.argsort(row[valid]))
            n_valid = valid.sum()
            ranks[t, valid] = (idx / (n_valid - 1) - 0.5) if n_valid > 1 else 0.0
    return ranks


def information_ratio(returns: np.ndarray, signal: np.ndarray,
                       window: int = 63) -> np.ndarray:
    """
    Rolling information ratio of signal vs returns.
    IC = corr(signal_t, return_{t+1}) over window.
    """
    n = len(returns)
    ir = np.full(n, np.nan)
    for i in range(window, n - 1):
        r = returns[i - window + 1:i + 1]
        s = signal[i - window:i]
        if np.std(s) > 1e-10 and np.std(r) > 1e-10:
            ic = np.corrcoef(s, r)[0, 1]
            ic_series = np.array([np.corrcoef(signal[j-1], returns[j])[0, 1]
                                  for j in range(i - window + 1, i + 1)
                                  if np.ndim(signal[j-1]) == 0])
            if len(ic_series) > 1:
                ir[i] = np.mean(ic_series) / (np.std(ic_series) + 1e-10)
    return ir


# ── Regime features ─────────────────────────────────────────────────────────

def rolling_correlation(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Average pairwise correlation across assets (regime indicator).
    Input: (T, N) returns. Output: (T,) average correlation.
    """
    T, N = returns.shape
    avg_corr = np.full(T, np.nan)
    for t in range(window, T):
        R = returns[t - window:t]
        C = np.corrcoef(R.T)
        # Average off-diagonal
        mask = ~np.eye(N, dtype=bool)
        avg_corr[t] = np.nanmean(C[mask])
    return avg_corr


def cross_sectional_dispersion(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Cross-sectional standard deviation of returns (stock-picking opportunity proxy).
    High dispersion → more alpha available for stock selectors.
    """
    T = len(returns)
    disp = np.full(T, np.nan)
    if returns.ndim == 1:
        return disp
    for t in range(window, T):
        disp[t] = np.std(returns[t])
    return disp


# ── Full feature matrix ──────────────────────────────────────────────────────

def build_feature_matrix(prices: np.ndarray,
                          macro: Optional[np.ndarray] = None,
                          window_short: int = 21,
                          window_long: int = 63) -> np.ndarray:
    """
    Build the full feature matrix for the Encoder E input.

    Args:
        prices: (T, N) daily close prices
        macro: (T, M) macro features (optional)
        window_short: short rolling window (days)
        window_long: long rolling window (days)

    Returns:
        features: (T, N, F) feature tensor per asset per day
                  F = [log_ret, vol_short, vol_long, mom, str_rev,
                       bpv, jump_ratio, cs_zscore_ret, cs_rank_ret]
    """
    T, N = prices.shape
    ret = log_returns(prices)  # (T-1, N)

    # Pad returns to length T
    ret_padded = np.vstack([np.full((1, N), np.nan), ret])  # (T, N)

    feature_list = []

    # Per-asset features
    for i in range(N):
        r = ret_padded[:, i]
        vol_s = rolling_vol(r, window_short)
        vol_l = rolling_vol(r, window_long)
        mom_s = momentum(r, window_short)
        mom_l = momentum(r, window_long)
        str_rev = short_term_reversal(r)
        bpv_feat = bipower_variation(r, window_short)
        jr = jump_ratio(r, window_short)

        asset_feats = np.stack([r, vol_s, vol_l, mom_s, mom_l, str_rev, bpv_feat, jr], axis=-1)
        feature_list.append(asset_feats)

    features = np.stack(feature_list, axis=1)  # (T, N, F)

    # Cross-sectional features (add to existing)
    cs_z = cross_sectional_zscore(ret_padded)[:, :, np.newaxis]
    cs_r = cross_sectional_rank(ret_padded)[:, :, np.newaxis]
    features = np.concatenate([features, cs_z, cs_r], axis=-1)

    # Replace NaN with 0 (encoder can learn to ignore padding)
    features = np.nan_to_num(features, nan=0.0)

    return features
