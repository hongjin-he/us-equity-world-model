"""
Financial event operators from §IV of the mathematical framework.

T_w(s) = A_w · s + b_w + Σ_w ε_w,  ε_w ~ N(0, I)

Three algebraic modes:
  Mode I  — Local endomorphism: single asset, dimension-preserving (stock split, secondary offering)
  Mode II — Global tensor product: all assets simultaneously (rate hike, systemic crisis)
  Mode III— Pairwise morphism: non-square operator (M&A merger → two → one; spin-off → one → two)

Mode III events change n (number of assets) → algebra must be a Groupoid, not a Semigroup.
"""
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EventMode(Enum):
    LOCAL = "I"       # single asset, dimension-preserving
    GLOBAL = "II"     # all assets simultaneously
    PAIRWISE = "III"  # changes number of assets (M&A, spin-off)


@dataclass
class EventOperator:
    """
    Affine event operator: T_w(s) = A_w · s + b_w + Σ_w ε_w
    """
    name: str
    mode: EventMode
    A_w: np.ndarray          # transformation matrix
    b_w: np.ndarray          # additive shift
    Sigma_w: np.ndarray      # noise covariance
    source_tickers: list     # input asset(s)
    target_tickers: list     # output asset(s) (may differ for Mode III)

    def apply(self, s: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Apply operator to state vector s."""
        noise_rng = rng or np.random.default_rng()
        eps = noise_rng.standard_normal(self.b_w.shape)
        return self.A_w @ s + self.b_w + self.Sigma_w @ eps


# ── Mode I operators ───────────────────────────────────────────────────────────

def stock_split_operator(ratio: float, d: int = 5) -> EventOperator:
    """
    k-for-1 split: p_t → p_t - log(k), κ_t → κ_t + log(k), others unchanged.
    """
    A = np.eye(d)
    b = np.zeros(d)
    A[0, 0] = 1.0   # p: subtract log(ratio) via b
    A[3, 3] = 1.0   # κ: add log(ratio) via b
    b[0] = -np.log(ratio)
    b[3] = +np.log(ratio)
    return EventOperator(
        name=f"stock_split_{ratio}x",
        mode=EventMode.LOCAL,
        A_w=A, b_w=b,
        Sigma_w=np.zeros((d, d)),
        source_tickers=[], target_tickers=[],
    )


def secondary_offering_operator(dilution_frac: float, d: int = 5) -> EventOperator:
    """
    Secondary offering: κ_t increases, p_t slightly decreases (dilution).
    """
    A = np.eye(d)
    b = np.zeros(d)
    b[0] = np.log(1 - dilution_frac * 0.5)     # price impact ≈ half dilution
    b[3] = np.log(1 + dilution_frac)            # shares outstanding up
    return EventOperator(
        name=f"secondary_offering_{dilution_frac:.0%}",
        mode=EventMode.LOCAL,
        A_w=A, b_w=b,
        Sigma_w=np.diag([0.02, 0, 0, 0.01, 0]),
        source_tickers=[], target_tickers=[],
    )


# ── Mode II operators ──────────────────────────────────────────────────────────

def rate_hike_operator(hike_bps: float, n_assets: int, d: int = 5) -> EventOperator:
    """
    Fed rate hike: correlated downward price shock across all assets.
    Size of shock = f(hike_bps, asset beta to rates).
    """
    total_dim = n_assets * d
    A = np.eye(total_dim)
    b = np.zeros(total_dim)
    # Apply price shock to every asset's p_t dimension (indices 0, 5, 10, ...)
    for i in range(n_assets):
        b[i * d + 0] = -hike_bps / 10000 * 2.5   # duration-style impact
        b[i * d + 2] = +hike_bps / 10000 * 0.5   # leverage up (financing cost)
    return EventOperator(
        name=f"rate_hike_{hike_bps}bps",
        mode=EventMode.GLOBAL,
        A_w=A, b_w=b,
        Sigma_w=np.eye(total_dim) * (hike_bps / 10000 * 0.3),
        source_tickers=[], target_tickers=[],
    )


# ── Mode III operators (Groupoid morphisms) ────────────────────────────────────

def merger_operator(d: int = 5) -> EventOperator:
    """
    M&A: two assets → one combined asset.
    A_w is (d × 2d): maps (s_acquirer, s_target) → s_combined
    """
    A = np.zeros((d, 2 * d))
    # Combined price ≈ acquirer price (plus premium captured in b)
    A[0, 0] = 0.6; A[0, d + 0] = 0.4
    # Combined volume ≈ sum
    A[1, 1] = 1.0; A[1, d + 1] = 1.0
    # Combined leverage: weighted average
    A[2, 2] = 0.6; A[2, d + 2] = 0.4
    # Combined κ: log(κ_a + κ_t) ≈ max + log(1 + exp(min-max))
    A[3, 3] = 0.5; A[3, d + 3] = 0.5
    # Disclosure: combined disclosure
    A[4, 4] = 0.7; A[4, d + 4] = 0.3
    b = np.zeros(d)
    b[0] = 0.05  # typical M&A premium ~5%
    return EventOperator(
        name="merger_acquisition",
        mode=EventMode.PAIRWISE,
        A_w=A, b_w=b,
        Sigma_w=np.eye(d) * 0.05,
        source_tickers=[], target_tickers=[],
    )


def spinoff_operator(d: int = 5) -> EventOperator:
    """
    Spin-off: one asset → two assets (parent + child).
    A_w is (2d × d): maps s_parent → (s_parent_new, s_child)
    """
    A = np.zeros((2 * d, d))
    # Parent retains ~70%
    A[0, 0] = 1.0; A[3, 3] = 1.0
    A[1, 1] = 0.7; A[2, 2] = 1.0; A[4, 4] = 0.6
    # Child gets ~30% carved out
    A[d + 0, 0] = 1.0; A[d + 3, 3] = 1.0
    A[d + 1, 1] = 0.3; A[d + 2, 2] = 1.0; A[d + 4, 4] = 0.4
    b = np.zeros(2 * d)
    b[0] = -0.1   # parent: price haircut at spin-off
    b[d] = -0.3   # child: initially trades at discount
    return EventOperator(
        name="spinoff",
        mode=EventMode.PAIRWISE,
        A_w=A, b_w=b,
        Sigma_w=np.eye(2 * d) * 0.08,
        source_tickers=[], target_tickers=[],
    )
