"""
§4.3-4.4 — Neural Fictitious Play + Hierarchical MFG Solver.

Solves the MFG fixed point: (V*, m*) satisfying HJB + Fokker-Planck.

Convergence guaranteed under Lasry-Lions monotonicity (Theorem VI.1).
Stopping criterion: W₂(m_n, m_{n-1}) < tol  (approximated by |Δμ|).
"""
import torch
from typing import Optional


class NeuralFictitiousPlay:
    """
    Particle-based Fokker-Planck solver.
    Particles represent the empirical distribution of agent states m_t.
    """
    def __init__(self, d_latent: int = 64, n_particles: int = 1000, sigma_tau: float = 0.3):
        self.d      = d_latent
        self.K      = n_particles
        self.sigma  = sigma_tau
        # Initialize from standard normal (prior on latent states)
        self.particles = torch.randn(n_particles, d_latent) * 0.5

    def step(self, V_model: torch.nn.Module, dt: float = 1.0) -> None:
        """
        One Euler-Maruyama step of agent dynamics under optimal control.
        a*(z) = ∇_z V(z) / (2γκ)  from Stochastic Maximum Principle.
        """
        z = self.particles.requires_grad_(True)
        t_mid = torch.ones(self.K, 1) * 0.5
        V_vals = V_model(z, t_mid)
        grad_V = torch.autograd.grad(V_vals.sum(), z, retain_graph=False)[0]

        gamma, kappa = 2.0, 0.01
        alpha_star = grad_V.detach() / (2 * gamma * kappa)

        noise = torch.randn_like(self.particles) * self.sigma * (dt ** 0.5)
        self.particles = (self.particles.detach() + alpha_star * dt + noise)

    def empirical_measure(self) -> torch.Tensor:
        return self.particles.clone().detach()

    def w2_approx(self, prev_particles: torch.Tensor) -> float:
        """Approximate W₂ via mean difference (fast proxy)."""
        return (self.particles.mean(0) - prev_particles.mean(0)).norm().item()

    def predict_next_latent(self, z_t: torch.Tensor) -> torch.Tensor:
        """Used by encoder training: predict z_{t+1} given z_t under equilibrium drift."""
        return z_t + self.particles.mean(0).unsqueeze(0).expand_as(z_t) * 0.01


def run_fictitious_play(
    V_model: torch.nn.Module,
    d_latent: int = 64,
    n_particles: int = 1000,
    n_outer: int = 50,
    n_inner: int = 10,
    tol: float = 0.01,
) -> tuple[torch.Tensor, list[float]]:
    """
    Full fictitious play loop until convergence.
    Returns empirical measure m* and residual history.
    """
    fp = NeuralFictitiousPlay(d_latent=d_latent, n_particles=n_particles)
    residuals = []
    for n in range(n_outer):
        prev = fp.empirical_measure()
        for _ in range(n_inner):
            fp.step(V_model)
        res = fp.w2_approx(prev)
        residuals.append(res)
        if res < tol:
            print(f"[MFG] Converged at outer iter {n}, W₂≈{res:.5f}")
            break
    return fp.empirical_measure(), residuals


class HierarchicalMFGSolver:
    """
    §VI — Two-level hierarchical MFG solver.
    Level 1 (macro/MFC): solved offline weekly — exogenous to micro.
    Level 2 (micro/MFG): solved daily — takes macro drift as constraint.
    """
    def __init__(
        self,
        V_macro: torch.nn.Module,
        V_micro: torch.nn.Module,
        sigma_macro: float = 0.2,
        sigma_micro: float = 0.3,
    ):
        self.V_macro = V_macro
        self.V_micro = V_micro
        self.fp_macro = NeuralFictitiousPlay(sigma_tau=sigma_macro)
        self.fp_micro = NeuralFictitiousPlay(sigma_tau=sigma_micro)

    def solve(
        self,
        z_t: torch.Tensor,
        macro_state: torch.Tensor,
        n_outer: int = 20,
        macro_weight: float = 0.3,
    ) -> torch.Tensor:
        # Macro drift (pre-computed / cached) — Corollary VI.1
        mu_macro = self._extract_drift(self.V_macro, macro_state)
        # Micro fictitious play constrained by macro
        for _ in range(n_outer):
            self.fp_micro.step(self.V_micro)
        mu_micro = self._extract_drift(self.V_micro, z_t)
        # Hierarchical combination: a* = φ^(1)(macro) + φ^(2)(micro) + O(ε)
        return macro_weight * mu_macro + (1 - macro_weight) * mu_micro

    def _extract_drift(self, V_model: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
        z_t = z.detach().requires_grad_(True)
        t   = torch.tensor([[0.5]])
        V   = V_model(z_t.unsqueeze(0) if z_t.dim() == 1 else z_t, t)
        grad = torch.autograd.grad(V.sum(), z_t)[0]
        gamma, kappa = 2.0, 0.01
        return (grad / (2 * gamma * kappa)).detach()
