"""
§4.2 — HJB solver via Deep Galerkin Method (DGM) in JAX.

Solves: -∂V/∂t + H(z, ∇V, m*) = 0
where H = ∇V · m* + |∇V|²/(4γκ) + (σ²/2)·tr(∇²V)

Optimal control from §II.4 (Stochastic Maximum Principle):
  a*(z) = ∇_z V* / (2γκ)

Reference: Sirignano & Spiliopoulos (2018), DGM paper.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable


class DGMNet(nn.Module):
    """
    DGM architecture for V(z, t).
    DGM gating prevents vanishing gradients in deep networks.
    """
    features: int = 256
    layers: int = 4

    @nn.compact
    def __call__(self, z: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # z: [batch, d_latent], t: [batch, 1]
        x = jnp.concatenate([z, t], axis=-1)
        x = nn.Dense(self.features)(x)
        x = nn.tanh(x)
        for _ in range(self.layers):
            zt = jnp.concatenate([z, t], axis=-1)
            S      = nn.Dense(self.features)(x)
            Z_gate = nn.sigmoid(nn.Dense(self.features)(zt) + nn.Dense(self.features)(x))
            G_gate = nn.sigmoid(nn.Dense(self.features)(zt) + nn.Dense(self.features)(x))
            H_gate = nn.tanh(nn.Dense(self.features)(zt)    + nn.Dense(self.features)(x))
            x = (1 - G_gate) * H_gate + Z_gate * x + (1 - Z_gate) * S
        return nn.Dense(1)(x).squeeze(-1)


def hjb_residual(
    V_fn: Callable,
    params: dict,
    z: jnp.ndarray,
    t: jnp.ndarray,
    mu_star: jnp.ndarray,
    sigma_tau: float = 0.3,
    gamma: float = 2.0,
    kappa: float = 0.01,
) -> jnp.ndarray:
    """
    Compute HJB residual² at a single (z, t) point.
    Used as the training loss for the DGM net (residual minimization).
    """
    def V(z_): return V_fn.apply(params, z_, t)
    def Vt(t_): return V_fn.apply(params, z, t_)

    grad_V = jax.grad(V)(z)
    dV_dt  = jax.grad(Vt)(t).squeeze()
    # Diffusion term: (σ²/2) tr(∇²V) — diagonal Hessian approximation
    hess_diag = jax.grad(lambda z_: jax.grad(V)(z_).sum())(z)
    diffusion  = 0.5 * sigma_tau**2 * hess_diag.sum()
    # Hamiltonian: H = ∇V · m* + |∇V|²/(4γκ)
    H_value = jnp.dot(grad_V, mu_star) + jnp.dot(grad_V, grad_V) / (4 * gamma * kappa)

    residual = -dV_dt + H_value + diffusion
    return residual**2


@jax.jit
def dgm_loss(V_fn, params, z_batch, t_batch, mu_star_batch, sigma_tau=0.3):
    """Batch HJB residual loss (Sobol quasi-random sampling recommended)."""
    residuals = jax.vmap(
        lambda z, t, mu: hjb_residual(V_fn, params, z, t, mu, sigma_tau)
    )(z_batch, t_batch, mu_star_batch)
    return residuals.mean()
