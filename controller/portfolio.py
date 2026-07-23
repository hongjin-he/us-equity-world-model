"""
§5.1 — Controller C: latent gradient → portfolio weights with CVaR constraints.

Optimal latent action: a*(z) = ∇_z V* / (2γκ)
Then decode to asset space and project onto risk feasible set.

Risk constraints (from §5.1):
  |w|₁ ≤ leverage_limit         (gross leverage)
  CVaR_5%(r_portfolio) ≥ -cvar_limit   (tail risk)
"""
import numpy as np
import torch
from scipy.optimize import minimize


class EGameCController:
    def __init__(
        self,
        decoder,
        gamma: float = 2.0,
        kappa: float = 0.01,
        leverage_limit: float = 2.0,
        cvar_limit: float = 0.05,
    ):
        self.decoder   = decoder
        self.gamma     = gamma
        self.kappa     = kappa
        self.lev_lim   = leverage_limit
        self.cvar_lim  = cvar_limit
        self.prev_weights: np.ndarray | None = None

    def compute_weights(
        self,
        z_t: torch.Tensor,
        V_model: torch.nn.Module,
        return_scenarios: np.ndarray,
    ) -> np.ndarray:
        """
        z_t:              [d_latent] current latent state
        V_model:          DGMNet callable
        return_scenarios: [K, N_assets] Monte Carlo return scenarios for CVaR
        """
        # Optimal latent action
        z_req = z_t.detach().requires_grad_(True)
        V_val = V_model(z_req.unsqueeze(0), torch.tensor([[0.5]]))
        grad_V = torch.autograd.grad(V_val.sum(), z_req)[0]
        alpha_star = (grad_V / (2 * self.gamma * self.kappa)).detach().numpy()

        # Decode to asset weight space
        raw_weights = self.decoder(alpha_star)

        # Project onto risk feasible set
        w_opt = self._project_risk_constraints(raw_weights, return_scenarios)
        self.prev_weights = w_opt
        return w_opt

    def _project_risk_constraints(
        self, w_init: np.ndarray, scenarios: np.ndarray
    ) -> np.ndarray:
        def objective(w):
            return 0.5 * np.dot(w - w_init, w - w_init)

        def cvar_constraint(w):
            port_returns = scenarios @ w
            var_5  = np.percentile(port_returns, 5)
            cvar_5 = port_returns[port_returns <= var_5].mean()
            return cvar_5 + self.cvar_lim  # must be >= 0

        result = minimize(
            objective,
            w_init,
            constraints=[
                {"type": "ineq", "fun": cvar_constraint},
                {"type": "ineq", "fun": lambda w: self.lev_lim - np.abs(w).sum()},
            ],
            method="SLSQP",
            options={"maxiter": 200, "ftol": 1e-8},
        )
        if not result.success:
            # Fallback: scale down if infeasible
            return w_init * 0.5
        return result.x

    def generate_orders(
        self,
        target_weights: np.ndarray,
        current_positions: np.ndarray,
        prices: np.ndarray,
    ) -> np.ndarray:
        """Return share deltas (positive = buy, negative = sell)."""
        portfolio_value = float(np.sum(current_positions * prices))
        target_shares   = np.floor(target_weights * portfolio_value / np.maximum(prices, 1e-8))
        return target_shares - current_positions
