"""
§3.2-3.3 — Encoder training: three-term loss + EWC continual learning.

Loss = Recon + β·KL + λ·PredCoupling

Training schedule:
  Phase 1 (Warmup,  50ep): λ=0.0, β=0.1→1.0   — learn reconstruction
  Phase 2 (Coupling, 100ep): λ=0.3             — couple with Game module
  Phase 3 (Finetune, 20ep): λ=0.5, recent data — adapt to recent regime
  Online  (Daily,    5ep):  rolling 2yr window
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class EncoderLossWeights:
    beta: float = 1.0   # KL weight
    lam: float = 0.3    # predictive coupling weight


def encoder_loss(
    encoder,
    decoder,
    game_module,
    batch: tuple,
    weights: EncoderLossWeights = EncoderLossWeights(),
) -> tuple[torch.Tensor, dict]:
    x_assets, x_macro, x_news, x_next = batch
    # Encode current state
    mu, logvar = encoder(x_assets, x_macro, x_news)
    z_t = encoder.sample(mu, logvar)
    # Term 1: Reconstruction
    x_hat  = decoder(z_t)
    recon  = F.mse_loss(x_hat, x_assets)
    # Term 2: KL divergence to N(0, I)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # Term 3: Predictive coupling (joint with Game module)
    z_next_pred = game_module.predict_next_latent(z_t)
    mu_next, _  = encoder(*x_next)
    pred_loss   = F.mse_loss(z_next_pred, mu_next.detach())

    total = recon + weights.beta * kl + weights.lam * pred_loss
    return total, {"recon": recon.item(), "kl": kl.item(), "pred": pred_loss.item()}


class EWCLoss:
    """
    §6.3 — Elastic Weight Consolidation.
    Prevents catastrophic forgetting during daily online retraining.
    Penalty: λ_EWC · Σ F_i · (θ_i - θ_i*)²
    """
    def __init__(self, model, dataloader, importance: float = 1000.0,
                 loss_fn=None, weights=None):
        self.model      = model
        self.importance = importance
        self.params     = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher     = self._compute_fisher(dataloader, loss_fn, weights)

    def _compute_fisher(self, loader, loss_fn, weights):
        fisher = {}
        self.model.zero_grad()
        for batch in loader:
            loss, _ = loss_fn(self.model, *batch) if loss_fn else (torch.tensor(0.0), {})
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] = fisher.get(n, torch.zeros_like(p)) + p.grad.detach()**2 / len(loader)
        return fisher

    def penalty(self) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.params[n])**2).sum()
        return self.importance * loss
