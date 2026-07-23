"""
§3.1 — Encoder E: Transformer VAE mapping I_t → z_t ~ N(μ_φ, Σ_φ).

Input:  x_assets [B, N_assets, d_per_asset]  (5D state + RSI + BPV + jump_flag per asset)
        x_macro  [B, d_macro=6]               (DFF, CPI, UNRATE, T10Y2Y, VIX, M2SL)
        x_news   [B, 384]                     (sentence-transformer mean embedding)

Output: z_t ∈ ℝ^64 (reparameterized latent state)
"""
import torch
import torch.nn as nn


class FinancialEncoder(nn.Module):
    def __init__(self, d_obs: int = 512, d_latent: int = 64,
                 n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.d_latent = d_latent
        # Per-asset + macro + news → common d_model
        self.asset_proj = nn.Linear(d_obs, 256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=n_heads, dim_feedforward=512,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mu_head     = nn.Linear(256, d_latent)
        self.logvar_head = nn.Linear(256, d_latent)

    def forward(
        self,
        x_assets: torch.Tensor,   # [B, N, d_per_asset]
        x_macro:  torch.Tensor,   # [B, 6]
        x_news:   torch.Tensor,   # [B, 384]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x_assets.shape
        # Broadcast macro and news as extra tokens
        macro_token = x_macro.unsqueeze(1).expand(B, 1, x_macro.size(-1))
        news_token  = x_news.unsqueeze(1).expand(B, 1, x_news.size(-1))
        # Pad all feature types to d_obs via projection
        x = torch.cat([x_assets, macro_token, news_token], dim=1)  # [B, N+2, *]
        x = self.asset_proj(x)
        x = self.transformer(x)
        h = self.pool(x.permute(0, 2, 1)).squeeze(-1)  # [B, 256]
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-4, 4)
        return mu, logvar

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε·exp(½·logvar)."""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, x_assets, x_macro, x_news) -> torch.Tensor:
        mu, logvar = self.forward(x_assets, x_macro, x_news)
        return self.sample(mu, logvar)


class FinancialDecoder(nn.Module):
    """Decode z_t back to asset feature space (for reconstruction loss)."""
    def __init__(self, d_latent: int = 64, d_out: int = 512, n_assets: int = 500):
        super().__init__()
        self.n_assets = n_assets
        self.net = nn.Sequential(
            nn.Linear(d_latent, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, n_assets * d_out),
        )
        self.d_out = d_out

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        return self.net(z).view(B, self.n_assets, self.d_out)
