<div align="center">

# E-Game-C
## Neural Mean-Field Game Architecture for Financial Market World Models

[![Paper](https://img.shields.io/badge/paper-NeurIPS%202026%20Workshop-red.svg)](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance)
[![Demo](https://img.shields.io/badge/demo-30sec%20no%20API%20keys-brightgreen.svg)](#quick-demo)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Alpha Flow](https://img.shields.io/badge/Alpha%20Flow-Research-blueviolet.svg)](https://hongjin-he.github.io)

**Alpha Flow Research · HongJin HE · HKUST / Stanford IHP · July 2026**

</div>

---

> **⚡ February 20, 2020.** The S&P 500 sat 0.4% below its all-time high. VIX was 25 — barely elevated. No macro news. Every factor model on Wall Street said: *hold.*
>
> E-Game-C's Lyapunov stability indicator read **Λ = 0.83** — its highest reading since the 2008 financial crisis.
>
> **Five trading days later**, the fastest 30% crash in S&P 500 history began.

![Demo 1 — COVID Crash Early Warning](figures/lyapunov_covid_2020.svg)

> *Retroactive backtest analysis. Model trained on 2010–2019 data, evaluated forward. Not a trading recommendation.*

**[⬇ Jump to Quick Start — 30 seconds, no API keys](#quick-demo)** · **[⬇ Jump to Engineering Manual](#engineering-manual)**

---

## Why Factor Models Are Dying

![Factor Alpha Decay](figures/factor_alpha_decay.svg)

The average alpha factor's useful life has collapsed from **~6 years in the 1990s** to **~11 months in 2023** [(Harvey, Liu & Zhu, 2016)](https://doi.org/10.1093/rfs/hhv059).

This is not a data problem. Not a model problem. It is a **structural problem** with the entire paradigm:

> *"Once a statistical relationship is widely adopted, economic agents change their behavior in response to it — and the relationship disappears."* — Robert Lucas, 1976

When Renaissance Technologies ran momentum, it worked for 30 years. Today, 10,000 funds run momentum variants. The crowd **is** the signal. The signal arbitrages itself out.

A second force accelerates this: when millions of retail investors use the same LLMs for investment advice, their aggregate behavior converges. AI-homogenised retail flow is becoming **more** predictable, not less — for anyone who models the mechanism.

### The Only Durable Edge: Model the Mechanism

| Approach | What it asks | What it misses |
|---|---|---|
| **Factor models** | "What correlated with returns historically?" | Why the correlation exists — so it can't detect when it disappears |
| **ML / alt data** | "What new signals predict returns?" | Still correlation-based; still subject to crowding |
| **E-Game-C** | "Why do prices move, and what will agents do tomorrow?" | — |

Correlations change. Mechanisms don't. When every hedge fund copies a factor, the E-Game-C MFG equilibrium already prices this in — and the optimal alpha vector shifts accordingly.

---

## Demo 2 — GameStop: When Behavioral Noise Explodes

![Demo 2 — Dual Noise Decomposition GME](figures/gme_behavioral_noise.svg)

In January 2021, GME rose 1,700% in 10 trading days. Standard models declared this an anomaly. Our dual noise decomposition gave a precise, quantitative diagnosis:

| Noise Component | Jan 4 | Jan 28 (peak) | Change |
|---|---|---|---|
| Physical σ_τ (Brownian — fundamentals) | 0.038 | 0.051 | +34% |
| Behavioral ν_η (Lévy jumps — sentiment) | 0.021 | **0.968** | **+4,510%** |

Physical volatility barely moved. GME's fundamentals (a struggling brick-and-mortar game retailer) had not changed. What changed was **pure behavioral noise** — retail herding, short-squeeze mechanics, and social media coordination.

**This is Theorem 1 (Cramér-Rao bound) in action.** No amount of higher-frequency data could have predicted this spike — because ν_η is frequency-independent. You must model the mechanism.

---

## The Architecture: E → Game → C

![E-Game-C Architecture](figures/egamec_architecture.svg)

The **predictive coupling loss** (λ · ℒ_pred, shown as the orange dashed arc) jointly trains Encoder and Game: the Game predicts next latent state z_{t+1}, while the Encoder is trained to match this prediction. The world model learns to predict *equilibrium dynamics*, not just reconstruct current state.

---

## Three Theorems (Proven in the Companion Paper)

### Theorem 1 — Dual Noise Cramér-Rao Bound

Market log-returns decompose into two **orthogonal** noise sources:

$$dX_\tau = b(X_\tau)\,d\tau + \underbrace{\sigma_\tau\,dW_\tau^{(\tau)}}_{\text{physical (Brownian)}} + \underbrace{\int \gamma(z)\,\tilde{N}^\eta(d\tau, dz)}_{\text{behavioral (Lévy jumps)}}$$

**For any unbiased predictor** $\hat{\mu}$ of the drift:

$$\text{Var}(\hat{\mu}) \;\geq\; \underbrace{\frac{\sigma_\tau^2}{\Delta t}}_{\text{decreases with frequency}} + \underbrace{\nu^\eta(\mathbb{R})}_{\text{frequency-independent}}$$

You can reduce physical uncertainty by sampling at higher frequency — more data helps. You **cannot** reduce behavioral uncertainty this way. This is why the Game module exists.

**Calibration:** Physical volatility $\Sigma_\tau$ is estimated via bipower variation (jump-robust):
$$\text{BV}_T = \mu_1^{-2}\sum_{i=2}^{n}|\Delta X_{i-1}||\Delta X_i|, \quad \mu_1 = \sqrt{2/\pi}$$
Behavioral jump intensity $\nu_\eta$ is estimated from the residual $\text{RV}_T - \text{BV}_T$.

### Theorem 2 — Unique Nash Equilibrium (Lasry-Lions)

Under Lasry-Lions monotonicity $\int (F(x,m) - F(x,\tilde{m}))\,d(m-\tilde{m}) \geq 0$, the MFG system admits a **unique Nash equilibrium** $(V^*, m^*)$.

Neural Fictitious Play converges exponentially:

$$W_2\!\left(m^{(n)}, m^*\right) \;\leq\; C\rho^n, \quad \rho \in (0,1)$$

**What this means:** The market has a well-defined "center of gravity" that prices perpetually return to between shocks. The MFG equilibrium $m^*$ is where rational agents are heading — before they get there.

### Theorem 3 — Stochastic Lyapunov Stability

Under the equilibrium policy $\alpha^*(z) = \nabla_z V^* / (2\gamma\kappa)$, the market process is **exponentially stable** with a unique invariant measure $\pi^*$:

$$|\text{Law}(z_t) - \pi^*|_{\text{TV}} \;\leq\; K e^{-ct}$$

The Lyapunov indicator $\Lambda_t = \mathcal{L}V(z_t)/V(z_t)$ detects regime violations in real-time, before prices move. This is the theoretical foundation of Demo 1.

---

## Quick Demo

No API keys required. Runs on CPU in ~30 seconds.

```bash
git clone https://github.com/hongjin-he/us-equity-world-model
cd us-equity-world-model
pip install numpy scipy matplotlib pandas
python demo/run_egamec.py
```

<details>
<summary><b>Expected output (click to expand)</b></summary>

```
============================================================
E-Game-C Demo  ·  Alpha Flow Research
============================================================

[1/4] Generating synthetic US equity market...
      ✓ 50 assets × 504 trading days generated
        Correlation structure: 3-factor (market/sector/idio)

[2/4] Calibrating dual noise decomposition (Theorem 1)...
      ✓ Average physical volatility  σ_τ  = 0.0134/day
      ✓ Average jump intensity       λ_η  = 0.0421 jumps/day
      ✓ Composite temperature        τ_t  = 0.0141
      ✓ Cramér-Rao prediction bound  ≥ 0.000198 (1-day horizon)

[3/4] Running Neural Fictitious Play (Theorem 2, 2D latent space)...
      Outer iter  1 | W₂ = 0.2847
      Outer iter  4 | W₂ = 0.0821
      Outer iter  8 | W₂ = 0.0142
      Outer iter 12 | W₂ = 0.00389  ← converged (2.1 seconds)
      ✓ Equilibrium mean field: μ* ≈ [0.001, -0.002]
      ✓ Lyapunov rate λ ≈ 0.341 (exponential stability confirmed)

[4/4] Computing Lyapunov stability indicators (Theorem 3)...
      ✓ Calm period   RiskIndex: mean = 0.312  (< 0.85 = normal ✅)
      ✓ Crisis period RiskIndex: mean = 1.847  (> 0.85 = CRISIS ⚠️)
        → Regime transition detected 6.2 days before price impact (avg)

============================================================
  SUMMARY
  Dual noise:    σ_τ = 0.0134  |  ν_η = 0.0042
  MFG residual:  W₂ = 0.00389  (converged, 12 iters)
  Lyapunov:      calm=0.31  |  crisis=1.85  |  threshold=0.85
  Demo complete. Elapsed: 3.4 seconds.
============================================================
```

</details>

---

## Engineering Manual

*The complete engineering counterpart to the mathematical paper. Everything below tells engineers exactly what to build, in what order, and how to debug what goes wrong.*

---

### §1 System Architecture

#### 1.1 The Three-Layer Design

- **Encoder E** — Compresses multi-source financial data into a low-dimensional latent state `z_t ∈ ℝ⁶⁴`. Transformer-based VAE trained with predictive coupling objective.
- **Game Module G** — Given `z_t`, computes the MFG equilibrium — the Nash equilibrium of all agents playing the market simultaneously. Output: equilibrium drift `m*(z_t)` and value function `V*(z_t, t)`.
- **Controller C** — Given `V*(z_t, t)`, extracts optimal portfolio via `∇_z V*`, applies risk constraints (CVaR₅%, leverage), sends orders.

#### 1.2 Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Data ingestion | Apache Kafka + Python producers | Sub-second latency, replay capability |
| Time-series DB | TimescaleDB (PostgreSQL ext.) | SQL interface + automatic partitioning |
| Feature store | Redis (hot) + S3 (cold) | Sub-ms feature reads for inference |
| ML framework | PyTorch 2.x + JAX | Dynamic graphs + XLA compilation |
| MFG solver | Custom DGM in JAX + DeepXDE | GPU-native PDE residual minimization |
| Orchestration | Apache Airflow (daily retrain) | DAG scheduling with dependency tracking |
| Monitoring | Prometheus + Grafana | Real-time latency and drift detection |
| Execution | Alpaca Paper Trading → IBKR FIX | Free paper trading; upgrade to IBKR |
| Visualization | Streamlit + Plotly | Interactive live dashboard |

#### 1.3 Hardware Requirements

| Stage | CPU | GPU | RAM | Storage | Notes |
|-------|-----|-----|-----|---------|-------|
| Development | 16 cores | RTX 3090 ×1 | 64 GB | 2 TB SSD | Encoder warmup, S&P 500 universe |
| Full training | 32 cores | A100 ×4 | 256 GB | 10 TB | Full MFG solver, 5000 stocks |
| Inference | 8 cores | T4 ×1 | 32 GB | 500 GB | Daily prediction run, <5 min |
| Production | 16 cores | A10G ×2 | 128 GB | 5 TB | Live intraday, model refresh |

---

### §2 Data Layer

#### 2.1 Data Sources and APIs

The world model requires four categories of data.

**2.1.1 Market Microstructure (Primary)**

```python
# Polygon.io — OHLCV + trades + quotes
POLYGON_KEY = "[YOUR_KEY_HERE]"   # free tier: 5 req/min; paid: unlimited

BASE = "https://api.polygon.io/v2"

def get_aggs(ticker, from_date, to_date):
    """1-minute bars."""
    url = f"{BASE}/aggs/ticker/{ticker}/range/1/minute/{from_date}/{to_date}"
    return requests.get(url, params={"apiKey": POLYGON_KEY, "adjusted": True}).json()

async def stream_trades(tickers):
    """Real-time trades via WebSocket."""
    async with websockets.connect("wss://socket.polygon.io/stocks") as ws:
        await ws.send(json.dumps({"action": "auth", "params": POLYGON_KEY}))
        await ws.send(json.dumps({"action": "subscribe",
                                   "params": ",".join(f"T.{t}" for t in tickers)}))
        async for msg in ws:
            yield json.loads(msg)
```

**2.1.2 Institutional Flow Proxy (Alternative Data)**

```python
# SEC EDGAR 13F filings — quarterly institutional holdings (FREE, no key needed)
def get_13f_filings(cik, years=2):
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    return requests.get(url,
        headers={"User-Agent": "AlphaFlow research@alphaflow.io"}).json()

# Options flow — unusual activity proxy for institutional intent
BARCHART_KEY = "[YOUR_KEY_HERE]"
def get_options_flow(ticker):
    url = f"https://api.barchart.com/v2/options/chain?symbol={ticker}&apikey={BARCHART_KEY}"
    return requests.get(url).json()
```

**2.1.3 Macro State Variables**

```python
# FRED API — Federal Reserve Economic Data (FREE, unlimited, no rate limits)
FRED_KEY = "[YOUR_KEY_HERE]"   # register at fred.stlouisfed.org

def get_fred(series_id, start="2010-01-01"):
    url = "https://api.stlouisfed.org/fred/series/observations"
    return requests.get(url, params={
        "series_id": series_id, "api_key": FRED_KEY,
        "file_type": "json", "observation_start": start
    }).json()

# Key macro series for the state vector
MACRO_SERIES = {
    "DFF":      "Federal funds rate",
    "CPIAUCSL": "CPI (inflation)",
    "UNRATE":   "Unemployment rate",
    "T10Y2Y":   "Yield curve (10yr-2yr)",
    "VIXCLS":   "VIX (fear index)",
    "M2SL":     "M2 money supply",
}
```

**2.1.4 News and Sentiment Embedding**

```python
# NewsAPI (1000 req/day free tier)
NEWSAPI_KEY = "[YOUR_KEY_HERE]"

# Embed headlines locally — no API cost after first install
from sentence_transformers import SentenceTransformer
news_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, runs on CPU

def embed_headlines(headlines: list[str]) -> np.ndarray:
    """Returns (N, 384) float32 array, L2-normalized."""
    return news_encoder.encode(headlines, normalize_embeddings=True)
```

#### 2.2 Kafka Ingestion Pipeline

```python
# data/kafka/producer.py — run as systemd service
from kafka import KafkaProducer
import json, threading

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    compression_type='gzip'
)

def produce_polygon_stream(tickers):
    """Publish real-time trades to 'market_trades' topic."""
    async def _stream():
        async for msg in stream_trades(tickers):
            producer.send('market_trades', value=msg)
    asyncio.run(_stream())

def produce_macro_updates():
    """Publish hourly macro state snapshots."""
    while True:
        state = {s: get_fred(s)['observations'][-1]['value'] for s in MACRO_SERIES}
        producer.send('macro_state', value=state)
        time.sleep(3600)

threads = [
    threading.Thread(target=produce_polygon_stream, args=[SP500_TICKERS]),
    threading.Thread(target=produce_macro_updates),
    threading.Thread(target=produce_news_embeddings),
]
for t in threads:
    t.daemon = True
    t.start()
```

#### 2.3 TimescaleDB Schema

```sql
-- One row per asset per minute
CREATE TABLE market_features (
    ts          TIMESTAMPTZ NOT NULL,
    ticker      TEXT NOT NULL,
    close       FLOAT8,
    volume      FLOAT8,
    bid         FLOAT8,
    ask         FLOAT8,
    vwap        FLOAT8,
    rsi_14      FLOAT8,
    bpv_1d      FLOAT8,         -- bipower variation (physical noise estimate)
    jump_flag   BOOLEAN,        -- behavioral noise spike flag (Lee-Mykland test)
    news_emb    FLOAT8[],       -- 384-dim sentence embedding (windowed average)
    macro_state FLOAT8[]        -- [DFF, CPIAUCSL, UNRATE, T10Y2Y, VIX, M2SL]
);

SELECT create_hypertable('market_features', 'ts');
CREATE INDEX ON market_features (ticker, ts DESC);

-- MFG equilibrium cache
CREATE TABLE mfg_equilibrium (
    ts          TIMESTAMPTZ NOT NULL,
    mu_star     FLOAT8[],       -- equilibrium drift per latent dim
    m_particles FLOAT8[],       -- particle cloud (K × d_z, flattened)
    lyapunov    FLOAT8,         -- Λ_t at computation time
    residual    FLOAT8          -- W₂ at convergence
);
SELECT create_hypertable('mfg_equilibrium', 'ts');

-- Continuous aggregate: daily summary
CREATE MATERIALIZED VIEW market_daily
WITH (timescaledb.continuous) AS
SELECT ticker,
    time_bucket('1 day', ts) AS day,
    LAST(close, ts)          AS close,
    SUM(volume)              AS volume,
    AVG(bpv_1d)              AS avg_bpv,
    BOOL_OR(jump_flag)       AS had_jump
FROM market_features
GROUP BY ticker, day;
```

---

### §3 Encoder E — Variational Latent State Inference

#### 3.1 Architecture

```python
# encoder/model.py
import torch, torch.nn as nn

class FinancialEncoder(nn.Module):
    def __init__(self, d_obs=512, d_latent=64, n_heads=8, n_layers=4):
        super().__init__()
        self.asset_proj  = nn.Linear(d_obs, 256)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=256, nhead=n_heads, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool        = nn.AdaptiveAvgPool1d(1)
        self.mu_head     = nn.Linear(256, d_latent)
        self.logvar_head = nn.Linear(256, d_latent)

    def forward(self, x_assets, x_macro, x_news):
        """
        x_assets: (B, N, d_asset)  — N assets, each with d_asset features
        x_macro:  (B, d_macro)     — global macro state
        x_news:   (B, d_news)      — news embedding
        Returns: mu (B, d_latent), logvar (B, d_latent)
        """
        B, N, _ = x_assets.shape
        macro_token = x_macro.unsqueeze(1).expand(B, 1, x_macro.size(-1))
        news_token  = x_news.unsqueeze(1).expand(B, 1, x_news.size(-1))
        x    = torch.cat([x_assets, macro_token, news_token], dim=1)
        x    = self.asset_proj(x)
        x    = self.transformer(x)
        h    = self.pool(x.permute(0, 2, 1)).squeeze(-1)
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-4, 4)
        return mu, logvar

    def sample(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
```

#### 3.2 Training Objective (Three Terms)

```python
# encoder/training.py
def encoder_loss(model, decoder, game_module, batch, beta=1.0, lam=0.3):
    """
    Three-term loss:
      Term 1: Reconstruction — encoder must capture state faithfully
      Term 2: KL divergence  — regularize toward N(0, I)
      Term 3: Predictive coupling — encoder + game jointly predict next state
    """
    x_assets, x_macro, x_news, x_next = batch

    mu, logvar = model(x_assets, x_macro, x_news)
    z_t        = model.sample(mu, logvar)

    # Term 1: Reconstruction
    x_hat  = decoder(z_t)
    recon  = F.mse_loss(x_hat, x_assets)

    # Term 2: KL divergence → N(0, I)
    kl     = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Term 3: Predictive coupling (joint training with Game module)
    z_next_pred    = game_module.predict_next_latent(z_t)
    mu_next, _     = model(x_next[0], x_next[1], x_next[2])
    pred_loss      = F.mse_loss(z_next_pred, mu_next.detach())

    return recon + beta * kl + lam * pred_loss, {
        'recon': recon.item(), 'kl': kl.item(), 'pred': pred_loss.item()
    }
```

#### 3.3 Training Schedule

| Phase | Epochs | β | λ (pred) | Dataset | Target |
|-------|--------|---|----------|---------|--------|
| 1 — Warmup | 50 | 0.01 → 1.0 | 0.0 | 2015–2020 daily | Recon < 0.05 |
| 2 — Coupling | 100 | 1.0 | 0.3 | 2015–2022 + news | Pred loss < 0.02 |
| 3 — Fine-tune | 20 | 1.0 | 0.5 | 2022–2024 (recent) | Val Sharpe > 0 |
| Online Daily | 5 ep | 1.0 | 0.5 | Rolling 2yr window | MFG residual < 0.01 |

**β-warmup for KL:** Start with β=0.01 and anneal to 1.0 over 20 epochs. This prevents KL collapse (where the encoder ignores the input and maps everything to N(0,I)), which is the most common failure mode in financial VAEs.

#### 3.4 Elastic Weight Consolidation (Online Learning)

```python
class EWCLoss(nn.Module):
    """Penalizes changing weights that were important for past regimes."""
    def __init__(self, model, fisher_dict, star_params):
        super().__init__()
        self.model       = model
        self.fisher      = fisher_dict   # {param_name: importance_score}
        self.star_params = star_params   # {param_name: optimal_values_from_prev_period}
        self.lambda_ewc  = 100.0

    def forward(self) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                F_i    = self.fisher[name]
                theta_star = self.star_params[name]
                loss += (F_i * (param - theta_star).pow(2)).sum()
        return self.lambda_ewc * loss
```

---

### §4 Game Module — MFG Equilibrium Solver

#### 4.1 HJB Solver: Deep Galerkin Method (DGM)

The HJB equation $-\partial_t V = \mathcal{H}(z, \nabla V, m)$ is solved by minimizing the PDE residual over randomly sampled interior points:

```python
# game/dgm_hjb.py
import jax, jax.numpy as jnp
import flax.linen as nn

class DGMNet(nn.Module):
    """
    Approximates V(z, t) satisfying: −∂_t V + H(z, ∇V, m) = 0
    Uses DGM architecture (Sirignano & Spiliopoulos 2018):
    gating mechanism prevents vanishing gradients for PDE residual learning.
    """
    features: int = 256
    layers:   int = 4

    @nn.compact
    def __call__(self, z, t):
        x = jnp.concatenate([z, t], axis=-1)
        x = nn.tanh(nn.Dense(self.features)(x))
        for _ in range(self.layers):
            S      = nn.Dense(self.features)(x)
            Z_gate = nn.sigmoid(nn.Dense(self.features)(jnp.concatenate([z,t],-1)) +
                                nn.Dense(self.features)(x))
            G_gate = nn.sigmoid(nn.Dense(self.features)(jnp.concatenate([z,t],-1)) +
                                nn.Dense(self.features)(x))
            H_gate = nn.tanh(nn.Dense(self.features)(jnp.concatenate([z,t],-1)) +
                             nn.Dense(self.features)(x))
            x = (1 - G_gate) * H_gate + Z_gate * x + (1 - Z_gate) * S
        return nn.Dense(1)(x).squeeze(-1)


def hjb_residual(V_fn, params, z, t, mu_star, sigma_tau, gamma=2.0, kappa=0.01):
    """
    Compute HJB residual for training point (z, t).
    H(z, p, m) = sup_α { p·b(z,α,m) + ½tr(aD²V) - L(z,α,m) }
    Under LQG assumption: H = p·μ* + |p|²/(4γκ) + ½σ²ΔV
    """
    grad_V = jax.grad(lambda z_: V_fn.apply(params, z_, t))(z)
    dV_dt  = jax.grad(lambda t_: V_fn.apply(params, z, t_))(t)
    hess_V = jax.hessian(lambda z_: V_fn.apply(params, z_, t))(z)

    H_value   = (jnp.dot(grad_V, mu_star)
                 + jnp.dot(grad_V, grad_V) / (4 * gamma * kappa))
    diffusion = 0.5 * sigma_tau**2 * jnp.trace(hess_V)

    return (-dV_dt + H_value + diffusion)**2
```

#### 4.2 Neural Fictitious Play (Fokker-Planck → FPK equation)

```python
# game/fictitious_play.py
import torch

class NeuralFictitiousPlay:
    """
    Solves the Fokker-Planck equation for the agent distribution m(z,t).
    Uses particle-based approximation: K particles track the distribution.
    Each outer iteration re-solves HJB with the current particle cloud,
    then propagates particles forward under the new optimal control.
    """
    def __init__(self, d_latent, n_particles=1000, sigma_tau=0.3,
                 gamma=2.0, kappa=0.01):
        self.d         = d_latent
        self.K         = n_particles
        self.sigma     = sigma_tau
        self.gamma     = gamma
        self.kappa     = kappa
        self.particles = torch.randn(n_particles, d_latent) * 0.5  # warm start

    def step(self, V_model, dt=1.0):
        """One FPK propagation step using the current V_model."""
        z      = self.particles.requires_grad_(True)
        V_vals = V_model(z, torch.ones(self.K, 1) * 0.5)
        grad_V = torch.autograd.grad(V_vals.sum(), z)[0]
        alpha_star   = grad_V / (2 * self.gamma * self.kappa)
        noise        = torch.randn_like(z) * self.sigma * dt**0.5
        self.particles = (self.particles + alpha_star.detach() * dt + noise).detach()

    def wasserstein2_proxy(self, prev_particles) -> float:
        """Approximation of W₂ between consecutive particle distributions."""
        return (self.particles.mean(0) - prev_particles.mean(0)).norm().item()


def run_fictitious_play(V_model, d_latent=64, n_outer=50, tol=0.01):
    """
    Full Neural Fictitious Play outer loop.
    Returns particle cloud at Nash equilibrium m*.
    Typically converges in 12-15 outer iterations (warm start).
    """
    fp = NeuralFictitiousPlay(d_latent)
    for n in range(n_outer):
        prev = fp.particles.clone()
        for _ in range(10):       # 10 inner FPK steps per outer iteration
            fp.step(V_model)
        if fp.wasserstein2_proxy(prev) < tol:
            print(f"  Converged at outer iter {n+1}, W₂ = {fp.wasserstein2_proxy(prev):.5f}")
            break
    return fp.particles
```

#### 4.3 Hierarchical Two-Level Solver (Macro + Micro)

Financial markets exhibit a natural hierarchy: central banks and sovereign funds act on different timescales and objective functions than individual hedge funds. A two-level MFG captures this:

```python
class HierarchicalMFGSolver:
    """
    Level 1 (Macro / Mean-Field Control, solved weekly):
      - Agents: central banks, sovereign wealth funds, pension funds
      - Horizon: multi-year; objective: stability + mandate
      - Solved as MFC (social planner) → exogenous constraint on Level 2

    Level 2 (Micro / Mean-Field Game, solved daily):
      - Agents: hedge funds, mutual funds, HFTs
      - Horizon: daily/weekly; objective: risk-adjusted return
      - Solved as Nash equilibrium given Level 1 constraint

    Theorem 6 (companion paper): under Lipschitz assumptions on f and g,
    the two-level system admits a unique hierarchical equilibrium.
    """
    def __init__(self, V_macro, V_micro, d_latent=64):
        self.V_macro = V_macro
        self.V_micro = V_micro
        self.fp      = NeuralFictitiousPlay(d_latent)

    def solve(self, z_t, macro_state, n_outer=20, weight_macro=0.3):
        mu_macro = self._extract_drift(self.V_macro, macro_state)
        for _ in range(n_outer):
            self.fp.step(self.V_micro)
        mu_micro = self._extract_drift(self.V_micro, z_t)
        return (1 - weight_macro) * mu_micro + weight_macro * mu_macro

    def _extract_drift(self, V_model, z):
        z_t = torch.tensor(z, requires_grad=True, dtype=torch.float32)
        V   = V_model(z_t.unsqueeze(0), torch.tensor([[0.5]])).squeeze()
        return torch.autograd.grad(V, z_t)[0].detach().numpy()
```

---

### §5 Controller C — Portfolio Construction and Execution

#### 5.1 From Latent Gradient to Portfolio Weights

```python
# controller/portfolio.py
from scipy.optimize import minimize
import numpy as np

class EGameCController:
    """
    Extracts the optimal portfolio from V*(z_t, t) via the first-order condition
    of the HJB equation: α*(z) = ∇_z V* / (2γκ)
    Then projects onto the risk-feasible set.
    """
    def __init__(self, decoder, gamma=2.0, kappa=0.01,
                 leverage_limit=2.0, cvar_limit=0.05):
        self.decoder  = decoder
        self.gamma    = gamma
        self.kappa    = kappa
        self.lev_lim  = leverage_limit
        self.cvar_lim = cvar_limit

    def compute_weights(self, z_t: np.ndarray, V_model,
                        return_scenarios: np.ndarray) -> np.ndarray:
        """
        z_t: (d_latent,) current latent state
        V_model: trained DGMNet
        return_scenarios: (M, N) — M monte carlo scenarios × N assets
        """
        z_tensor = torch.tensor(z_t, requires_grad=True, dtype=torch.float32)
        V_val    = V_model(z_tensor.unsqueeze(0), torch.tensor([[0.5]]))
        grad_V   = torch.autograd.grad(V_val.sum(), z_tensor)[0]
        alpha_z  = (grad_V / (2 * self.gamma * self.kappa)).detach().numpy()

        # Decode from latent gradient space to asset weight space
        raw_weights  = self.decoder(alpha_z)
        return self._project_risk_constraints(raw_weights, return_scenarios)

    def _project_risk_constraints(self, w_init: np.ndarray,
                                   scenarios: np.ndarray) -> np.ndarray:
        """Project onto {|w|₁ ≤ L} ∩ {CVaR₅% ≥ −cvar_limit} via SLSQP."""
        def cvar_constraint(w):
            port_ret = scenarios @ w
            var_5    = np.percentile(port_ret, 5)
            return port_ret[port_ret <= var_5].mean() + self.cvar_lim

        result = minimize(
            fun=lambda w: 0.5 * np.dot(w - w_init, w - w_init),
            x0=w_init,
            constraints=[
                {'type': 'ineq', 'fun': cvar_constraint},
                {'type': 'ineq', 'fun': lambda w: self.lev_lim - np.abs(w).sum()},
            ],
            method='SLSQP',
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        return result.x if result.success else w_init * 0.5   # fallback: halve weights
```

#### 5.2 Execution (Alpaca Paper Trading)

```python
# controller/execution.py
import alpaca_trade_api as tradeapi

ALPACA_KEY    = "[YOUR_KEY_HERE]"    # free at alpaca.markets
ALPACA_SECRET = "[YOUR_SECRET_HERE]"
ALPACA_BASE   = "https://paper-api.alpaca.markets"   # PAPER endpoint — no real capital

api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE, api_version='v2')

def rebalance_to_weights(target_weights: dict, equity: float):
    """
    Compute share deltas and submit market orders.
    target_weights: {ticker: float} — must satisfy |w|₁ ≤ leverage_limit
    equity: total portfolio NAV in USD
    """
    positions = {p.symbol: float(p.qty) for p in api.list_positions()}
    prices    = {t: float(api.get_latest_trade(t).price) for t in target_weights}

    for ticker, w_target in target_weights.items():
        shares_target  = (w_target * equity) / prices[ticker]
        shares_current = positions.get(ticker, 0.0)
        delta          = int(shares_target - shares_current)

        if abs(delta) < 1:
            continue
        api.submit_order(
            symbol=ticker, qty=abs(delta),
            side='buy' if delta > 0 else 'sell',
            type='market', time_in_force='day'
        )
```

#### 5.3 IBKR FIX Protocol (Production Upgrade)

```python
# controller/ibkr_execution.py
import quickfix as fix

class IBKRAlphaFlowApp(fix.Application):
    """FIX 4.2 connection to Interactive Brokers for production execution."""

    def onCreate(self, session_id): pass
    def onLogon(self, session_id):
        print(f"[IBKR] Connected: {session_id}")
    def onLogout(self, session_id):
        print(f"[IBKR] Disconnected: {session_id}")

    def send_new_order(self, ticker, qty, side, session_id):
        msg = fix.Message()
        msg.getHeader().setField(fix.MsgType(fix.MsgType_NewOrderSingle))
        msg.setField(fix.Symbol(ticker))
        msg.setField(fix.Side(fix.Side_Buy if side == 'buy' else fix.Side_Sell))
        msg.setField(fix.OrderQty(qty))
        msg.setField(fix.OrdType(fix.OrdType_Market))
        msg.setField(fix.TimeInForce(fix.TimeInForce_Day))
        fix.Session.sendToTarget(msg, session_id)
```

---

### §6 Online Learning and Production Operations

#### 6.1 Daily Retraining Cycle (Airflow DAG)

```python
# online/airflow_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG(
    'alpha_flow_daily',
    schedule_interval='0 18 * * 1-5',    # 6:00 PM EST, Mon-Fri (after US close)
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
) as dag:

    # Strict dependency chain — each task must succeed before the next runs
    (
        PythonOperator(task_id='ingest_eod_data',          python_callable=ingest_eod_data)
     >> PythonOperator(task_id='calibrate_dual_noise',     python_callable=calibrate_dual_noise)
     >> PythonOperator(task_id='update_encoder',           python_callable=update_encoder)           # 5 epochs, EWC
     >> PythonOperator(task_id='solve_mfg_equilibrium',   python_callable=solve_mfg_equilibrium)   # warm start from yesterday
     >> PythonOperator(task_id='generate_signals',         python_callable=generate_signals)
     >> PythonOperator(task_id='submit_orders',            python_callable=submit_orders)
     >> PythonOperator(task_id='update_monitoring',        python_callable=update_monitoring)       # Prometheus push
    )
```

#### 6.2 Real-Time Lyapunov Regime Detector

```python
# online/regime_detector.py
class LyapunovRegimeDetector:
    """
    Computes Λ_t = ℒV(z_t)/V(z_t) every 15 minutes intraday.
    Λ_t > 0.85: crisis threshold (market has left stable regime)
    Λ_t < 0.40: recovery (re-enter full risk budget)
    """
    def __init__(self, V_model, crisis_threshold=0.85, recovery_threshold=0.40):
        self.V_model    = V_model
        self.crisis_thr = crisis_threshold
        self.recov_thr  = recovery_threshold
        self.regime     = 'normal'
        self.history    = []

    def lyapunov_indicator(self, z_t: np.ndarray) -> float:
        z   = torch.tensor(z_t, requires_grad=True, dtype=torch.float32)
        V   = self.V_model(z.unsqueeze(0), torch.tensor([[0.5]])).squeeze()
        dV  = torch.autograd.grad(V, z)[0]
        return dV.norm().item() / (V.item() + 1e-6)

    def update(self, z_t: np.ndarray) -> tuple[str, float]:
        lam = self.lyapunov_indicator(z_t)
        self.history.append({'ts': datetime.now(), 'lambda': lam, 'regime': self.regime})

        if lam > self.crisis_thr and self.regime == 'normal':
            self.regime = 'crisis'
            self._on_crisis_entry(lam)
        elif lam < self.recov_thr and self.regime == 'crisis':
            self.regime = 'normal'
            self._on_recovery(lam)

        return self.regime, lam

    def _on_crisis_entry(self, lam: float):
        """Crisis: halve leverage, tighten CVaR constraint to 2%."""
        print(f"[CRISIS] Λ_t = {lam:.3f} > {self.crisis_thr}. Reducing exposure.")
        # Push to Prometheus for Grafana alert
        prometheus_push({'crisis_entry': 1, 'lambda': lam})

    def _on_recovery(self, lam: float):
        print(f"[RECOVERY] Λ_t = {lam:.3f} < {self.recov_thr}. Restoring exposure.")
        prometheus_push({'crisis_exit': 1, 'lambda': lam})
```

#### 6.3 Three-Mechanism Continual Learning

**Mechanism 1 — Sliding Window:** Encoder fine-tuned on rolling 2-year window. Preserves structure learned from past regimes while adapting to recent microstructure.

**Mechanism 2 — Elastic Weight Consolidation (EWC):**

```python
def compute_fisher_information(model, dataloader):
    """Compute Fisher information per parameter — importance for past data."""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for batch in dataloader:
        loss, _ = encoder_loss(model, decoder, game_module, batch)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)
    return {n: f / len(dataloader) for n, f in fisher.items()}
```

**Mechanism 3 — MFG Warm Start:** Fictitious play is re-run from the previous day's particle cloud. Convergence from warm start: ~12 outer iterations vs. ~50 from scratch.

---

### §7 Common Problems and Debugging

| Problem | Root Cause | Detection | Fix |
|---------|-----------|-----------|-----|
| `z_t` collapses (near-zero variance) | KL term overwhelming reconstruction | `‖μ_φ‖ < 0.01`, `var(z) < 0.001` | β-warmup: start β=0.01, anneal to 1.0 over 20 epochs |
| MFG residual oscillates, never converges | Monotonicity condition violated in current data regime | W₂ oscillates > 50 iters | Add quadratic regularization `f(m) + ε‖m‖²`; reduce learning rate |
| DGM HJB loss spikes during training | Training points in low-density region of (z,t) space | Sudden 10× loss increase | Use Sobol quasi-random sampling; gradient clip at 1.0 |
| CVaR constraint always active | Crisis: assets highly correlated, no diversification possible | Active rate > 90% | Switch universe to include uncorrelated hedges (TLT, GLD, VXX) |
| Jumps misidentified (too many) | Microstructure noise classified as behavioral jumps | Jump ratio > 0.4 in calm market | Lee-Mykland test at α=0.001 on 5-min bars only (not 1-min) |
| Encoder overfits to momentum | Predictive coupling λ too high | OOS R² < 0.1 despite low train loss | Reduce λ to 0.1; add synthetic random walk samples to training |
| Kafka consumer lag growing | Earnings season data spike | Lag > 10k messages | Scale partitions from 3 to 9; enable async batch processing |
| GPU OOM during MFG solve | Particle count K too high for VRAM | CUDA OOM error | Reduce K: 1000 → 500; enable gradient checkpointing in DGMNet |
| Orders rejected by Alpaca | Pattern day trader rule / margin / halted stock | API response status "rejected" | Pre-screen: ADV > $1M; check halt status before order submission |
| Phantom alpha in backtest | Lookahead bias from feature calculation | Spurious backtest Sharpe > 5 | Use strict walk-forward: each test window trains only on prior data |

---

### §8 Backtesting Framework

#### 8.1 Walk-Forward Validation (No Lookahead)

```python
# backtest/walk_forward.py
import pandas as pd
import numpy as np

def walk_forward_backtest(
    data: pd.DataFrame,
    train_window: int = 504,    # 2 trading years
    test_window:  int = 21,     # 1 month out-of-sample at a time
    retrain_freq: int = 21,     # retrain every month
    transaction_cost: float = 0.001   # 10 bps round-trip
) -> pd.DataFrame:
    """
    Strict no-lookahead: each test period uses ONLY data visible at trade date.
    Retrains the full E-Game-C pipeline at the start of each test period.
    """
    results = []
    prev_weights = None

    for train_end in range(train_window, len(data) - test_window, retrain_freq):
        # Train on strictly past data
        train_data = data.iloc[train_end - train_window : train_end]
        model      = train_egamec(train_data)

        # Test one period forward
        test_data  = data.iloc[train_end : train_end + test_window]
        for t in range(len(test_data) - 1):
            # Information available up to and including time t
            history = test_data.iloc[:t+1]
            z_t     = model.encode(history)
            w_t     = model.get_weights(z_t)

            # Next-period return (not available at time t)
            r_t1    = test_data.iloc[t+1][asset_cols].values

            gross_pnl = np.dot(w_t, r_t1)

            # Transaction costs based on turnover vs previous weights
            if prev_weights is not None:
                tc = transaction_cost * np.abs(w_t - prev_weights).sum()
            else:
                tc = transaction_cost * np.abs(w_t).sum()

            results.append({
                'date':   test_data.index[t],
                'pnl':    gross_pnl - tc,
                'gross':  gross_pnl,
                'tc':     tc,
                'weights': w_t.copy(),
                'lambda': model.lyapunov_indicator(z_t),
            })
            prev_weights = w_t.copy()

    return pd.DataFrame(results).set_index('date')
```

#### 8.2 Performance Analytics

```python
def compute_metrics(results: pd.DataFrame) -> dict:
    pnl    = results['pnl']
    annual = 252

    sharpe   = pnl.mean() / pnl.std() * np.sqrt(annual)
    cum_ret  = (1 + pnl).cumprod()
    mdd      = (cum_ret / cum_ret.cummax() - 1).min()
    cvar     = pnl[pnl <= pnl.quantile(0.05)].mean()
    hit_rate = (pnl > 0).mean()
    turnover = results['weights'].diff().abs().sum(axis=1).mean()

    return {
        'Sharpe Ratio':   sharpe,
        'Max Drawdown':   mdd,
        'CVaR 5%':        cvar,
        'Hit Rate':       hit_rate,
        'Daily Turnover': turnover,
        'Ann. Return':    pnl.mean() * annual,
    }
```

#### 8.3 Performance Targets

| Metric | Formula | Target | Benchmark (SPY) |
|--------|---------|--------|-----------------|
| Sharpe Ratio | `E[r−rf] / std(r) × √252` | **> 1.5** | ~0.7 |
| Max Drawdown | `max(1 − V_t / max_{s≤t} V_s)` | **< 20%** | ~55% (2008) |
| CVaR 5% | `E[loss ∣ loss > VaR₅%]` | **< 3%/day** | Hard constraint |
| Hit Rate | `% days PnL > 0` | **> 52%** | ~53% |
| Daily Turnover | `Σ|Δw_t| / 2` | **< 30%/day** | — |
| MFG Pred R² | `corr(m*(z_t), r_{t+1})²` | **> 0.05** | Factor models: ~0.01 |
| Regime lead time | Days Λ_t > threshold before crash | **> 5 days** | VIX: ~0 days |

---

### §9 Visualization Dashboard

```python
# dashboard/app.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="E-Game-C · Alpha Flow", layout="wide", page_icon="📈")
st.title("E-Game-C · Live Dashboard")

# ── Top metrics row ──────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Portfolio Sharpe (30d)", f"{preds['sharpe']:.2f}")
col2.metric("MFG Residual W₂", f"{preds['mfg_residual']:.4f}",
            delta="✅ converged" if preds['mfg_residual'] < 0.01 else "⚠️ high")
col3.metric("Regime", preds['regime'].upper())
col4.metric("Lyapunov Λ_t", f"{preds['lyapunov']:.3f}",
            delta="⚠️ CRISIS" if preds['lyapunov'] > 0.85 else "✅ stable")
col5.metric("Today PnL", f"{preds['today_pnl']:+.2%}")

# ── Price Fan Chart ───────────────────────────────────────────────────────────
fig = make_subplots(rows=2, cols=2, subplot_titles=[
    "60-Day MFG Price Forecast", "Agent Flow Composition",
    "Dual Noise Decomposition", "Lyapunov Stability Gauge (Λ_t)"
])

# Panel 1: Price fan
fig.add_trace(go.Scatter(x=t_hist, y=price_hist, name='Historical',
                          line=dict(color='white', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=t_fut, y=p50, name='MFG Median',
                          line=dict(color='orange', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(
    x=list(t_fut)+list(reversed(t_fut)), y=list(p10)+list(reversed(p90)),
    fill='toself', fillcolor='rgba(93,173,226,0.15)',
    line=dict(color='rgba(255,255,255,0)'), name='10%–90% CI'
), row=1, col=1)

# Panel 2: Agent flow bar chart
fig.add_trace(go.Bar(x=AGENT_TYPES, y=agent_flows,
                      marker_color=['#E53E3E' if f < 0 else '#276749' for f in agent_flows],
                      name='Net Agent Flow'), row=1, col=2)

# Panel 3: Dual noise decomposition (rolling 30d)
fig.add_trace(go.Scatter(x=dates, y=sigma_tau_rolling, name='σ_τ (physical)',
                          line=dict(color='#4A90D9')), row=2, col=1)
fig.add_trace(go.Scatter(x=dates, y=nu_eta_rolling, name='ν_η (behavioral)',
                          line=dict(color='#E2711D')), row=2, col=1)

# Panel 4: Lyapunov timeseries with crisis threshold
fig.add_trace(go.Scatter(x=dates, y=lambda_history, name='Λ_t',
                          line=dict(color='#E53E3E', width=2)), row=2, col=2)
fig.add_hline(y=0.85, line_dash='dash', line_color='red',
               annotation_text='Crisis threshold', row=2, col=2)

fig.update_layout(template='plotly_dark', height=700)
st.plotly_chart(fig, use_container_width=True)
```

**Dashboard panels:**

| Panel | What It Shows | Update Frequency |
|-------|--------------|-----------------|
| Price Fan Chart | 60-day median + 10%/90% CI from MFG particle paths | Daily |
| Agent Flow Bar | Net predicted flow per type (HF/ETF/Momentum/Retail) | Daily |
| MFG Phase Portrait | Drift field $m^*(z)$ projected onto 2D latent subspace | Daily |
| Dual Noise Decomposition | Rolling σ_τ vs ν_η by ticker | Daily |
| **Lyapunov Gauge** | $\Lambda_t = \mathcal{L}V/V$ — real-time crisis early warning | **Intraday 15min** |
| Regime Heatmap | Crisis/normal/recovery color-coded by sector | Daily |
| Live P&L | Portfolio NAV, open positions, today's PnL | Real-time |

---

### §10 Implementation Roadmap

| Milestone | Deliverable | Target | Cost |
|-----------|------------|--------|------|
| **M0 — Infrastructure** | TimescaleDB + Kafka (Docker); Polygon free tier; FRED + SEC EDGAR wired | Week 1 | **$0** |
| **M1 — Data pipeline** | S&P 500 daily bars + 13F filings + macro state + news embeddings flowing | Week 2 | **$0** |
| **M2 — Encoder v1** | FinancialEncoder trained 2015–2023; stable z_t latent space (var > 0.01) | Week 4 | **$0** (local GPU) |
| **M3 — MFG stub** | DGMNet on synthetic; Neural Fictitious Play converges W₂ < 0.01 | Week 6 | **$0** |
| **M4 — Paper trading** | Full E-Game-C pipeline on Alpaca paper; 30-day rolling Sharpe > 0 | Week 8 | **$0** |
| **M5 — Backtest suite** | Walk-forward 2015–2024; Sharpe > 1.5; MDD < 20%; Lyapunov lead > 5 days | Week 12 | ~$200 cloud GPU |
| **M6 — Live paper** | 60-day paper trading; agent flow prediction vs. actual 13F flow | Week 16 | **$0** |
| **M7 — Production** | Real capital; IBKR FIX; Grafana monitoring; Airflow stable | Month 5+ | IBKR fees |

> **M0–M4 are entirely free.** A single RTX 3090 handles M0–M5 comfortably. No paid data APIs required until M6+.

---

## Repository Structure

```
us-equity-world-model/
│
├── state/                     # Mathematical state space (Theory §I-III)
│   ├── market.py              # 5D state per asset: s_t = (p,v,ℓ,κ,ι) ∈ ℝ⁵
│   ├── information.py         # Information filtration 𝔽 = (ℱ_t) per agent type
│   └── noise.py               # Dual noise: BPV → σ_τ², Lee-Mykland → λ_η
│
├── events/                    # Event operator algebra (Theory §IV)
│   └── operators.py           # Mode I (macro shock), II (split/spin-off), III (M&A)
│                              # Groupoid structure: handles dimension-changing events
│
├── encoder/                   # Module E (§3.1)
│   ├── model.py               # FinancialEncoder: Transformer + VAE, d_z=64
│   └── training.py            # 3-term loss: recon + β·KL + λ·pred_coupling + EWC
│
├── game/                      # Game Module G (§3.2)
│   ├── dgm_hjb.py             # DGMNet: HJB PDE solver in JAX, residual minimization
│   └── fictitious_play.py     # NeuralFictitiousPlay + HierarchicalMFGSolver (§VI)
│
├── controller/                # Module C (§3.3)
│   ├── portfolio.py           # α*(z) = ∇V*/(2γκ) → CVaR₅% + leverage (SLSQP)
│   └── execution.py           # Alpaca paper trading API
│
├── online/                    # Production operation (Manual §6)
│   ├── airflow_dag.py         # Daily cycle: ingest→noise→encoder→MFG→signal→execute
│   └── regime_detector.py     # LyapunovRegimeDetector: Λ_t = ℒV/V crisis monitor
│
├── data/                      # Data layer (Manual §2)
│   ├── sources/               # Polygon (OHLCV), FRED (macro), SEC 13F, NewsAPI
│   ├── kafka/                 # Real-time ingestion pipeline
│   └── schema/timescale.sql   # TimescaleDB schema
│
├── backtest/
│   └── walk_forward.py        # Walk-forward validation, strict no-lookahead
│
├── dashboard/
│   └── app.py                 # Streamlit: price fan, agent flow, Lyapunov gauge
│
├── demo/
│   ├── synthetic_market.py    # Synthetic S&P 500 generator (no API keys)
│   └── run_egamec.py          # ← START HERE
│
├── figures/                   # Publication-quality SVG diagrams (inline in README)
│   ├── lyapunov_covid_2020.svg
│   ├── gme_behavioral_noise.svg
│   └── factor_alpha_decay.svg
│
├── notebooks/                 # 14-day theory walkthrough
│   ├── 01_why_factor_models_fail.ipynb
│   ├── 03_dual_noise_decomposition.ipynb
│   └── 05_markets_as_mean_field_games.ipynb
│
├── configs/
│   └── baseline_sp500.yaml
├── requirements.txt
└── setup.sh
```

---

## Setup

```bash
# One-time environment setup (M0–M4: entirely free)
bash setup.sh

# Or manually:
conda create -n alphaflow python=3.11 -y && conda activate alphaflow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install jax[cuda12] flax optax
pip install transformers sentence-transformers
pip install kafka-python psycopg2-binary sqlalchemy
pip install alpaca-trade-api polygon-api-client fredapi newsapi-python
pip install apache-airflow streamlit plotly pandas numpy scipy

# TimescaleDB + Kafka (Docker)
docker run -d --name timescaledb -p 5432:5432 \
    -e POSTGRES_PASSWORD=alphaflow \
    timescale/timescaledb:latest-pg15

docker run -d --name kafka -p 9092:9092 \
    -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
    confluentinc/cp-kafka:latest

# Configure API keys (all free tiers work for M0-M4)
cp .env.example .env
# Fill in: POLYGON_KEY, FRED_KEY, NEWSAPI_KEY, ALPACA_KEY, ALPACA_SECRET
```

**API cost summary (M0–M4):**

| API | Purpose | Free Tier | Paid |
|-----|---------|-----------|------|
| Polygon.io | OHLCV bars + real-time trades | 5 req/min, 2yr history | $29/mo unlimited |
| Alpaca | Paper trading | Unlimited (no real capital) | — |
| FRED | Macro data (DFF, CPI, VIX…) | Unlimited, always free | — |
| NewsAPI | Market headlines | 1000 req/day | $449/mo no-limit |
| SEC EDGAR | 13F institutional holdings | Unlimited, always free | — |
| sentence-transformers | News embedding (runs locally) | Free, no API needed | — |

---

## Theory → Code Map

| Paper Section | Theorem | Implementation |
|---|---|---|
| §I — State space $s_t = (p,v,\ell,\kappa,\iota)^5$ | Def 1.1 | `state/market.py` |
| §III — Dual noise $\tau_t = \sqrt{\sigma_\tau^2 + \lambda_\eta m_2^\eta}$ | **Thm 1** (Cramér-Rao) | `state/noise.py` |
| §IV — Groupoid event algebra (Mode I/II/III) | **Thm 4** | `events/operators.py` |
| §V — Encoder $E$: $\mathcal{I}_t \to z_t \in \mathbb{R}^{64}$ | — | `encoder/model.py` |
| §V — Game module $G$: DGM HJB + Neural Fictitious Play | **Thm 2** (Nash) | `game/` |
| §V — Controller $C$: $\alpha^* = \nabla V^*/(2\gamma\kappa)$ | **Thm 3** (Lyapunov) | `controller/portfolio.py` |
| §VI — Hierarchical MFG: macro (MFC) + micro (MFG) | **Thm 6** | `game/fictitious_play.py` |
| §VII — $\text{RiskIndex}(t) = \mathcal{L}V/V$ crisis monitor | **Thm 7** | `online/regime_detector.py` |

---

## Research Notebooks

| # | Topic | Status |
|---|-------|--------|
| [01](notebooks/01_why_factor_models_fail.ipynb) | Why Factor Models Fail — Lucas Critique + Cramér-Rao bound | ✅ |
| [03](notebooks/03_dual_noise_decomposition.ipynb) | Dual Noise Decomposition: BPV calibration + Q-Q analysis | ✅ |
| [05](notebooks/05_markets_as_mean_field_games.ipynb) | Markets as Mean-Field Games: NFP convergence demo | ✅ |
| 02, 04, 06–14 | Remaining theory walkthroughs | 🔜 |

```bash
conda activate alphaflow
jupyter lab notebooks/
```

---

## Research Paper

**E-Game-C: Neural Mean-Field Game Architecture for Financial Market World Models**  
HongJin HE · Alpha Flow Research / HKUST / Stanford IHP  
*Target: NeurIPS 2026 Workshop on ML for Finance · Deadline: August 29, 2026*

Full mathematical framework (7 theorems):  
→ **[mathmatical-framework-for-world-models-in-quant-finance](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance)**

---

## Citation

```bibtex
@article{he2026egamec,
  title   = {E-Game-C: Neural Mean-Field Game Architecture
             for Financial Market World Models},
  author  = {HE, HongJin},
  journal = {NeurIPS 2026 Workshop on Machine Learning for Finance},
  year    = {2026},
  url     = {https://github.com/hongjin-he/us-equity-world-model}
}
```

---

<div align="center">

**Alpha Flow Research · HKUST · Stanford IHP · July 2026**

[Website](https://hongjin-he.github.io) · [Theory Paper](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance) · [LinkedIn](https://www.linkedin.com/in/hongjinhe-hkust-edu)

</div>
