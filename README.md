<div align="center">

# E-Game-C
### Neural Mean-Field Game Architecture for Financial Market World Models

**[Paper (NeurIPS 2026 Workshop)](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance/blob/main/paper_draft_v1.pdf)** · **[Theory](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance)** · **[Notebooks](notebooks/)** · **[Demo](#quick-demo)**

*Alpha Flow Research · HongJin HE · HKUST / Stanford IHP · July 2026*

---

> **Factor models mine correlations. E-Game-C learns the mechanism.**

</div>

---

## The Problem

Standard quantitative strategies harvest statistical correlations — momentum, earnings yield, sector tilts — that predict returns historically. This approach has a structural weakness:

> *"Once a strategy becomes widely adopted, the statistical relationship it exploits changes."* — Robert Lucas, 1976

**Empirical consequence**: The average half-life of a new factor has declined from ~5 years in the 1990s to ~18 months today [(Harvey, Liu & Zhu, 2016)](https://doi.org/10.1093/rfs/hhv059). Factor models are dying faster because they learn correlations, not causes.

A second structural shift accelerates this:

> *When millions of retail investors query the same LLMs for investment advice, their aggregate flow converges toward a narrower distribution — one that sophisticated participants can model.*

AI-homogenised retail behaviour is becoming **more predictable**, not less. The right architecture captures this. Factor models cannot.

---

## Our Answer: Learn the Mechanism

**E-Game-C** replaces correlation mining with a genuine world model of financial markets:

```
Who is trading?      →  Agent states from 13F filings + ownership graphs
Why do they trade?   →  MFG Nash equilibrium of all participants
What prices emerge?  →  Order matching, not a parametric price process
```

The architecture has three modules co-designed for financial dynamics:

```
  Market Information I_t
  (OHLCV · 13F filings · macro · news)
          │
          ▼
  ┌───────────────────────────────┐
  │  E  — Encoder                │
  │  Transformer VAE              │
  │  I_t  →  z_t ∈ ℝ⁶⁴          │
  └───────────────────────────────┘
          │  z_t (latent market state)
          ▼
  ┌───────────────────────────────┐
  │  Game — MFG Equilibrium      │
  │  DGM HJB  +  Neural FP       │
  │  z_t  →  (V*, m*)            │
  └───────────────────────────────┘
          │  V*(z_t), m*(z_t)
          ▼
  ┌───────────────────────────────┐
  │  C  — Controller             │
  │  α*(z) = ∇V*/(2γκ)          │
  │  CVaR + leverage projection  │
  └───────────────────────────────┘
          │  w* ∈ ℝᴺ (portfolio weights)
          ▼
  Alpaca Paper Trading → IBKR FIX
```

> The dashed feedback arrow: the Game module's next-latent prediction jointly trains the Encoder via **predictive coupling loss** — the world model learns to predict equilibrium dynamics, not just reconstruct current state.

---

## Three Key Theorems

The architecture is backed by three theorems proven in the [companion paper](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance/blob/main/paper_draft_v1.pdf):

### Theorem 1 — Dual Noise Cramér-Rao Bound

Market returns decompose into two **orthogonal** noise sources:

$$dS_t = \mu_t\,dt + \underbrace{\sigma_\tau\,dW_t^{(\tau)}}_{\text{physical (Brownian)}} + \underbrace{dJ_t^{(\eta)}}_{\text{behavioral (Lévy jumps)}}$$

For any unbiased predictor:
$$\text{Var}(\hat{\mu}) \;\geq\; \underbrace{\frac{\sigma_\tau^2}{\Delta t}}_{\text{physical (↓ with frequency)}} + \underbrace{\nu_\eta(\mathbb{R})}_{\text{behavioral (frequency-independent!)}}$$

**Implication**: Getting more data cannot reduce behavioral uncertainty. The only way to lower this floor is to **model the behavioral mechanism** — which is exactly what the Game module does.

### Theorem 2 — Unique Nash Equilibrium

Under Lasry–Lions monotonicity, the MFG system admits a unique Nash equilibrium $(V^*, m^*)$, and Neural Fictitious Play converges exponentially:

$$W_2(m^{(n)}, m^*) \;\leq\; C\rho^n, \quad \rho \in (0,1)$$

**Implication**: The market has a **well-defined center of gravity** that prices perpetually return to. This is what the world model learns to predict.

### Theorem 3 — Stochastic Lyapunov Stability

Under equilibrium policy $\alpha^*$, the market process is exponentially stable:

$$\|\text{Law}(z_t) - \pi^*\|_{\text{TV}} \;\leq\; K e^{-ct}$$

**Implication**: The world model's Lyapunov indicator $\Lambda_t = \mathcal{L}V(z_t)/V(z_t)$ detects regime shifts (bubbles, crises) **before** they appear in prices. When $\Lambda_t > \text{threshold}$, the market has left its stable regime.

---

## Quick Demo

No API keys required. Runs on CPU in ~30 seconds.

```bash
git clone https://github.com/hongjin-he/us-equity-world-model
cd us-equity-world-model
pip install numpy scipy matplotlib pandas
python demo/run_egamec.py
```

Expected output:
```
============================================================
  E-Game-C Demo  ·  Alpha Flow Research
============================================================

[1/4] Generating synthetic US equity market...
      ✓ 50 assets × 504 trading days generated

[2/4] Calibrating dual noise decomposition (§III)...
      ✓ Average physical volatility  σ_τ  = 0.0134/day
      ✓ Average jump intensity       λ_η  = 0.0421 jumps/day
      ✓ Composite temperature        τ_t  = 0.0141
      ✓ Cramér-Rao prediction bound  ≥ 0.000198 (1-day horizon)

[3/4] Running Neural Fictitious Play (§IV mini demo, 2D latent space)...
      ✓ Converged at outer iteration 12, W₂ ≈ 0.00389  (2.1s)
      ✓ Equilibrium mean field:  μ* ≈ [0.001, -0.002]

[4/4] Computing Lyapunov stability indicators (§VII)...
      ✓ Calm period   RiskIndex:  mean = 0.312  (< 0.85 = normal)
      ✓ Crisis period RiskIndex:  mean = 1.847  (> 0.85 = CRISIS ⚠️)
```

---

## Research Notebooks

14-day first-principles walkthrough of the mathematical framework:

| # | Topic | Status |
|---|-------|--------|
| [01](notebooks/day01_why_factor_models_fail.ipynb) | Why Factor Models Fail — Lucas Critique + Cramér-Rao | ✅ |
| [03](notebooks/day03_dual_noise.ipynb) | Dual Noise Decomposition: BPV calibration + Q-Q analysis | ✅ |
| [05](notebooks/day05_markets_as_mean_field_games.ipynb) | Markets as Mean-Field Games: NFP convergence demo | ✅ |
| 02, 04, 06–14 | Remaining theory walkthroughs | 🔜 |

```bash
conda activate alphaflow
jupyter lab notebooks/
```

---

## Repository Structure

```
us-equity-world-model/
│
├── state/                    # Mathematical state space (Theory §I-III)
│   ├── market.py             # 5D state per asset: s_t = (p,v,ℓ,κ,ι) ∈ ℝ⁵
│   ├── information.py        # Information filtration 𝔽 = (ℱ_t) per agent
│   └── noise.py              # Dual noise: BPV → σ_τ, Lee-Mykland → λ_η
│
├── events/                   # Event operator algebra (Theory §IV)
│   └── operators.py          # Mode I (split), II (macro shock), III (M&A, spin-off)
│                             # Groupoid structure: non-square A_w for dim-changing events
│
├── encoder/                  # Module E (Paper §3.1)
│   ├── model.py              # FinancialEncoder: Transformer + VAE head, d_z=64
│   └── training.py           # 3-term loss: recon + β·KL + λ·pred_coupling + EWC
│
├── game/                     # Game Module G (Paper §3.2)
│   ├── dgm_hjb.py            # DGMNet: HJB PDE solver in JAX via residual minimization
│   └── fictitious_play.py    # NeuralFictitiousPlay + HierarchicalMFGSolver (§VI)
│
├── controller/               # Module C (Paper §3.3)
│   ├── portfolio.py          # α*(z) = ∇V*/(2γκ) → CVaR + leverage projection (SLSQP)
│   └── execution.py          # Alpaca paper trading API
│
├── online/                   # Production operation (Manual §6)
│   ├── airflow_dag.py        # Daily retrain: ingest→noise→encoder→MFG→signal→execute
│   └── regime_detector.py    # LyapunovRegimeDetector: Λ_t = ℒV/V crisis monitor
│
├── data/                     # Data layer (Manual §2)
│   ├── sources/              # Polygon (OHLCV), FRED (macro), SEC EDGAR (13F), NewsAPI
│   ├── kafka/                # Real-time ingestion pipeline
│   └── schema/timescale.sql  # TimescaleDB schema (market_features, noise_params, mfg_eq)
│
├── backtest/
│   └── walk_forward.py       # Walk-forward validation, no lookahead bias
│
├── dashboard/
│   └── app.py                # Streamlit: price fan chart, agent flow, dual noise, Lyapunov
│
├── demo/
│   ├── synthetic_market.py   # Synthetic S&P 500 data generator (no API keys)
│   └── run_egamec.py         # End-to-end demo ← START HERE
│
├── notebooks/                # 14-day theory walkthrough
├── configs/baseline_sp500.yaml
├── requirements.txt
└── setup.sh                  # One-time setup (M0-M4 cost: $0)
```

---

## Theoretical Foundation → Code Map

| Paper Section | Theorem | File |
|---|---|---|
| §I State space $s_t = (p,v,\ell,\kappa,\iota)^5$ | Def 1.1 | `state/market.py` |
| §III Dual noise $\tau_t = \sqrt{\sigma_\tau^2 + \lambda_\eta m_2^\eta}$ | Thm 1 (Cramér-Rao) | `state/noise.py` |
| §IV Groupoid event algebra (Mode I/II/III) | Thm 4 | `events/operators.py` |
| §V Encoder $E$: $\mathcal{I}_t \to z_t \in \mathbb{R}^{64}$ | — | `encoder/model.py` |
| §V Game module $G$: DGM HJB + Neural FP | Thm 2 (Nash) | `game/` |
| §V Controller $C$: $\alpha^* = \nabla V^*/(2\gamma\kappa)$ | Thm 3 (Lyapunov) | `controller/portfolio.py` |
| §VI Hierarchical MFG: macro (MFC) + micro (MFG) | Thm 6 | `game/fictitious_play.py` |
| §VII $\text{RiskIndex}(t) = \mathcal{L}V/V$ | Thm 7 | `online/regime_detector.py` |

---

## Installation & Roadmap

```bash
# Setup (M0-M4 entirely free)
bash setup.sh
cp .env.example .env  # fill FRED + Alpaca (both free)

# Demo (no API keys)
python demo/run_egamec.py

# Dashboard
streamlit run dashboard/app.py
```

| Milestone | Deliverable | Cost |
|---|---|---|
| **M0** — Infrastructure | TimescaleDB + Kafka (Docker); Polygon free tier | **$0** |
| **M1** — Data pipeline | S&P 500 daily bars + 13F + FRED + news embeddings | **$0** |
| **M2** — Encoder v1 | Trained on 2015–2023; latent state stable | **$0** (local GPU) |
| **M3** — MFG stub | DGM on synthetic; fictitious play converges | **$0** |
| **M4** — Paper trading | Full pipeline; 30-day backtest Sharpe > 0 | **$0** (paper account) |
| **M5** — Backtest suite | Walk-forward 2015–2024; Sharpe > 1.5; MDD < 20% | ~$200 cloud GPU |
| **M6** — Live paper | 60-day paper trading; predicted vs. actual flow | **$0** |
| **M7** — Production | Real capital; IBKR FIX; full monitoring | IBKR fees |

---

## Performance Targets

| Metric | Target | Benchmark (SPY) |
|---|---|---|
| Annualized Sharpe | > 1.5 | ~0.7 |
| Max Drawdown | < 20% | — |
| MFG Pred R² | > 0.05 | Factor models ~0.01 |
| Daily Turnover | < 30% | — |
| CVaR 5% | < 3%/day | Hard constraint |
| Regime detection lead | > 10 days | VIX threshold: ~0 days |

---

## Research Paper

The companion paper formalizes 7 original theorems:

> **[E-Game-C: Neural Mean-Field Game Architecture for Financial Market World Models](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance/blob/main/paper_draft_v1.pdf)**
> HongJin HE · Alpha Flow Research / HKUST / Stanford IHP
> *Target: NeurIPS 2026 Workshop on ML for Finance · Deadline: August 29, 2026*

Full mathematical framework (7 theorems, 32 pages):
→ [mathmatical-framework-for-world-models-in-quant-finance](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance)

---

## Citation

```bibtex
@article{he2026egamec,
  title   = {E-Game-C: Neural Mean-Field Game Architecture for Financial Market World Models},
  author  = {HE, HongJin},
  journal = {NeurIPS 2026 Workshop on Machine Learning for Finance},
  year    = {2026},
  url     = {https://github.com/hongjin-he/us-equity-world-model}
}
```

---

<div align="center">
<sub>Alpha Flow Research · HKUST · Stanford IHP · July 2026</sub>
</div>
