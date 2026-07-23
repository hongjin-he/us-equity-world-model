# US Equity World Model — Alpha Flow E-Game-C System

**Alpha Flow Research · HongJin HE · HKUST / Stanford IHP · July 2026**

A production implementation of the E-Game-C architecture for US equity markets.
Treats price discovery as an emergent outcome of heterogeneous agents competing under information asymmetry,
modeled as a Mean-Field Game equilibrium.

> **Theory**: [Mathematical Framework for World Models in Quantitative Finance](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance)
> (7 theorems: Itô integral equations, dual noise decomposition, groupoid event algebra, hierarchical MFG, Lyapunov stability)

---

## Architecture: E-Game-C

```
Raw observations I_t (OHLCV + macro + news + 13F)
        │
        ▼
┌─────────────────────────────────┐
│  E — Encoder (Transformer VAE)  │  §3
│  I_t → z_t ~ N(μ_φ, Σ_φ)      │  encoder/
└─────────────────────────────────┘
        │  z_t ∈ ℝ⁶⁴
        ▼
┌─────────────────────────────────┐
│  Game — MFG Equilibrium Solver  │  §4
│  DGM HJB + Neural Fictitious    │  game/
│  Play → (V*, m*)               │
│  Hierarchical: macro + micro    │
└─────────────────────────────────┘
        │  V*(z_t), m*(z_t)
        ▼
┌─────────────────────────────────┐
│  C — Controller                 │  §5
│  a* = ∇V* / (2γκ)              │  controller/
│  → portfolio weights w*         │
│  CVaR constraint projection     │
└─────────────────────────────────┘
        │
        ▼
   Alpaca Paper → IBKR FIX
```

---

## Directory Structure

```
us-equity-world-model/
│
├── state/                    # Mathematical state space (Theory §I-III)
│   ├── market.py             # 5D per-asset state: (p,v,ℓ,κ,ι) ∈ ℝ⁵
│   ├── information.py        # Information filtration F_t per agent
│   └── noise.py              # Dual noise calibration: σ_τ (BPV) + λ_η (Lee-Mykland)
│
├── events/                   # Event operator algebra (Theory §IV)
│   └── operators.py          # Mode I/II/III operators; Groupoid structure for M&A/spin-off
│
├── data/                     # Data layer (Manual §2)
│   ├── sources/
│   │   ├── polygon.py        # OHLCV + trades (Polygon.io, free tier)
│   │   ├── sec_13f.py        # 13F institutional holdings (SEC EDGAR, free)
│   │   ├── fred.py           # Macro state: DFF/CPI/VIX/T10Y2Y (FRED, free)
│   │   └── news.py           # Headlines → 384-dim embeddings (local, free)
│   ├── kafka/
│   │   └── producer.py       # Kafka ingestion pipeline
│   ├── schema/
│   │   └── timescale.sql     # TimescaleDB schema (market_features, noise_params, mfg_equilibrium)
│   └── features/             # Feature engineering (BPV, RSI, jump flags)
│
├── encoder/                  # Module E (Manual §3)
│   ├── model.py              # FinancialEncoder: Transformer backbone + VAE head
│   └── training.py           # 3-term loss (recon + KL + pred coupling) + EWC
│
├── game/                     # Game Module (Manual §4)
│   ├── dgm_hjb.py            # DGM net for HJB PDE in JAX; hjb_residual loss
│   └── fictitious_play.py    # Neural Fictitious Play; HierarchicalMFGSolver
│
├── controller/               # Module C (Manual §5)
│   ├── portfolio.py          # EGameCController: CVaR + leverage constraint projection
│   └── execution.py          # Alpaca paper trading API
│
├── online/                   # Production operation (Manual §6)
│   ├── airflow_dag.py        # Daily retrain DAG (6 PM EST Mon-Fri)
│   └── regime_detector.py    # LyapunovRegimeDetector: RiskIndex = LV/V
│
├── backtest/                 # Backtesting (Manual §8)
│   └── walk_forward.py       # Walk-forward validation; performance metrics
│
├── dashboard/                # Visualization (Manual §9)
│   └── app.py                # Streamlit: price fan chart, agent flow, dual noise, Lyapunov gauge
│
├── configs/
│   └── baseline_sp500.yaml   # Proxy profiles for RenTech, TwoSigma, AQR, Vanguard, BlackRock
│
├── setup.sh                  # One-time environment setup (M0-M4 cost: $0)
└── .env.example              # API key stubs
```

---

## Mathematical Foundation → Code Mapping

| Theory (§) | Theorem | Implementation |
|---|---|---|
| §I State space | $s_t = (p,v,\ell,\kappa,\iota)^5$ | `state/market.py` |
| §II Itô integral | No-arbitrage via adaptedness | `game/fictitious_play.py` (Euler-Maruyama) |
| §III Dual noise | $\tau_t = \sqrt{\sigma_\tau^2 + \lambda_\eta m_2^\eta}$ | `state/noise.py` → `DualNoiseCalibrator` |
| §III Cramér-Rao | Prediction lower bound | `state/noise.py` → `cramer_rao_bound` |
| §IV Event algebra | Mode I/II/III operators | `events/operators.py` |
| §V E-Game-C | $x_t \xrightarrow{E} z_t \xrightarrow{\text{Game}} V^* \xrightarrow{C} w^*$ | `encoder/`, `game/`, `controller/` |
| §VI Hierarchical MFG | $a^* = \phi^{(1)}(\xi) + \phi^{(2)}(x,\mu) + O(\varepsilon)$ | `game/fictitious_play.py` → `HierarchicalMFGSolver` |
| §VII Lyapunov | $\text{RiskIndex}(t) = \mathcal{L}V/V$ | `online/regime_detector.py` |

---

## Implementation Roadmap

| Milestone | Deliverable | Cost |
|---|---|---|
| **M0** — Infrastructure | Docker: TimescaleDB + Kafka; Polygon free tier | **Free** |
| **M1** — Data pipeline | S&P 500 daily bars + 13F + FRED ingested | **Free** |
| **M2** — Encoder v1 | Trained on 2015–2023 daily data; latent stable | **Free** (local GPU) |
| **M3** — MFG stub | DGM HJB on synthetic data; fictitious play converges | **Free** |
| **M4** — Paper trading | End-to-end pipeline; 30-day backtest Sharpe > 0 | **Free** (paper account) |
| **M5** — Backtest suite | Walk-forward 2015–2024; Sharpe > 1.0; MDD < 20% | ~$200 (cloud GPU) |
| **M6** — Live paper | 60-day paper; predicted vs. actual flow tracking | **Free** |
| **M7** — Production | Real capital; IBKR FIX; full monitoring | IBKR fees |

---

## Quick Start

```bash
# 1. Environment
bash setup.sh

# 2. Fill in .env (FRED and Alpaca paper are free immediately)
cp .env.example .env && vim .env

# 3. Start dashboard
streamlit run dashboard/app.py

# 4. Run backtest (once encoder is trained)
python -m backtest.walk_forward

# 5. Start daily Airflow DAG
airflow dags trigger alpha_flow_daily
```

---

## Performance Targets

| Metric | Target | Benchmark |
|---|---|---|
| Annualized Sharpe | > 1.5 | SPY ~0.7 |
| Max Drawdown | < 20% | — |
| MFG Pred R² | > 0.05 | Factor models ~0.01 |
| Daily Turnover | < 30% | Controls TC |
| CVaR 5% | < 3%/day | Hard constraint |
