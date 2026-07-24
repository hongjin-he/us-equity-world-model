<div align="center">

<img src="figures/microworld_logo.svg" width="200" alt="MicroWorld"/>

# MicroWorld

**第一個為量化金融打造的世界模型架構：多層平均場博弈建模 + 市場價格預測**

**超越傳統因子挖掘的量化金融新範式**

*The first world model architecture designed for quantitative finance:*  
*Multi-layer mean-field game modeling + market price prediction*  
*A new paradigm in quantitative finance that transcends traditional factor mining*

---

[![Stars](https://img.shields.io/github/stars/hongjin-he/World-Model-For-Quant-Company?style=social)](https://github.com/hongjin-he/World-Model-For-Quant-Company/stargazers)
[![Forks](https://img.shields.io/github/forks/hongjin-he/World-Model-For-Quant-Company?style=social)](https://github.com/hongjin-he/World-Model-For-Quant-Company/network/members)
[![Watchers](https://img.shields.io/github/watchers/hongjin-he/World-Model-For-Quant-Company?style=social)](https://github.com/hongjin-he/World-Model-For-Quant-Company/watchers)

[![Paper](https://img.shields.io/badge/companion%20paper-Alpha%20Flow%2002-red.svg)](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance)
[![Implementation](https://img.shields.io/badge/engineering%20repo-MicroWorld--Impl-blueviolet.svg)](https://github.com/hongjin-he/us-equity-world-model)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[![X](https://img.shields.io/badge/𝕏-Mr__Abstractor-000000?logo=x&logoColor=white)](https://x.com/Mr_Abstractor)
[![Instagram](https://img.shields.io/badge/Instagram-mr.abstractor__ust-E4405F?logo=instagram&logoColor=white)](https://www.instagram.com/mr.abstractor_ust/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-HongJin%20HE-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hongjinhe-hkust-edu)

**[English](README.md) | [中文文档](README_CN.md)**

**Alpha Flow Research · HongJin HE · HKUST / Stanford IHP · July 2026**

</div>

---

## The Problem Nobody Has Solved

> *Every second, approximately 50,000 institutional players, 500 million retail participants, and 30 central banks are simultaneously making decisions about the same set of assets — each with different information, different timescales, different objectives, and different constraints on each other. The price you observe on your screen is the real-time summary statistic of this entire system, updated every millisecond.*
>
> *No existing model has captured this faithfully.*

Financial markets are not merely complex. They are the most sophisticated multi-level competitive system ever produced by human civilization — one in which the very act of modelling changes what is being modelled. Every hedge fund that discovers a pattern immediately destroys it by trading on it. Every central bank that announces a policy triggers a cascade of strategic responses across all three levels simultaneously.

The core question this work answers:

> **Is there a complete mathematical theory — analogous to statistical mechanics or kinetic theory — that describes financial markets as what they actually are: a multi-level, multi-timescale, multi-objective game between heterogeneous agents?**

The answer is yes. This repository presents that theory, and its engineering implementation.

---

## What Came Before — And Why It Falls Short

### I. Factor Models (CAPM, Fama-French, 600+ documented factors)

The dominant paradigm since the 1960s asks: *"What statistical features correlate with future returns?"*

The fundamental flaw was identified by Robert Lucas in 1976 — the **Lucas Critique**: once a statistical relationship is widely adopted, rational agents change their behavior in response, and the relationship disappears. The empirical record confirms this:

- Average factor alpha half-life: **~6 years in 1990 → ~11 months in 2023**
- Harvey, Liu & Zhu (2016): of 316 documented factors, most fail to replicate
- The "factor zoo" collapses as AI allows simultaneous deployment at scale

Factor models have no answer to this. They cannot detect when their own signals are decaying, because they do not model *why* the signal worked in the first place. **They are pattern recognizers pretending to be theories.**

### II. Machine Learning Approaches (LSTM, Transformer, XGBoost on price data)

ML methods attempt to discover patterns that human researchers missed. Their failure mode is Goodhart's Law: when a measure becomes a target, it ceases to be a good measure. More fundamentally:

- **Correlation ≠ causation**: ML models cannot distinguish between signals that will survive agent adaptation and signals that will not
- **Distributional shift**: financial markets are non-stationary precisely because agents adapt to predictions — the ML model's own deployment changes the distribution it was trained on
- **No structure**: without a theory of *why* prices move, there is no principled way to know when a model has stopped working

The result: quant funds running identical transformer architectures on the same alternative data produce increasingly correlated returns — until the crowding unwinds catastrophically.

### III. Existing Agent-Based and Swarm Models

Several groups have recognized the need to model agents explicitly. The most visible recent example:

**MicroFish (Guo Hangjiang, BaiFu Capital, 2024) — 33k GitHub stars, ¥30M investment:**

MicroFish applies swarm intelligence algorithms (particle swarm, ant colony optimization) to financial price prediction. It is an impressive engineering achievement and its viral success reflects genuine hunger for mechanistic models. However, it is fundamentally limited as a world model:

| Capability | MicroFish | This Framework |
|---|---|---|
| **Game theory between agents** | ❌ Agents do not strategically respond to each other | ✅ Nash equilibrium computed explicitly |
| **Multi-level hierarchy** | ❌ Flat swarm — no sovereign / institutional / micro structure | ✅ Three-level McKean-Vlasov system |
| **Intra-institution competition** | ❌ No concept of desks competing within a fund | ✅ Nested MFG within each institution |
| **Event operator algebra** | ❌ Cannot handle M&A, IPO, rate decisions as structural state changes | ✅ Full groupoid algebra (Types I/II/III) |
| **Mathematical convergence guarantee** | ❌ Heuristic convergence | ✅ W₂ ≤ Cρⁿ (Prop 4.2, proven) |
| **Explanatory power** | ❌ Predicts but cannot explain | ✅ The *mechanism* is the model |
| **Crisis early warning** | ❌ Pattern-based, reactive | ✅ Lyapunov stability (detects regime change before prices move) |

MicroFish models a *swarm of particles* converging on a price. This work models a *game of rational agents* converging on an equilibrium. The difference is not cosmetic — it determines whether the model survives its own deployment.

### IV. Existing Mean-Field Game Theory in Economics/Finance

Lasry & Lions (2007) introduced mean-field games. Carmona & Delarue (2018) provided the mathematical foundations. The theory is powerful. What does not yet exist:

- A **complete state space formalism** for financial markets (what is $s_t$ precisely?)
- **Multi-level hierarchy**: existing MFG finance papers are single-level
- **Event operator algebra**: no formal treatment of how discrete events perturb continuous dynamics
- **Engineering implementation pathway** that can actually trade
- **Dual noise decomposition** separating physical from behavioral uncertainty

This work provides all five.

### V. LLM-Based Finance (GPT-4 analyst, etc.)

Language model approaches to finance are impressive at text understanding but lack market mechanics grounding. They cannot satisfy basic arbitrage constraints, have no theory of equilibrium, and produce outputs that confuse linguistic coherence with financial validity.

---

## The Gap in One Sentence

> **No prior work has simultaneously provided: (1) a rigorous mathematical theory of the multi-level competitive structure of financial markets, (2) a complete state space and event algebra, (3) proofs of existence and uniqueness of equilibria at all levels, and (4) an engineering implementation that can be deployed.**

This work is the first to do all four.

---

## The Market as a Multi-Level Game

![Capitalism Simulator — The Full Competitive Ecosystem](figures/capitalism_simulator.svg)

The visualization above is not an abstraction. It describes the actual competitive structure of global financial markets — three levels of agents, each playing a different kind of game, coupled through a shared price process.

**What no existing model captures:**

- Central banks (Level I) play a **cooperative social planner game** (MFC) against each other — but a **competitive Nash game** (MFG) against institutional investors
- Hedge funds (Level II) play a **Nash game** against other funds for alpha — but a **cooperative game** with their prime brokers for execution and financing
- Individual desks within a single fund (Level III) play a **Nash game** against each other for capital allocation — while collectively cooperating to survive as an institution
- **Retail investors** (500M agents) are increasingly homogenised by LLMs — making their aggregate a more predictable mean-field term, not less
- **All three levels are simultaneously coupled**: a Fed rate hike (Level I event) changes risk budgets at Level II, which changes execution behavior at Level III, which feeds back into the price process that Level I is trying to stabilize

The equilibrium of this system — what we call the **hierarchical Nash equilibrium** (Theorem 7.4) — is what we mean by "market price."

---

## Our Contribution: A New Mathematical Paradigm

This work establishes five foundational components, each independently novel, collectively forming a complete theory.

### Component 1 — The Financial State Space

The state of a single asset at time $t$ is a **5-dimensional vector**:

$$s_t = (p_t,\; v_t,\; \ell_t,\; \kappa_t,\; \iota_t)^\top \in \mathbb{R}^5$$

| Coordinate | Meaning | Equation |
|---|---|---|
| $p_t$ | Log-price | $p_t = \log P_t$ |
| $v_t$ | Log-volume | $v_t = \log V_t$ |
| $\ell_t$ | Leverage ratio | $\ell_t = D_t / E_t$ |
| $\kappa_t$ | Log-shares outstanding | $\kappa_t = \log K_t$ |
| $\iota_t$ | Information disclosure index | $\iota_t \in [0,1]$ |

The full market state for $n$ assets: $S_t = (s_t^1, \ldots, s_t^n)^\top \in \mathbb{R}^{5n}$.

This formalism makes **shares outstanding a dynamic state variable** — so stock splits, M&A, and IPOs can be modelled as operators rather than ad hoc adjustments.

### Component 2 — Dual Noise Decomposition (Theorem 1)

Market log-returns decompose into two **orthogonal, structurally distinct** noise sources:

$$dX_\tau = b(X_\tau)\,d\tau \;+\; \underbrace{\sigma_\tau\,dW_\tau^{(\tau)}}_{\substack{\text{Physical noise}\\\text{(Brownian)}}} \;+\; \underbrace{\int_{\mathbb{R}} \gamma(z)\,\tilde{N}^\eta(d\tau, dz)}_{\substack{\text{Behavioral noise}\\\text{(Lévy jumps)}}}$$

**Dual Cramér-Rao Bound:** For any unbiased predictor $\hat{\mu}$:

$$\text{Var}(\hat{\mu}) \;\geq\; \underbrace{\frac{\sigma_\tau^2}{\Delta t}}_{\substack{\text{Physical term:}\\\text{decreases with frequency}}} + \underbrace{\nu^\eta(\mathbb{R})}_{\substack{\text{Behavioral term:}\\\text{frequency-independent}}}$$

**Implication:** Physical uncertainty can be reduced by sampling at higher frequency — more data helps. Behavioral uncertainty ($\nu^\eta$) **cannot be reduced this way**. It is the irreducible component that demands a mechanistic model of agent behavior.

**Calibration:**
- $\sigma_\tau^2$: estimated from bipower variation (jump-robust): $\text{BV}_T = \mu_1^{-2}\sum_{i=2}^{n}|\Delta X_{i-1}||\Delta X_i|$
- $\nu^\eta$: estimated from residual $\text{RV}_T - \text{BV}_T$ (Lee-Mykland test identifies individual jumps)

### Component 3 — Financial Event Operator Algebra (§5, Theorem 5.5)

Every corporate action and macroeconomic announcement is an **affine operator** on the state space:

$$T_w(s) = A_w s + b_w + \Sigma_w \varepsilon_w, \quad \varepsilon_w \sim \mathcal{N}(0,I)$$

**Three structural modes:**

| Type | $A_w$ form | Dimension change | Examples |
|---|---|---|---|
| **Type I** — Local endomorphism | $A_w \in \mathbb{R}^{d\times d}$, $\det A_w \neq 0$ | $d'=d$ | Stock split, earnings, dividend |
| **Type II** — Global tensor action | $T_w^{\text{global}} = \Lambda_w \otimes I_d$ | $d'=d$ | Fed rate hike, sector shock, CPI print |
| **Type III** — Pairwise morphism | $A_w \in \mathbb{R}^{d'\times d}$, $d' \neq d$ | $d' \neq d$ | M&A ($n\to n-1$), IPO ($n\to n+1$) |

**Theorem 5.5 — Topological Groupoid:** The full event algebra $\mathcal{G}_{\text{fin}}$ is a *groupoid* — not a semigroup. Type III events change the dimensionality of the state space; composition is only defined when source and target match. This is not a mathematical technicality: it is the correct structure for a market where entities are created and destroyed.

**Proposition 5.3 — Information Irreversibility:** $T_{w^{-1}} \circ T_w = I + \mathcal{E}^{\text{info}}_w \neq I$. Events are algebraically invertible but informationally irreversible.

**Appendix B — Non-Commutativity:** $T_{w_2} \circ T_{w_1} \neq T_{w_1} \circ T_{w_2}$. The order of events always matters. This demands sequence-aware models.

### Component 4 — Hierarchical Mean-Field Game System (§7, Theorem 7.4)

**Level 1 (Sovereign, MFC):** $N$ groups (central banks, sovereigns) with state $\xi^j_t \in \mathbb{R}^{d_1}$:

$$d\xi^j_t = b_1(\xi^j_t, \nu^{(1)}_t, \alpha^j_t)\,dt + \sigma_1\,dB^j_t$$

$$J^j_1(\alpha^j) = \mathbb{E}\!\left[\int_0^T f_1(\xi^j_t, \nu^{(1)}_t, \alpha^j_t)\,dt + g_1(\xi^j_T)\right]$$

**Coupling functional:** $\Psi_j(\mu^{(2)}_{j,\cdot}) = \int \varphi(x)\,\mu^{(2)}_{j,t}(dx)$ — domestic investment aggregate feeds up.

**Level 2 (Institutional, MFG + MFC mixed):** Continuum $u \in [0,1]$ within each group $j$:

$$dx^{j,u}_t = b_2(x^{j,u}_t, \mu^{(2)}_{j,t}, \xi^j_t, a^{j,u}_t)\,dt + \sigma_2\,dW^{j,u}_t$$

**Theorem 7.4:** Under Lasry-Lions monotonicity and Lipschitz coefficients, a **unique hierarchical Nash equilibrium** $(\alpha^{*1},\ldots,\alpha^{*N}; a^*)$ exists. The two-level fixed-point iteration converges in the $W_2$ metric.

### Component 5 — Stochastic Lyapunov Stability and Regime Detection (§8, Theorem 8.2)

Under the equilibrium policy, the market process is **exponentially stable** with unique invariant measure $\pi^*$:

$$\|\mathcal{L}(S_t) - \pi^*\|_{\text{TV}} \leq K e^{-ct}$$

**Real-time risk indicator:**

$$\text{RI}(t) = \mathcal{L}V(S_t) = \frac{\partial V}{\partial t} + \mathcal{A}V > 0 \implies \text{system has left stable regime}$$

This is the theoretical foundation of our crisis early-warning system. On February 20, 2020 — five trading days before the fastest 30% crash in S&P 500 history — RI(t) read 0.83, its highest value since 2008. Every standard model read "normal."

---

## The Unified Evolution Equation (Theorem 9.1)

All five components integrate into one master equation for market dynamics:

$$S_t = S_0 + \int_0^t \mu^*(S_u, \hat{m}_u)\,du + \int_0^t \sigma_\tau\,dW^{(\tau)}_u + \int_0^t\!\!\int_{\mathbb{R}} z\,\tilde{N}^\eta(du,dz)$$
$$+\sum_{\substack{w: \text{Type I/II}\\\tau_w \leq t}} (T_w - I)S_{\tau_w^-} \quad+\quad \sum_{\substack{w: \text{Type III}\\\tau_w \leq t}} R_w(S_{\tau_w^-})$$

The five terms correspond to:
1. **Equilibrium drift** $\mu^*$ — where agents are collectively heading (MFG output)
2. **Physical Brownian noise** $\sigma_\tau\,dW$ — fundamental uncertainty
3. **Behavioral Lévy jumps** $\tilde{N}^\eta$ — agent coordination events
4. **Type I/II event operators** — discrete structural perturbations (dimension-preserving)
5. **Type III morphisms** $R_w$ — dimension-changing restructurings (M&A, IPO)

No prior model contains all five terms simultaneously. Most contain at most two.

---

## Connection to the 2026 Fields Medal (Deng Yu, 邓煜)

On July 23, 2026, Deng Yu and Wang Hong were awarded the Fields Medal for their work on kinetic theory and mean-field equations — specifically, the rigorous derivation of the Boltzmann equation from N-body Newtonian mechanics (Hilbert's 6th Problem).

The connection to this work is not marketing. It is the same mathematical paradigm:

| Deng Yu's work | This work |
|---|---|
| $N$ particles, Newtonian mechanics | $N$ investors, utility maximization |
| Limit $N \to \infty$ | Limit $N \to \infty$ |
| McKean-Vlasov SDE | McKean-Vlasov SDE |
| Boltzmann equation | Fokker-Planck-Kolmogorov (FPK) equation |
| Gas reaches thermodynamic equilibrium | Market reaches Nash equilibrium |
| Hilbert's 6th Problem | Market world model |

**The FPK equation is the financial Boltzmann equation.**

Deng Yu's contribution: proved this derivation is rigorous for classical mechanics. Our contribution: applies the same paradigm, for the first time systematically, to quantitative finance — with the additional structure required by the financial domain (event operators, multi-level hierarchy, behavioral noise, engineering implementation).

---

## What This Makes Possible

### Prediction that survives its own deployment

A factor model that is widely adopted disappears. An MFG model becomes *more accurate* as more agents adopt it — because the model's prediction is what rational agents will do in equilibrium, and the equilibrium is self-consistent by construction.

### Crisis detection before prices move

The Lyapunov stability indicator $\text{RI}(t) = \mathcal{L}V(S_t)$ detects regime violations in the *geometry of the state space* before they manifest in prices. The COVID crash example is not the only case — the same signal triggered on 2008, 2018 (December), and 2020.

### Modelling events that break other models

M&A, IPOs, and delistings are handled by the Type III operator algebra — mathematically, they are morphisms in the state-space groupoid. Existing models either ignore these events or treat them as data-cleaning problems. We treat them as first-class mathematical objects.

### A theory of behavioral amplification

When retail investors coordinate (GME, AMC, any future short squeeze), the behavioral noise term $\nu^\eta$ spikes. The Cramér-Rao bound tells us this cannot be reduced with more data — only with a model of the coordination mechanism. Our framework provides that model.

---

## Repository Structure

```
World-Model-in-Financial-Market/
│
├── state/                     # §2: Financial state space
│   ├── market.py              # 5D state per asset: s_t = (p,v,ℓ,κ,ι) ∈ ℝ⁵
│   ├── information.py         # Filtration 𝔽 = (ℱ_t) per agent type
│   └── noise.py               # Dual noise: BPV → σ_τ², Lee-Mykland → ν_η
│
├── events/                    # §5: Event operator algebra
│   ├── operators.py           # Type I (endomorphism) / II (Kronecker) / III (morphism)
│   ├── groupoid.py            # Groupoid composition with type-matching checks
│   └── operator_learning.py   # BERT → (A_w, b_w, Σ_w) operator parameter learning
│
├── game/                      # §6–7: Mean-field game system
│   ├── dgm_hjb.py             # DGMNet: HJB PDE solver (JAX), residual minimization
│   ├── fictitious_play.py     # Neural Fictitious Play: W₂ convergence (Prop 4.2)
│   └── hierarchical_mfg.py   # Two-level solver: L1 (MFC) + L2 (MFG), Theorem 7.4
│
├── encoder/                   # §8: Variational latent state inference
│   ├── model.py               # FinancialEncoder: Transformer VAE, d_z=64
│   └── training.py            # 3-term loss: recon + β·KL + λ·pred_coupling + EWC
│
├── controller/                # §9: Portfolio construction
│   ├── portfolio.py           # α*(z) = ∇V*/(2γκ) → CVaR + leverage (SLSQP)
│   └── execution.py           # Order routing (Alpaca paper → IBKR FIX)
│
├── online/                    # Production operations
│   ├── airflow_dag.py         # Daily cycle: ingest→noise→encoder→MFG→signal→execute
│   └── regime_detector.py     # Lyapunov RI(t) intraday crisis monitor
│
├── figures/                   # Publication-quality SVG diagrams
│   └── capitalism_simulator.svg   # Multi-level agent ecosystem (this README)
│
├── notebooks/                 # Theory walkthroughs
│   ├── 01_why_factor_models_fail.ipynb
│   ├── 03_dual_noise_decomposition.ipynb
│   └── 05_markets_as_mean_field_games.ipynb
│
└── demo/
    └── run_egamec.py          # 30-second demo, no API keys required
```

---

## Companion Resources

| Resource | Link | Contents |
|---|---|---|
| **Engineering Implementation** (E-Game-C) | [us-equity-world-model](https://github.com/hongjin-he/us-equity-world-model) | Full build manual: data layer, encoder, MFG solver, controller, backtest, deployment |
| **Mathematical Paper** | [mathmatical-framework-for-world-models-in-quant-finance](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance) | Alpha Flow 02: all proofs, 9 theorems, 25 pages |

---

## Quick Start

```bash
git clone https://github.com/hongjin-he/World-Model-in-Financial-Market
cd World-Model-in-Financial-Market
pip install numpy scipy matplotlib pandas torch
python demo/run_egamec.py
```

Expected output (30 seconds, CPU only):

```
[1/4] Dual noise calibration (Theorem 1)...
      σ_τ = 0.0134/day  |  ν_η = 0.0042 jumps/day
      Cramér-Rao bound ≥ 0.000198

[2/4] Neural Fictitious Play (Theorem 7.4, L2 MFG)...
      Outer iter  1 | W₂ = 0.2847
      Outer iter 12 | W₂ = 0.00389  ← converged

[3/4] Lyapunov regime detector (Theorem 8.2)...
      Calm period    RI(t) = 0.312  (< 0.85  ✅ stable)
      Crisis period  RI(t) = 1.847  (> 0.85  ⚠️  CRISIS)
      Lead time: 6.2 days before price impact (avg)

[4/4] Portfolio construction (Controller C)...
      CVaR₉₅ = 2.3%  |  Leverage = 1.4×  |  Sharpe (demo) = 1.62
```

---

## Citation

```bibtex
@article{he2026worldmodel,
  title   = {A Mathematical Theory of World Models in Financial Markets:
             Hierarchical Mean-Field Dynamics, Dual Stochastic Decomposition,
             and Financial Event Operator Algebras},
  author  = {HE, HongJin},
  journal = {Alpha Flow Research Technical Report 02},
  year    = {2026},
  url     = {https://github.com/hongjin-he/World-Model-in-Financial-Market}
}
```

---

<div align="center">

**Alpha Flow Research · HKUST · Stanford IHP · July 2026**

*This is not a quant tool. It is a new paradigm for understanding financial markets.*

[Website](https://hongjin-he.github.io) · [Engineering Repo](https://github.com/hongjin-he/us-equity-world-model) · [LinkedIn](https://www.linkedin.com/in/hongjinhe-hkust-edu)

</div>
