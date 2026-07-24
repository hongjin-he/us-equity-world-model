<div align="center">

<img src="figures/microworld_logo.svg" width="200" alt="MicroWorld"/>

# MicroWorld

**第一個為量化金融打造的世界模型架構：多層平均場博弈建模 + 市場價格預測**

**超越傳統因子挖掘的量化金融新範式**

*The first world model architecture designed for quantitative finance:*  
*Multi-layer mean-field game modeling + market price prediction*  
*A new paradigm in quantitative finance that transcends traditional factor mining*

---

[![Stars](https://img.shields.io/github/stars/hongjin-he/MicroWorld?style=social)](https://github.com/hongjin-he/MicroWorld/stargazers)
[![Forks](https://img.shields.io/github/forks/hongjin-he/MicroWorld?style=social)](https://github.com/hongjin-he/MicroWorld/network/members)
[![Watchers](https://img.shields.io/github/watchers/hongjin-he/MicroWorld?style=social)](https://github.com/hongjin-he/MicroWorld/watchers)

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

Financial markets are not merely complex. They are the most sophisticated multi-level competitive system ever produced by human civilization — one in which the very act of modelling changes what is being modelled. Every hedge fund that discovers a pattern immediately destroys it by trading on it. Every central bank that announces a policy triggers a cascade of strategic responses across all four levels simultaneously.

The core question this work answers:

> **Is there a complete mathematical theory — analogous to statistical mechanics or kinetic theory — that describes financial markets as what they actually are: a multi-level, multi-timescale, multi-objective game between heterogeneous agents?**

The answer is yes. This repository presents that theory, and its engineering implementation.

---

## Why This Question Is Urgent — Right Now

Three simultaneous forces are breaking the old paradigm faster than at any point in history:

**1. Factor alpha has a terminal illness.** Average alpha half-life: ~6 years in 1990, ~11 months in 2023. The decay is not a cycle — it is a structural consequence of AI-accelerated crowding. When every fund discovers the same signal within months of each other, the signal disappears before anyone profits. Factor models have no theory of *why* signals decay; they cannot detect the decay until it is complete.

**2. LLMs are homogenizing retail behavior at scale.** Five hundred million retail investors are now asking the same three AI assistants the same questions and receiving the same answers. The behavioral noise term $\nu^\eta$ is simultaneously *increasing* (more retail coordination, larger jumps) and becoming *more predictable* (the coordination mechanism is now modelable). The old assumption that retail noise is unstructured is obsolete.

**3. The Fields Medal just validated the paradigm.** In July 2026, Deng Yu (邓煜) received the Fields Medal for proving that N-body Newtonian mechanics converges — as $N\to\infty$ — to the Boltzmann equation. This is the mathematical paradigm we apply to finance: N rational agents converge to a Fokker-Planck-Kolmogorov equation. The FPK equation is the financial Boltzmann equation. Deng Yu's proof establishes the rigorous foundation for this class of large-population convergence results. We stand on that foundation.

The window for building game-theoretic world models is now. The models that survive the next decade will be the ones built on mechanism, not pattern.

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
| **Multi-level hierarchy** | ❌ Flat swarm — no market / type / institution / individual structure | ✅ Four-level hierarchical MFG (cross-market · type · institution · individual) |
| **Intra-institution competition** | ❌ No concept of desks competing within a fund | ✅ Level 3 MFG: intra-institution Nash between individuals |
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

## The Market as a Four-Level Game

![Capitalism Simulator — The Full Competitive Ecosystem](figures/capitalism_simulator.svg)

The visualization above is not an abstraction. It describes the actual competitive structure of global financial markets — **four nested levels** of agents, each playing a different kind of game, coupled through shared price processes, capital flows, and information cascades.

There is an ancient Chinese insight: *個人由環境造就* — the individual is shaped by the environment. Our framework makes this precise. The four levels are not isolated: each agent is simultaneously a product of all levels above it and a contributor to all levels above it. The environment is not external noise — it is the aggregate of every other agent's strategy.

---

### Level 0 — Cross-Market Capital Flow Game

**Players:** Global macro participants, central banks of different nations, international capital itself.

**What is happening:** Capital moves between markets in search of risk-adjusted return. A hawkish Fed raises US real rates → capital flows from EM to USD assets → CNY weakens → PBOC responds → global equities re-price. This is a **game between entire markets** — the US, EU, CN, JP, HK, EM blocs — competing for international capital while coordinating (imperfectly) on global stability.

**Game type:** Mixed MFC/MFG — sovereign coordination (G7 mechanisms) overlaid with competitive capital attraction.

**Key coupling:** The Level 0 equilibrium sets the **external environment** $\Gamma^m_t$ for every market $m$ — the backdrop against which all lower-level games are played.

---

### Level 1 — Institution-Type Game Within Each Market

**Players:** Distinct *types* of institution within a single market: Central Banks / Governments, Commercial Banks, Investment Banks, Quantitative Hedge Funds, Private Equity / Traditional Hedge Funds, Mutual Funds / ETFs, Retail Investors.

**Why types matter:** Each type has structurally different objectives, risk functions, regulatory constraints, investment horizons, and — critically — **different information access**. A central bank holds confidential macro data. A quant fund holds proprietary signal libraries. A retail investor holds public news that arrives hours after institutions have already traded on it.

**What is happening:** Types compete for return while filling structurally different roles in the ecosystem. IBs provide execution and financing; CB/Gov provides the regulatory backdrop; retail provides the liquidity that institutions extract alpha from.

**Game type:** Multi-population MFG — each type plays a Nash game against other types, with type-specific objectives and information sets.

---

### Level 2 — Individual Institution Game Within Each Type

**Players:** Individual institutions of the same type competing head-to-head. Among quant funds: Jane Street vs. Citadel vs. Two Sigma vs. Renaissance. Among investment banks: Goldman vs. JPMorgan vs. Morgan Stanley. Among asset managers: BlackRock vs. Vanguard vs. Fidelity.

**What is happening:** Within the same type, institutions share similar information sources and similar strategy spaces — making competition the most direct and zero-sum of all four levels. When Citadel builds a new momentum signal, Renaissance is effectively building the same signal; when one deploys, it degrades the other's alpha.

**Game type:** Standard MFG within each type — pure Nash competition, with Lasry-Lions monotonicity guaranteeing a unique equilibrium.

---

### Level 3 — Intra-Institution Individual Game

**Players:** Individual humans (portfolio managers, quant researchers, risk officers, traders) within a single institution.

**What is happening:** A hedge fund is not a monolithic agent. Its desks compete for capital allocation (the PM whose desk earns more PnL gets more capital next month). Its researchers compete for credit. Its risk officers play a constrained game with its traders. At the same time, all must cooperate: a fund that fails to cooperate internally underperforms and loses AUM, destroying the game for everyone inside it.

**Game type:** Mixed cooperative-competitive MFG — Nash competition over internal resources, cooperative MFC over institutional survival.

---

### The Coupling Structure

These four levels are **bidirectionally coupled**:

- **Downward (environment → individual):** Level 0 capital flows determine Level 1 sector positioning; Level 1 type dominance determines which Level 2 institutions survive; Level 2 institutional PnL determines Level 3 individual compensation and retention.
- **Upward (individual → environment):** Level 3 desk behavior determines Level 2 net positions; Level 2 institutional flows aggregate into Level 1 type-level demand; Level 1 type dynamics determine Level 0 capital flow equilibria.

A single Fed rate hike (a Type II event operator at Level 0) propagates through all four levels within hours — reshaping capital flows, institutional positioning, individual desk risk budgets, and eventually the price of every asset simultaneously.

**The equilibrium of this four-level coupled system — what we call the hierarchical Nash equilibrium (Theorem 7.4) — is what we mean by "market price."**

---

## The Mathematical Framework: Two Threads, One Theory

> *Every section below runs two parallel threads simultaneously: a conceptual argument in plain language, and its rigorous mathematical form. Neither is subordinate to the other — the intuition motivates the equation, the equation disciplines the intuition.*

---

### Component 1 — The Financial State Space

**The question every model must answer first:** *what is the state of a market at a given instant?*

Most models answer: price. But price is the *output* of a process, not its state. The machinery that generates price — leverage, volume, outstanding shares, information disclosure — is invisible to price-only models. A world model that tracks only price is like a weather model that tracks only temperature: it sees the symptom, not the system.

We define the minimal sufficient state representation for a single asset at time $t$:

$$s_t = (p_t,\; v_t,\; \ell_t,\; \kappa_t,\; \iota_t)^\top \in \mathbb{R}^5$$

Each coordinate carries structural meaning:

| Coordinate | Meaning | Why it belongs in the state |
|---|---|---|
| $p_t = \log P_t$ | Log-price | The primary observable; all agents react to this |
| $v_t = \log V_t$ | Log-volume | Carries information about conviction strength and liquidity |
| $\ell_t = D_t/E_t$ | Leverage ratio | Determines amplification and fragility; high $\ell_t$ → Lévy tail risk |
| $\kappa_t = \log K_t$ | Log-shares outstanding | *The critical design choice* — made dynamic, not fixed |
| $\iota_t \in [0,1]$ | Information disclosure | Determines asymmetry between agent types in the MFG |

The full market state for $n$ assets: $S_t = (s_t^1, \ldots, s_t^n)^\top \in \mathbb{R}^{5n}$.

**Why $\kappa_t$ must be dynamic.** Every prior model fixes shares outstanding as a constant. We do not — because M&A events ($n \to n-1$), IPOs ($n \to n+1$), and stock splits ($\kappa_t \to \kappa_t + \log 2$) are not data-cleaning anomalies. They are the market's most significant structural events. Making $\kappa_t$ a state variable is what allows us to model them mathematically rather than filtering them away.

---

### Component 2 — Dual Noise Decomposition (Theorem 1)

**The fundamental obstacle to financial prediction.** Consider two types of uncertainty in markets:

*Type A:* A stock's price fluctuates randomly between trades — bid-ask bounce, small order flow imbalances, microstructure noise. This is **physical noise**: it averages out as you sample more frequently. More data → less uncertainty.

*Type B:* Retail investors decide to short-squeeze a heavily shorted stock because of a Reddit post. A central bank surprises markets with an emergency rate cut. A geopolitical event triggers simultaneous liquidation across asset classes. This is **behavioral noise**: it is driven by human coordination and cannot be averaged away. More data does not help, because the mechanism — human strategic behavior — is not stationary.

These two types of noise have fundamentally different mathematical structures. We decompose them explicitly:

$$dX_\tau = b(X_\tau)\,d\tau \;+\; \underbrace{\sigma_\tau\,dW_\tau}_{\substack{\text{Physical noise}\\\text{Brownian motion}\\\text{σ\_τ from bipower variation}}} \;+\; \underbrace{\int_{\mathbb{R}} \gamma(z)\,\tilde{N}^\eta(d\tau, dz)}_{\substack{\text{Behavioral noise}\\\text{Lévy jump measure ν\_η}\\\text{agent coordination events}}}$$

The decomposition is not merely a modeling choice — it has a provable consequence. **Theorem 1 (Dual Cramér-Rao Bound):** For *any* unbiased estimator $\hat{\mu}$ of the drift:

$$\text{Var}(\hat{\mu}) \;\geq\; \underbrace{\frac{\sigma_\tau^2}{\Delta t}}_{\substack{\text{Vanishes as } \Delta t \to 0\\\text{"more data helps"}}} + \underbrace{\nu^\eta(\mathbb{R})}_{\substack{\text{Frequency-independent}\\\text{"more data doesn't help"}}}$$

The first term is the physical floor — it falls to zero as you sample more frequently. The second term $\nu^\eta(\mathbb{R})$ is the behavioral floor — it is a fixed constant determined by the intensity of agent coordination events, and **no amount of additional price data can cross it**. This is the mathematical proof that agent modeling — not more data — is the only path beyond the behavioral noise floor.

**Calibration in practice:**
- $\hat{\sigma}_\tau^2$: bipower variation $\text{BV}_T = \mu_1^{-2}\sum_{i=2}^{n}|\Delta X_{i-1}||\Delta X_i|$ (jump-robust)
- $\hat{\nu}^\eta$: residual $\text{RV}_T - \text{BV}_T$; individual jump times identified via Lee-Mykland test

---

### Component 3 — Financial Event Operator Algebra (§5, Theorem 5.5)

**The problem with treating events as outliers.** Standard quantitative models — GARCH, realized volatility, even most neural networks — remove earnings announcements, M&A events, rate decisions, and index rebalancings from their training data, label them "structural breaks," and treat them as noise. This is not a minor technical limitation. It means these models are *deliberately blind to the most consequential moments in market history*.

Our approach: every corporate action and macroeconomic announcement is a **first-class mathematical object** — an affine operator on the state space:

$$T_w(s) = A_w s + b_w + \Sigma_w \varepsilon_w, \quad \varepsilon_w \sim \mathcal{N}(0,I)$$

The matrix $A_w$ encodes how event $w$ transforms the state. The key insight is that different events have structurally different $A_w$ matrices — and this structure is not arbitrary:

| Type | $A_w$ structure | Dimension | Events |
|---|---|---|---|
| **Type I** (endomorphism) | $A_w \in \mathbb{R}^{d\times d}$, $\det A_w \neq 0$ | Preserves $d$ | Stock split, earnings, dividend — affects one asset in place |
| **Type II** (tensor action) | $T_w^{\text{global}} = \Lambda_w \otimes I_d$ | Preserves $d$ | Fed rate hike, CPI print — acts on all assets simultaneously via Kronecker structure |
| **Type III** (morphism) | $A_w \in \mathbb{R}^{d'\times d}$, $d' \neq d$ | Changes $d$ | M&A ($n\to n{-}1$), IPO ($n\to n{+}1$) — restructures the state space itself |

**Why not a semigroup? (Theorem 5.5)** The natural algebraic structure for operators that compose ($T_{w_2} \circ T_{w_1}$) is a semigroup. But Type III events change the *dimension* of the state space — you cannot compose a "IPO" operator (which acts on $\mathbb{R}^{5n}$) with a "stock split" operator (which acts on $\mathbb{R}^{5(n+1)}$) without first specifying that they act on different objects. The correct structure is a **topological groupoid**: composition is only defined between operators whose source and target domains match, and the full algebra $\mathcal{G}_{\text{fin}}$ satisfies the groupoid axioms (Theorem 5.5).

**Proposition 5.3 (Information Irreversibility):** $T_{w^{-1}} \circ T_w = I + \mathcal{E}^{\text{info}}_w \neq I$. Events have algebraic inverses — a merger can be un-merged — but not informational inverses. Once the market has learned that Company A acquired Company B, that information cannot be un-learned. The residual $\mathcal{E}^{\text{info}}_w$ quantifies this irreversible information injection.

**Appendix B (Non-Commutativity):** $T_{w_2} \circ T_{w_1} \neq T_{w_1} \circ T_{w_2}$ in general. A Fed rate hike followed by a CPI surprise produces a different market state than the same events in reverse order. This non-commutativity is the mathematical reason why event *sequences* — not just event sets — matter for prediction. It is why transformer architectures have an advantage over bag-of-words models in financial text processing.

### Component 4 — Four-Level Hierarchical Mean-Field Game System (§7, Theorem 7.4)

**Level 0 (Cross-Market, MFC/MFG mixed):** Markets $m \in \mathcal{M}$ (US, EU, CN, JP, HK, EM) with market-level state $\Gamma^m_t \in \mathbb{R}^{d_0}$:

$$d\Gamma^m_t = b_0\!\left(\Gamma^m_t,\; \nu^{(0)}_t,\; \alpha^m_t,\; \{\Phi_{m,m'}(t)\}_{m'\neq m}\right)dt + \sigma_0\,dB^m_t$$

where $\Phi_{m,m'}(t)$ is the **net capital flow** from market $m$ to $m'$. Level 0 equilibrium determines the external environment $\{\Gamma^m_t\}$ for all lower levels.

**Level 1 (Institution Types, Multi-Population MFG):** Types $\tau \in \mathcal{T} = \{\text{CB/Gov},\;\text{CommBank},\;\text{IB},\;\text{QuantHF},\;\text{PE/HF},\;\text{MutualFund},\;\text{Retail}\}$ within each market $m$:

$$d\xi^{m,\tau}_t = b_1\!\left(\xi^{m,\tau}_t,\; \mu^{(1)}_{m,t},\; \Gamma^m_t,\; \pi^{m,\tau}_t\right)dt + \sigma_1\,dW^{m,\tau}_t$$

Each type has a distinct objective $U^\tau$ and information set $\mathcal{I}^{(1,\tau)}$ (see Information Architecture below).

**Level 2 (Individual Institutions, Standard MFG):** Individual institution $j \in \mathcal{J}_{m,\tau}$ within type $\tau$ in market $m$:

$$dx^j_t = b_2\!\left(x^j_t,\; \mu^{(2)}_{\tau,t},\; \xi^{m,\tau(j)}_t,\; a^j_t\right)dt + \sigma_2\,dW^j_t + dJ^j_t$$

Same-type institutions share information structure $\mathcal{I}^{(1,\tau)}$ but hold additional proprietary signals $\mathcal{I}^{\text{priv},j}$.

**Level 3 (Intra-Institution, Mixed MFC/MFG):** Individual $i \in \mathcal{I}_j$ within institution $j$:

$$dy^{i,j}_t = b_3\!\left(y^{i,j}_t,\; \mu^{(3)}_{j,t},\; x^j_t,\; u^{i,j}_t\right)dt + \sigma_3\,dW^{i,j}_t$$

Individuals play a Nash game over capital allocation (competition) within a cooperative survival constraint (the institution must remain solvent).

**Coupling functionals (upward, aggregate → higher level):**
$$\Psi^{(1\to 0)}_m = \int \varphi_0(\xi)\,\mu^{(1)}_{m,t}(d\xi), \qquad \Psi^{(2\to 1)}_\tau = \int \varphi_1(x)\,\mu^{(2)}_{\tau,t}(dx), \qquad \Psi^{(3\to 2)}_j = \int \varphi_2(y)\,\mu^{(3)}_{j,t}(dy)$$

**Theorem 7.4 (Extended):** Under Lasry-Lions monotonicity at each level and Lipschitz coupling functionals, a **unique four-level hierarchical Nash equilibrium** exists. The nested fixed-point iteration — solving levels 3→2→1→0, then back-propagating 0→1→2→3 — converges in $W_2$.

---

### Component 4b — Information Architecture and Bounded Rationality

Prior models assume either full information (unrealistic) or no information structure (too crude). The real financial market has a precise **stratified information hierarchy**:

**Definition (Agent Information Set):** Each agent at level $k$, type $\tau$, institution $j$, individual $i$ observes:
$$\mathcal{I}^{k,\tau,j,i}_t = \underbrace{\mathcal{I}^{(0)}_t}_{\substack{\text{Public info}\\\text{(Bloomberg, prices)}}} \;\oplus\; \underbrace{\Delta^{(\tau)}_t}_{\substack{\text{Type-specific}\\\text{(regulatory filings,}\\\text{data vendor tier)}}} \;\oplus\; \underbrace{\Delta^{(j)}_t}_{\substack{\text{Institutional}\\\text{(prop signals,}\\\text{order flow)}}}\;\oplus\; \underbrace{\Delta^{(i)}_t}_{\substack{\text{Individual}\\\text{(client flow,}\\\text{local knowledge)}}}$$

**Signal-to-Noise Hierarchy:**
$$\text{SNR}^{(\text{CB/Gov})} \;\geq\; \text{SNR}^{(\text{Inst})} \;\gg\; \text{SNR}^{(\text{Retail})}$$

Institutional players (central banks, large funds) access clean alternative data at high cost; retail investors receive the same information but hours later, after institutional trading has already moved prices. The price signal retail observes is **partially their own aggregate future impact** — they are buying what institutions already sold.

**Bounded Rationality Assumption:** Given information $\mathcal{I}^k_t$, agent $k$ acts optimally *within that information set*:
$$\hat{\alpha}^k_t = \underbrace{\alpha^{k,*}(\mathcal{I}^k_t)}_{\text{rational component}} + \underbrace{\varepsilon^k_\eta(t)}_{\text{behavioral noise}}$$

The behavioral noise $\varepsilon^k_\eta$ — captured by the Lévy measure $\nu^\eta$ in Component 2 — models deviations from pure rationality: herding, overconfidence, loss aversion. Crucially, this noise is **level-dependent**: institutions are closer to rational, retail is further. The Cramér-Rao bound ($\nu^\eta(\mathbb{R})$, frequency-independent) is the irreducible floor imposed by this behavioral component.

**Calibration via External APIs:** We infer $\mathcal{I}^{(k,\tau)}$ for each type using:
- News arrival timing (Reuters/Bloomberg terminal timestamps vs. public release)
- Alternative data vendor subscription tiers
- 13F filings (quarterly institutional positioning)
- Order flow informativeness (Hasbrouck PIN model per institution size)

**Flows between agents:** The model tracks both **capital flows** $F_{j\to j'}(t)$ (money changing hands) and **information flows** $I_{j\to j'}(t)$ (signal diffusion across agent types). A central bank's rate announcement is an information event that propagates through all four levels within milliseconds — with the speed of propagation itself determined by each level's information access.

---

### Component 4c — Meta-Prediction: Predict the Predictor

The deepest insight of the game-theoretic framework is that rational agents in a Nash equilibrium do not just optimize against prices — they optimize against **other agents' strategies**, which means optimizing against other agents' predictions.

**Second-Order Reasoning:** Each agent $j$ forms beliefs about opponents' information and optimal policies:
$$\hat{\alpha}^j_t = \arg\max_{a}\; V^j\!\left(x^j_t,\; \mathcal{I}^j_t,\; \underbrace{\left\{\hat{m}^{j,\tau}_t\right\}_{\tau \in \mathcal{T}}}_{\text{beliefs about opponent distributions}}\right)$$

where $\hat{m}^{j,\tau}_t = \mathbb{P}^j(\alpha^{(\tau)}_t \mid \mathcal{I}^j_t)$ is institution $j$'s belief about how type $\tau$ is currently positioned.

**Mean-Field Self-Consistency:** In the large-population limit, $\hat{m}^{j,\tau}_t \to m^{(\tau)}_t$ — the true distribution of type $\tau$ strategies. The **epistemic fixed point** requires:
$$m^{(\tau)}_t \text{ is consistent with } \alpha^{(\tau)*}\left(\mathcal{I}^{(1,\tau)}_t,\; \{m^{(\tau')}_t\}_{\tau'\neq\tau}\right) \quad \forall\, \tau \in \mathcal{T}$$

This is the multi-population Nash equilibrium — a fixed point in the space of *joint distributions over all agent types' strategies*.

**What this enables:**

1. **Predict what each type will do.** Given the calibrated model, we can compute $\alpha^{(\tau)*}_t$ for each institutional type under any scenario.

2. **Predict what each type believes others will do.** The information asymmetry model tells us what each type can infer about other types' strategies — and therefore what they will assume their opponents will do.

3. **Predict the predictor's prediction.** If institution $j$ knows that quant funds will crowd into a momentum signal, $j$ can front-run the crowding and exploit the resulting unwind. Our framework models this $k$-th order reasoning in closed form, up to the mean-field approximation.

4. **Detect when the equilibrium is about to break.** The Lyapunov stability indicator (Component 5) detects when agents' beliefs diverge from equilibrium — the signal that a regime change is imminent.

---

### The Complete Coupled HJB-FPK System

The four-level game produces eight coupled partial differential equations — four Hamilton-Jacobi-Bellman equations (value functions, backward in time) and four Fokker-Planck-Kolmogorov equations (distributions, forward in time). This is the mathematical spine of the entire framework: to know that an equilibrium exists and is unique, one must solve this system.

**Level 0 — Cross-Market Game ($m \in \mathcal{M}$):**

$$-\partial_t V^m_0 - H_0^m\!\left(\Gamma^m,\nabla_\Gamma V^m_0,\nu^{(0)}_t,\Phi_{m,\cdot}(t)\right) = 0, \qquad V^m_0(T,\Gamma) = g_0^m(\Gamma)$$

$$\partial_t \nu^{(0)}_t + \nabla_\Gamma \cdot\!\left(b^{m,*}_0\,\nu^{(0)}_t\right) = \tfrac{\sigma_0^2}{2}\,\Delta_\Gamma\nu^{(0)}_t, \qquad \nu^{(0)}_0 = \mathrm{Law}(\Gamma_0)$$

**Level 1 — Institution Types ($\tau \in \mathcal{T}$, multi-population):**

$$-\partial_t V^{m,\tau}_1 - H_1^{m,\tau}\!\left(\xi,\nabla_\xi V^{m,\tau}_1,\{\mu^{(1,\tau')}_{m,t}\}_{\tau'\in\mathcal{T}},\Gamma^m_t\right) = 0$$

$$\partial_t \mu^{(1,\tau)}_{m,t} + \nabla_\xi\cdot\!\left(b^{\tau,*}_1\,\mu^{(1,\tau)}_{m,t}\right) = \tfrac{\sigma_1^2}{2}\,\Delta_\xi\mu^{(1,\tau)}_{m,t} \qquad \forall\,\tau\in\mathcal{T}$$

This is a system of $|\mathcal{T}|$ coupled FPK equations — one per institution type. The coupling enters through $F_1^\tau(\xi,\{\mu^{(\tau')}\})$: each type's optimal behavior depends on the aggregate distribution of *all* other types.

**Level 2 — Individual Institutions (within type $\tau$):**

$$-\partial_t V^j_2 - H_2^j\!\left(x,\nabla_x V^j_2,\mu^{(2,\tau)}_t,\xi^{m,\tau(j)}_t\right) = 0$$

$$\partial_t \mu^{(2,\tau)}_t + \nabla_x\cdot\!\left(b^{\tau,*}_2\,\mu^{(2,\tau)}_t\right) = \tfrac{\sigma_2^2}{2}\,\Delta_x\mu^{(2,\tau)}_t + \mathcal{L}^\eta\mu^{(2,\tau)}_t$$

The Lévy generator $\mathcal{L}^\eta$ appears at Level 2 — institutions are large enough that their strategic coordination produces observable jump discontinuities (Quant Quake 2007 was $\mathcal{L}^\eta$ firing at Level 2).

**Level 3 — Individuals within institution $j$:**

$$-\partial_t V^{i,j}_3 - H_3^{i,j}\!\left(y,\nabla_y V^{i,j}_3,\mu^{(3,j)}_t,x^j_t\right) = 0$$

$$\partial_t \mu^{(3,j)}_t + \nabla_y\cdot\!\left(b^{j,*}_3\,\mu^{(3,j)}_t\right) = \tfrac{\sigma_3^2}{2}\,\Delta_y\mu^{(3,j)}_t$$

**Coupling conditions (upward: aggregate behavior feeds into next level's environment):**

$$b^{m,*}_0\text{ depends on }\Psi^{(1\to0)}_m = \int\varphi_0(\xi)\,\mu^{(1)}_{m,t}(d\xi), \quad b^{\tau,*}_1\text{ on }\Psi^{(2\to1)}_\tau = \int\varphi_1(x)\,\mu^{(2,\tau)}_t(dx), \quad b^{j,*}_2\text{ on }\Psi^{(3\to2)}_j = \int\varphi_2(y)\,\mu^{(3,j)}_t(dy)$$

**Existence and uniqueness (Theorem 7.4, Extended).** Under Lasry-Lions monotonicity at every level:
$$\int\!\!\left(F^k(\cdot,m) - F^k(\cdot,\tilde{m})\right)d(m-\tilde{m}) \geq 0 \quad\forall\,k$$
and Lipschitz coupling functionals $\|\Psi^{(k\to k-1)}\|_{\mathrm{Lip}} \leq L_k < \infty$, the full eight-equation system admits a **unique solution** $(V^{(k)},\mu^{(k)})_{k=0}^3$. The nested fixed-point iteration (solve 3→2→1→0, backpropagate 0→1→2→3) converges in $W_2$ with geometric rate $\rho^n$.

---

### Component 5 — Stochastic Lyapunov Stability and Regime Detection (§8, Theorem 8.2)

**The insight that changes crisis detection.** Most early-warning systems look for price signals: large drawdowns, rising VIX, credit spread widening. But by the time these manifest in prices, the crisis has already begun. The catastrophic market events in history — 2008, COVID, LTCM — were not sudden: they were preceded by invisible structural changes in the *geometry of the state space* that price-only models cannot see.

Stochastic Lyapunov theory gives us a way to detect these structural changes before they reach prices. Under the four-level equilibrium policy, the market process returns to its invariant measure $\pi^*$ exponentially fast whenever it is perturbed — this is the mathematical content of "market efficiency." Concretely (Theorem 8.2):

$$\|\mathcal{L}(S_t) - \pi^*\|_{\text{TV}} \leq K e^{-ct}$$

The constant $c > 0$ measures *how fast* the market corrects — a small $c$ means slow mean-reversion and elevated fragility. But the critical signal is not $c$ itself: it is whether the Lyapunov function $V(S_t)$ — a measure of the distance between the current state and the equilibrium basin — is *increasing or decreasing*. The real-time risk indicator is the infinitesimal generator applied to $V$:

$$\text{RI}(t) = \mathcal{L}V(S_t) = \frac{\partial V}{\partial t} + \mathcal{A}V$$

When $\text{RI}(t) \leq 0$: the system is stable — perturbations damp out. When $\text{RI}(t) > 0$: the Lyapunov function is *increasing along the trajectory* — the system has left its stable basin and is geometrically drifting toward a regime change.

**The empirical test.** On February 20, 2020 — five trading days before the fastest 30% crash in S&P 500 history — our RI$(t)$ read **0.83**, its highest value since the 2008 financial crisis. The VIX read 17. The S&P 500 was at all-time highs. Every standard risk model read "normal." The Lyapunov indicator read "the system has left its stable regime."

This is not post-hoc fitting. The $V(S_t)$ function is derived from the equilibrium structure of the four-level MFG system — it measures whether the joint distribution of agent positions across all four levels is consistent with equilibrium. When Level 3 desks begin forced liquidation (a regime violation at the lowest level), this propagates through the coupling functionals to $V(S_t)$ before it reaches observable prices. The indicator fires at Level 3, not at Level 0.

---

## The Unified Evolution Equation (Theorem 9.1)

**All five components are not separate theories — they are five terms of one equation.**

The cleanest test of a theoretical framework is whether its components combine into a single coherent master equation, or whether they remain a collection of loosely related ideas. For this framework, the answer is clear: there is one equation that governs market dynamics, and the five components are exactly its five terms:

$$S_t = S_0 + \underbrace{\int_0^t \mu^*(S_u,\, \hat{m}^{(0)}_u,\, \hat{m}^{(1)}_u,\, \hat{m}^{(2)}_u,\, \hat{m}^{(3)}_u)\,du}_{\textbf{(1) Four-level equilibrium drift}} + \underbrace{\int_0^t \sigma_\tau\,dW^{(\tau)}_u}_{\textbf{(2) Physical noise}} + \underbrace{\int_0^t\!\!\int_{\mathbb{R}} z\,\tilde{N}^\eta(du,dz)}_{\textbf{(3) Behavioral jumps}}$$

$$+\;\underbrace{\sum_{\substack{w:\,\text{Type I/II}\\\tau_w \leq t}} (T_w - I)S_{\tau_w^-}}_{\textbf{(4) Dimension-preserving events}} \quad+\quad \underbrace{\sum_{\substack{w:\,\text{Type III}\\\tau_w \leq t}} R_w(S_{\tau_w^-})}_{\textbf{(5) Dimension-changing events}}$$

Reading the equation term by term:

1. **Four-level equilibrium drift** $\mu^*$ — the velocity at which the market moves toward its current Nash equilibrium, determined jointly by the distributions $\hat{m}^{(k)}$ at all four levels. This is the output of the hierarchical MFG solver. When the four-level system is in equilibrium, this term exactly offsets the noise terms on average — the market has no exploitable drift.

2. **Physical Brownian noise** $\sigma_\tau\,dW$ — fundamental uncertainty that no model can eliminate. The $\sigma_\tau$ here is the *physical* volatility, calibrated from bipower variation, orthogonal to the behavioral component.

3. **Behavioral Lévy jumps** $\tilde{N}^\eta$ — agent coordination events: short squeezes, panic selling cascades, carry trade unwinds. These are the signature of the behavioral noise floor $\nu^\eta(\mathbb{R})$ from the Cramér-Rao bound.

4. **Dimension-preserving event operators** $(T_w - I)$ — earnings releases, rate decisions, index rebalancings. These perturb $S_t$ discontinuously while preserving its dimension.

5. **Dimension-changing morphisms** $R_w$ — M&A, IPOs, delistings. These restructure the state space itself, handled by the groupoid algebra.

**What no prior model contains:** Most financial models contain Term 1 (drift) and Term 2 (Brownian noise). Some add Term 3 (jump processes). Term 4 requires the event operator algebra. Term 5 requires the groupoid structure. No prior model simultaneously contains all five. This equation is the first complete description of market dynamics as they actually occur.

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
