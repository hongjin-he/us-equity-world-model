# Research Notebooks — E-Game-C Theory Walkthrough

17-day first-principles series building up the complete mathematical framework,
from why factor models fail to the full E-Game-C world model architecture with
complete matrix-formalized event algebra.

| Day | Notebook | Key Concept | Theory Section |
|-----|----------|-------------|----------------|
| 01 | [Why Factor Models Fail](day01_why_factor_models_fail.ipynb) | Lucas Critique → alpha decay | §I Motivation |
| 02 | [Roughness of Noise → Dual τ/η](day02_from_brownian_to_rough.ipynb) | BPV decomposition, Cramér-Rao bound | §II Noise |
| 03 | [Dual Noise Decomposition](day03_dual_noise.ipynb) | Physical noise τ + behavioral η | §III SDE |
| 04 | [Encoder E — Transformer VAE](day04_encoder_e_transformer_vae.ipynb) | Three-term ELBO, latent Markov property | §V Encoder |
| 05 | [Markets as Mean-Field Games](day05_markets_as_mean_field_games.ipynb) | HJB-Fokker-Planck system | §V MFG |
| 06 | [MFC/MFG Hierarchy](day06_mfc_hierarchy_nations_firms_traders.ipynb) | Stackelberg: L1 (MFC) over L2/L3 (MFG) | §VI Hierarchy |
| 07 | [Events as Groupoid Operators](day07_event_operators_groupoid_algebra.ipynb) | Mode I/II/III, partial composition | §IV Events |
| 08 | [Lyapunov Stability & Crisis Detector](day08_lyapunov_stability_crisis_detector.ipynb) | LV ≤ -cV+b → Λ_t crisis indicator | §VII Stability |
| 09 | [Reflexivity — Formalizing Soros](day09_reflexivity_soros_formalized.ipynb) | MFG fixed point = Soros equilibrium | §V MFG |
| 10 | [Avatar Analogy — Agent Taxonomy](day10_avatar_analogy_agent_types.ipynb) | 6 agent classes, crowding aversion | §VI Agents |
| 11 | [Optimal Control → Portfolio Weights](day11_optimal_control_hjb_portfolio.ipynb) | HJB → Merton + MFG drift adjustment | §VIII Control |
| 12 | [Seven Theorems Unified](day12_seven_theorems.ipynb) | Complete mathematical guarantee | §VIII Theorems |
| 13 | [From Finance to AGI](day13_from_finance_to_agi.ipynb) | E-Game-C as general world model | §IX Outlook |
| 14 | [Reflection & Roadmap](day14_reflection_and_roadmap.ipynb) | What we built, what comes next | — |
| 15 | [Level 0 — Cross-Market Capital Flows](day15_level0_cross_market_capital_flows.ipynb) | Four-level hierarchy, DXY cycle, L0→L1 transmission | §VI L0 |
| 16 | [Predict the Predictor — Retail AI](day16_predict_the_predictor_retail_ai.ipynb) | AI adoption convergence, μ^retail signal, Cramér-Rao gap | §IX AI Era |
| 17 | [Complete Event Algebra — Matrix Theory](day17_event_algebra_complete_matrix_theory.ipynb) | All 22 operators, groupoid composition, event timeline | §IV Complete |

## Run locally

```bash
pip install jupyter numpy scipy matplotlib
jupyter lab notebooks/
```

## Series structure

Each notebook is self-contained but builds on previous ones:

1. **Days 1–3**: Why existing approaches fail and what the right noise model is
2. **Days 4–6**: The E (Encoder) and G (Game) modules
3. **Days 7–9**: Advanced structures: events, stability, reflexivity
4. **Days 10–11**: Agents, control, and portfolio construction
5. **Days 12–14**: Synthesis, generalization, and outlook
6. **Days 15–17**: Extensions: L0 global hierarchy, AI-era dynamics, complete event algebra

## New in Days 15–17

**Day 15** adds **Level 0** to the hierarchy — nations, central banks, and sovereign wealth
funds. The dollar cycle (DXY) is shown to explain cross-market correlation not captured
by domestic-only models. Out-of-sample R² increases with L0 signals.

**Day 16** formalizes **"Predict the Predictor"**: as retail AI adoption grows, retail
strategies homogenize and become observable to institutional players. The aggregate
retail strategy distribution μ^retail_t becomes an alpha source with bounded but
non-zero edge (Cramér-Rao lower bound on prediction uncertainty).

**Day 17** gives the **complete matrix formalization** of all 22 financial event operators:
- **10 Mode I** (local endomorphisms): stock split, reverse split, dividend, secondary
  offering, buyback, earnings shock, analyst rating, index change, trading halt, short squeeze
- **7 Mode II** (global tensor product): rate change, QE, QT, systemic crisis,
  circuit breaker, volatility regime shift, inflation shock
- **5 Mode III** (pairwise morphisms): merger, spin-off, IPO, delisting, bankruptcy

The groupoid composition rule `A_comp = A1 @ A2`, `b_comp = A1 @ b2 + b1`,
`Σ_comp = chol(Σ1² + A1 Σ2² A1ᵀ)` is verified numerically.
