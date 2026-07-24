# Research Notebooks — E-Game-C Theory Walkthrough

14-day first-principles series building up the complete mathematical framework, from why
factor models fail to the full E-Game-C world model architecture.

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

## Run locally

```bash
pip install jupyter numpy scipy matplotlib
jupyter lab notebooks/
```

## Series structure

Each notebook is self-contained but builds on previous ones. The series follows the
logical order of the mathematical framework:

1. **Days 1-3**: Why existing approaches fail and what the right noise model is
2. **Days 4-6**: The E (Encoder) and G (Game) modules
3. **Days 7-9**: Advanced structures: events, stability, reflexivity
4. **Days 10-11**: Agents, control, and portfolio construction
5. **Days 12-14**: Synthesis, generalization, and outlook
