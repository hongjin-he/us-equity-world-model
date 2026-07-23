# US Equity World Model

A multi-agent world model of US equity markets, treating price discovery as an emergent outcome of heterogeneous agents competing under information asymmetry — analogous to Stanford's Generative Agents applied to quantitative finance.

## Core Idea

Financial markets are fundamentally a **multi-player game** with:
- Asymmetric information across participants
- Heterogeneous beliefs, strategies, and capital constraints
- Observable (but incomplete) public state

Instead of fitting a price model directly, we simulate the **agents that generate prices**.

## Architecture

```
us-equity-world-model/
├── agents/          # Market participant archetypes
│   ├── quant_fund.py        # Quant funds (inferred strategy style)
│   ├── corporate.py         # Corporations (treasury, buybacks, M&A)
│   ├── institutional.py     # Passive/active institutional investors
│   └── retail.py            # Retail flow (aggregate, sentiment-driven)
│
├── state/           # State space definitions
│   ├── portfolio.py         # Holdings, notional, leverage
│   ├── liquidity.py         # Cash, credit lines, margin
│   ├── information.py       # Information set per agent (what they observe)
│   └── market.py            # Global market state (OHLCV, order book)
│
├── world/           # World model core
│   ├── engine.py            # Main simulation loop
│   ├── matching.py          # Order matching / price formation
│   └── clock.py             # Event-driven time (market open/close, earnings, macro)
│
├── data/            # Public data ingestion
│   ├── scrapers/
│   │   ├── sec_13f.py       # 13F filings → institutional holdings
│   │   ├── sec_edgar.py     # Insider ownership, 8-K, 10-K
│   │   └── team_intel.py    # Public bios → quant fund strategy inference
│   ├── raw/
│   └── processed/
│
├── inference/       # Strategy style estimation
│   ├── quant_profiler.py    # Infer factor style from hiring patterns & public signals
│   ├── factor_zoo.py        # Factor library (momentum, value, low-vol, alt-data, ML)
│   └── ownership_graph.py   # Build corporate ownership graph from public filings
│
├── sim/             # Simulation entry points
│   ├── run.py               # Launch full simulation
│   └── backtest.py          # Replay historical episodes
│
├── configs/         # Scenario configs (YAML)
│   └── baseline_sp500.yaml
│
└── tests/
```

## Agent Taxonomy

| Agent Type | Observables Used | Strategy Inferred From |
|---|---|---|
| Quant Fund (ML-driven) | 13F holdings, factor exposure | Team hires (ML/DL engineers) |
| Quant Fund (Stat Arb) | Pair correlations, mean-reversion | Academics from stat/math depts |
| Long-Only Institutional | 13F, proxy votes | AUM, turnover ratio |
| Corporate (Buyback/M&A) | 10-K cash flow, board approvals | 8-K filings |
| Retail Aggregate | Options flow, Reddit sentiment | Public sentiment proxies |

## Key Design Principles

1. **State-space first** — every agent has an explicit `(portfolio, liquidity, information_set)` state before any action logic
2. **Calibrate to observables** — agent parameters are constrained by public 13F / SEC data, not free parameters
3. **Information asymmetry is explicit** — each agent has a distinct `InformationSet`; no god's-eye view
4. **Emergent prices** — prices arise from order matching, not a price process assumption

## References

- Generative Agents (Park et al., 2023) — agent architecture inspiration
- SEC EDGAR / 13F filings — primary data source for institutional state
- Factor Zoo literature — for quant fund strategy style inference

## Status

`[framework]` — scaffolding phase. No external APIs or paid data required.
