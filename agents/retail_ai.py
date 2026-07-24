"""
Retail AI behavior modeling — "Predict the Predictor" (Day 16, §IX).

Key insight: As retail investors adopt AI-assisted investment tools, their
decision processes become more legible — and therefore more predictable —
to institutional players. This module models:

1. The distribution of retail AI queries (what they ask their AI assistant)
2. The mapping from query → recommended action (LLM response stub)
3. The aggregate retail strategy distribution μ^retail_t
4. How this distribution creates exploitable signals

Mathematical formulation:
  - Let ρ_t(q) = density of retail query q at time t (Gaussian mixture)
  - LLM response: R: query → action ∈ ΔΑ (simplex over asset allocations)
  - Aggregate strategy: μ^retail_t = ∫ R(q) ρ_t(q) dq
  - Institutional signal: α_t = f(μ^retail_t) — exploits predictable retail flow

Cramér-Rao lower bound for signal uncertainty:
  Var(μ̂^retail) ≥ σ_τ²/Δt + ν_η(ℝ)  (§III irreducible noise floor)

Reference: Alpha Flow §IX; Soros (1987) reflexivity for μ^retail feedback loop.

IMPORTANT: All LLM API calls are stubs.
  Replace os.getenv("OPENAI_KEY", "[YOUR_KEY_HERE]") with a real key to activate.
"""
from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class InvestorType(Enum):
    """Retail investor archetype (determines query distribution shape)."""
    PASSIVE_INDEX   = "passive_index"    # just asks "should I rebalance?"
    ACTIVE_FOLLOWER = "active_follower"  # chases momentum, asks "what's hot?"
    NEWS_REACTOR    = "news_reactor"     # asks about specific headlines
    DIY_QUANT       = "diy_quant"        # prompts for backtests, factor exposure
    MEME_TRADER     = "meme_trader"      # social-media driven, asks about short interest


@dataclass
class RetailQuery:
    """A single retail investor's AI query at time t."""
    investor_type: InvestorType
    tickers: list[str]
    context: str      # natural language context (earnings, news, etc.)
    risk_tolerance: float  # 0=very conservative, 1=very aggressive


def simulate_retail_query_distribution(
    n_investors: int,
    tickers: list[str],
    market_stress: float = 0.0,
    ai_adoption_rate: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> list[RetailQuery]:
    """
    Sample a population of retail AI queries at time t.

    Parameters
    ----------
    n_investors : total retail investors (subset uses AI)
    tickers : asset universe
    market_stress : 0=calm, 1=crisis (shifts query mix toward news_reactor)
    ai_adoption_rate : fraction of retail investors using AI assistants
    rng : random state

    Returns
    -------
    List of RetailQuery from AI-using investors
    """
    rng = rng or np.random.default_rng()

    # Only AI-adopters generate structured queries
    n_ai_users = int(n_investors * ai_adoption_rate)

    # Query type distribution: stress drives news-reactor spikes
    base_mix = np.array([0.30, 0.25, 0.20, 0.15, 0.10])  # type prevalence
    stress_shift = market_stress * np.array([-0.10, -0.05, +0.25, -0.05, -0.05])
    type_probs = np.clip(base_mix + stress_shift, 0.0, 1.0)
    type_probs /= type_probs.sum()

    investor_types = list(InvestorType)
    queries: list[RetailQuery] = []
    for _ in range(n_ai_users):
        inv_type = investor_types[rng.choice(len(investor_types), p=type_probs)]
        n_tickers = rng.integers(1, min(5, len(tickers)) + 1)
        chosen = rng.choice(tickers, size=n_tickers, replace=False).tolist()
        risk_tol = float(np.clip(rng.beta(2, 3) + 0.1 * market_stress, 0, 1))
        contexts = [
            "Should I buy the dip?",
            "What does this earnings beat mean for my portfolio?",
            "Is now a good time to rebalance into bonds?",
            "What stocks are trending today?",
            "Analyze the short interest in this ticker.",
        ]
        context = contexts[investor_types.index(inv_type)]
        queries.append(RetailQuery(inv_type, chosen, context, risk_tol))

    return queries


def stub_llm_response(
    query: RetailQuery,
    tickers: list[str],
    api_key: str | None = None,
) -> np.ndarray:
    """
    Convert a retail query to an asset allocation vector.

    This is a STUB — it simulates LLM behavior without real API calls.
    To use a real LLM:
      1. Set OPENAI_KEY environment variable (or pass api_key)
      2. Uncomment the openai section below

    Parameters
    ----------
    query : RetailQuery
    tickers : full asset universe (determines output dimension)
    api_key : optional override (default: os.getenv("OPENAI_KEY", "[YOUR_KEY_HERE]"))

    Returns
    -------
    alloc : (len(tickers),) allocation vector, sums to 1
    """
    # _key = api_key or os.getenv("OPENAI_KEY", "[YOUR_KEY_HERE]")
    # Uncommenting and setting key above would enable real LLM calls:
    #
    # import openai
    # client = openai.OpenAI(api_key=_key)
    # prompt = f"Given a {query.investor_type.value} investor with tickers "
    #          f"{query.tickers} and question '{query.context}', "
    #          f"return a JSON allocation over {tickers}."
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # ... parse JSON response ...

    n = len(tickers)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    alloc = np.zeros(n)

    rng = np.random.default_rng(abs(hash(query.context)) % (2**31))

    if query.investor_type == InvestorType.PASSIVE_INDEX:
        # Equal-weight (rational passive)
        alloc[:] = 1.0 / n

    elif query.investor_type == InvestorType.ACTIVE_FOLLOWER:
        # Concentrate in mentioned tickers (momentum chasing)
        for t in query.tickers:
            if t in ticker_to_idx:
                alloc[ticker_to_idx[t]] = 1.0 / len(query.tickers)
        if alloc.sum() == 0:
            alloc[:] = 1.0 / n

    elif query.investor_type == InvestorType.NEWS_REACTOR:
        # Overweight mentioned tickers, underweight everything else
        focus_weight = 0.6
        for t in query.tickers:
            if t in ticker_to_idx:
                alloc[ticker_to_idx[t]] = focus_weight / len(query.tickers)
        residual = 1.0 - alloc.sum()
        alloc += residual / n  # spread remainder

    elif query.investor_type == InvestorType.DIY_QUANT:
        # Momentum-tilted allocation (simulated signal)
        raw = rng.standard_normal(n)
        raw -= raw.mean()
        alloc = 1.0 / n + 0.1 * raw / (np.abs(raw).sum() + 1e-9)
        alloc = np.clip(alloc, 0, 1)
        if alloc.sum() > 0:
            alloc /= alloc.sum()

    elif query.investor_type == InvestorType.MEME_TRADER:
        # All-in on single ticker with highest social signal (random here)
        hot_idx = rng.integers(n)
        for t in query.tickers:
            if t in ticker_to_idx:
                hot_idx = ticker_to_idx[t]
                break
        alloc[hot_idx] = 1.0

    if alloc.sum() < 1e-9:
        alloc[:] = 1.0 / n

    return alloc / alloc.sum()


def aggregate_retail_strategy(
    queries: list[RetailQuery],
    tickers: list[str],
) -> np.ndarray:
    """
    Compute μ^retail_t = average allocation across all retail AI users.

    This is the mean-field distribution that institutional players observe
    and can exploit: if μ^retail_t overweights a sector, institutions can
    anticipate the flow and front-run or fade it.

    Parameters
    ----------
    queries : list of retail queries (from simulate_retail_query_distribution)
    tickers : full asset universe

    Returns
    -------
    mu_retail : (n,) average allocation vector
    """
    n = len(tickers)
    if not queries:
        return np.ones(n) / n

    allocations = np.stack([stub_llm_response(q, tickers) for q in queries])
    return allocations.mean(axis=0)


class RetailAIAgent:
    """
    Model of the aggregate retail AI behavior — a single "super-agent"
    whose allocation μ^retail_t is the mean field of retail decisions.

    Institutional players observe (a noisy version of) μ^retail_t and use it
    as a signal. This is the "predict the predictor" alpha source.

    The gap between retail and institutional prediction quality is irreducible:
      σ_retail^2 ≥ σ_institutional^2 + Δ_gap

    where Δ_gap comes from data cleaning latency, execution speed, and
    capital scale constraints (the "AI advantage moat").
    """

    def __init__(
        self,
        tickers: list[str],
        ai_adoption_rate: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ):
        self.tickers = tickers
        self.ai_adoption_rate = ai_adoption_rate
        self.rng = rng or np.random.default_rng()
        self._mu_retail: Optional[np.ndarray] = None

    def step(
        self,
        n_investors: int = 10_000,
        market_stress: float = 0.0,
    ) -> np.ndarray:
        """
        Simulate one period's retail AI behavior and return μ^retail_t.

        Parameters
        ----------
        n_investors : total retail investor count
        market_stress : [0, 1] market stress level

        Returns
        -------
        mu_retail : (n,) mean allocation
        """
        queries = simulate_retail_query_distribution(
            n_investors=n_investors,
            tickers=self.tickers,
            market_stress=market_stress,
            ai_adoption_rate=self.ai_adoption_rate,
            rng=self.rng,
        )
        self._mu_retail = aggregate_retail_strategy(queries, self.tickers)
        return self._mu_retail

    def institutional_signal(
        self,
        mu_retail: Optional[np.ndarray] = None,
        noise_level: float = 0.01,
    ) -> np.ndarray:
        """
        Construct institutional alpha from observed retail flow.

        Institutional players observe μ^retail with noise (limited visibility
        into retail order flow). They fade extremes: heavily overweighted
        assets by retail → short signal; underweighted → long signal.

        α_inst = -sign(μ^retail - 1/n) × |μ^retail - 1/n|^{0.5}
               + ε,  ε ~ N(0, noise_level²·I)
        """
        mu = mu_retail if mu_retail is not None else self._mu_retail
        if mu is None:
            raise ValueError("Call .step() first or provide mu_retail")

        n = len(self.tickers)
        deviation = mu - 1.0 / n
        raw_signal = -np.sign(deviation) * np.sqrt(np.abs(deviation))

        # Irreducible noise from Cramér-Rao lower bound
        noise = self.rng.normal(0, noise_level, size=n)
        signal = raw_signal + noise

        # Cross-sectional normalize
        signal -= signal.mean()
        if np.abs(signal).max() > 1e-9:
            signal /= np.abs(signal).max()
        return signal

    def adoption_convergence_path(
        self,
        T: int = 20,
        adoption_start: float = 0.05,
        adoption_end: float = 0.80,
        n_assets: int = 10,
        n_investors: int = 5000,
        n_seeds: int = 8,
    ) -> dict:
        """
        Simulate how signal quality and crowding change as AI adoption grows.

        The key mechanism is HOMOGENIZATION, not aggregation: retail users
        querying the same handful of LLM platforms receive the *same*
        recommendation c_t (the platform consensus). The effective retail
        allocation is a mixture

            μ_retail = (1 − h(a)) · μ_idiosyncratic + h(a) · c_t

        where h(a) = platform concentration, increasing in the adoption
        rate a (few platforms serve everyone → answers correlate). As a → 1:
          - μ_retail collapses onto c_t → institutional fade signal strengthens
          - ‖μ_retail − uniform‖ grows → synchronized-panic fragility grows

        This is the Day 16 thesis in quantitative form: individual AI use is
        rational; universal AI use is legible — and therefore exploitable.

        Returns
        -------
        dict with keys: adoption_rates, signal_strengths, crowding_risks
        """
        tickers_temp = [f"A{i}" for i in range(n_assets)]
        adoption_rates = np.linspace(adoption_start, adoption_end, T)
        signal_strengths = np.zeros(T)
        crowding_risks = np.zeros(T)
        uniform = 1.0 / n_assets

        for t, rate in enumerate(adoption_rates):
            sig_acc, crowd_acc = 0.0, 0.0
            for s in range(n_seeds):
                seed_rng = np.random.default_rng(10_000 * s + t)
                agent = RetailAIAgent(tickers_temp, ai_adoption_rate=rate, rng=seed_rng)
                mu_idio = agent.step(n_investors=n_investors)

                # Platform consensus c_t: the answer "everyone" receives today —
                # concentrated in a few trending names
                trend = seed_rng.dirichlet(np.full(n_assets, 0.25))
                h = 0.10 + 0.80 * rate          # platform concentration
                mu = (1.0 - h) * mu_idio + h * trend

                dev = mu - uniform
                # Unnormalized fade-signal strength: exploitable deviation
                sig_acc += np.sqrt(np.abs(dev)).mean()
                crowd_acc += np.sqrt(np.sum(dev ** 2))
            signal_strengths[t] = sig_acc / n_seeds
            crowding_risks[t] = crowd_acc / n_seeds

        return {
            "adoption_rates": adoption_rates,
            "signal_strengths": signal_strengths,
            "crowding_risks": crowding_risks,
        }
