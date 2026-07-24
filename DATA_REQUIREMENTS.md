# Data Requirements & Research Roadmap

> **A world model is an organism. This document is its diet.**
>
> MicroWorld's mathematics is complete and its code runs end-to-end on synthetic data
> today (`python demo/global_demo.py` — zero API keys). What separates the current
> Type 1 prototype from a validated research instrument is **data**: hundreds of
> fragmented streams, each feeding a specific term of a specific equation. This
> document enumerates every one of them — what we need, why the equation needs it,
> where to get it, what it costs, and (for data that does not exist anywhere)
> the exact experiment that would create it.

![Data status map](figures/fig_data_status_map.png)

**Security policy of this repo:** every external connection is a stub
(`os.getenv("X_KEY", "[YOUR_KEY_HERE]")`). No key is committed, no paid call is made.
Adding a key activates the loader; nothing else changes.

---

## Legend

| Symbol | Meaning |
|---|---|
| 🔌 | Loader stub already wired in this repo — add an API key and it runs |
| 🆓 | Free source identified — loader still to be written |
| 🎓 | Available through a university WRDS subscription (HKUST ✅ / Stanford ✅) |
| 💰 | Commercial license or partnership required |
| 🧪 | **The data does not exist anywhere — a designed experiment must create it** |
| **P0** | Required for the NeurIPS-workshop MVP experiments |
| **P1** | Required for the full paper |
| **P2** | Required for the industrial-grade product |

---

## 1 · What each model component eats

### 1.1 Dual noise calibration — σ_τ, ν_η (§II–III, `state/noise.py`)

The bipower-variation estimator `BPV = (π/2)Σ|rᵢ||rᵢ₋₁|` and the Lee-Mykland jump
test are *intraday* statistics. Daily bars cannot separate physical from behavioral
noise — this component is the single strongest argument for institutional data access.

| Quantity | Data needed | Granularity / history | Sources | Status | Priority |
|---|---|---|---|---|---|
| σ_τ (physical vol) | Intraday bars, all US equities | 1–5 min, ≥ 10 years | 🎓 WRDS TAQ (millisecond) · 🔌 Polygon.io ([polygon.py:8](data/sources/polygon.py)) · 🆓 Alpaca (IEX feed) · 💰 Databento | stub wired | **P0** |
| ν_η (jump measure) | Same bars + event timestamps to label jump causes | 1 min + news timestamps | Same + 🆓 GDELT event times · 💰 RavenPack | stub wired | **P0** |
| Microstructure floor | Quote data (bid-ask), trade signs | tick level | 🎓 WRDS TAQ quotes · 💰 Databento MBO | not wired | P1 |
| Overnight vs intraday split | Session-tagged bars | 1 min | Same as σ_τ | not wired | P1 |

**Experiment E1 (P0).** *Noise atlas*: run the calibrator over 500 liquid US equities
× 10 years of 5-min bars. Deliverables: cross-sectional distribution of (σ_τ, λ_η, ν_η);
validation = jump-detection false-discovery rate against a curated macro/earnings
event list. This directly tests Theorem 1's decomposition on real data.

### 1.2 Event operator calibration — (A_w, b_w, Σ_w) for all 22 operators (§IV–V, `events/operators.py`)

Every operator in the catalogue currently carries *literature-informed placeholder*
coefficients (e.g. index inclusion ≈ +3.5%). Each must be re-estimated by an event
study on a labeled corpus of historical events.

| Event class (operator) | Data needed | Sources | Status | Priority |
|---|---|---|---|---|
| Splits, reverse splits, dividends | Corporate actions master file | 🎓 CRSP events · 🔌 Polygon corporate actions · 🆓 EODHD sample | stub | **P0** |
| Secondary offerings / buybacks | Issuance + repurchase filings | 🎓 SDC Global New Issues · 🆓 EDGAR S-3/8-K full-text | 🆓 | P1 |
| Earnings shocks | Actual vs consensus EPS | 🎓 I/B/E/S · 🆓 EDGAR + 🔌 AlphaVantage/FMP | 🆓 | **P0** |
| Analyst ratings | Recommendation changes w/ timestamps | 🎓 I/B/E/S detail · 💰 Benzinga API | 🎓 | P1 |
| Index add/delete | S&P + Russell reconstitution lists | 🆓 press releases (scrapable) · 💰 FTSE/S&P direct | 🆓 | P1 |
| Trading halts | Halt/resume timestamps | 🆓 NASDAQ Trader halt feed (real-time + history) | 🆓 | P1 |
| Short squeezes | Short interest + borrow fee + intraday | 🆓 FINRA SI (bi-monthly) · 💰 Markit/Ortex borrow · 🆓 iBorrowDesk scrape | 🆓 | P1 |
| Rate decisions (Mode II) | FOMC calendar, statements, surprise vs futures | 🆓 Fed calendar · 🆓 CME FedWatch · 🔌 FRED ([fred.py:7](data/sources/fred.py)) | stub | **P0** |
| QE/QT (Mode II) | CB balance sheets | 🆓 Fed H.4.1 weekly · 🆓 ECB/BoJ/PBoC monthly | 🆓 | **P0** |
| CPI/macro prints (Mode II) | Release calendar + consensus | 🔌 FRED (actual) · 💰 Bloomberg consensus · 🆓 Investing.com scrape | stub | **P0** |
| M&A (Mode III merger) | Deal terms: acquirer, target, premium, mix | 🎓 SDC Platinum M&A · 💰 Zephyr · 🆓 EDGAR DEFM14A | 🎓 | **P0** |
| IPOs (Mode III) | Offer price, first-day trades, lockups | 🆓 **Jay Ritter's IPO datasets** (gold standard, free) · 🆓 NASDAQ calendar | 🆓 | P1 |
| Spin-offs, delistings, bankruptcies | CRSP delist codes + UCLA-LoPucki BRD | 🎓 CRSP delist file · 🆓 LoPucki bankruptcy DB | 🎓 | P1 |

**Experiment E2 (P0).** *Operator estimation pipeline*: for each event class, stack all
historical instances, align in event time, and fit (A_w, b_w, Σ_w) by regularized
regression of post-event state on pre-event state. Validation = out-of-sample state
prediction error vs (i) ignore-the-event baseline, (ii) GARCH-with-dummies.
The 22-operator catalogue becomes 22 *empirical tables* — to our knowledge the first
complete matrix-valued event-study atlas.

### 1.3 Encoder E — latent state z (§V, `encoder/`)

| Quantity | Data needed | Sources | Status | Priority |
|---|---|---|---|---|
| Price/volume panel | EOD OHLCV, survivorship-free | 🎓 CRSP · 🔌 Polygon · 🆓 Stooq/Yahoo (biased) | stub | **P0** |
| Leverage ℓ_t | Quarterly D/E per firm | 🎓 Compustat · 🆓 EDGAR XBRL (free, parseable) | 🆓 | **P0** |
| Shares outstanding κ_t | Split-adjusted share counts | 🎓 CRSP · 🆓 EDGAR DEF 14A | 🎓 | **P0** |
| Disclosure ι_t | 8-K counts, filing cadence, coverage breadth | 🆓 EDGAR full-text search API · 💰 RavenPack coverage stats | 🆓 | P1 |
| Macro conditioning | Rates, CPI, unemployment, PMI, yield curve | 🔌 FRED ([fred.py:7](data/sources/fred.py)) — free key | stub | **P0** |
| News embeddings | Headline stream w/ timestamps | 🔌 NewsAPI ([news.py:7](data/sources/news.py)) · 🆓 GDELT 2.0 (free, 15-min) · 💰 RavenPack | stub | P1 |

### 1.4 Game module G — MFG at L1/L2 (§VI–VII, `game/`)

The mean field μ_t is *observable* — this is finance's advantage over physics.
Positioning data is the empirical μ.

| Quantity | Data needed | Sources | Status | Priority |
|---|---|---|---|---|
| Institutional μ^(1) by type | Quarterly holdings by manager type | 🆓 EDGAR 13F raw · 🎓 Thomson-Reuters s34 (cleaned) · 💰 WhaleWisdom ([sec_13f.py](data/scrapers/sec_13f.py) 🔌) | stub | **P0** |
| Futures positioning | Commercial vs non-commercial | 🆓 CFTC COT weekly | 🆓 | **P0** |
| Short interest | Bi-monthly SI per name | 🆓 FINRA | 🆓 | **P0** |
| ETF flows | Daily creations/redemptions | 🆓 ETF.com scrape · 💰 Bloomberg/FactSet | 🆓 | P1 |
| Fund sector AUM | Monthly by fund type | 🆓 ICI statistics | 🆓 | P1 |
| Dealer positioning | Primary dealer net positions | 🆓 NY Fed weekly | 🆓 | P1 |
| Intra-firm structure (L3) | Desk-level allocation | 💰 partnership only (no public source) | — | P2 |

**Experiment E3 (P0/P1).** *Equilibrium consistency test*: build the empirical
distribution of institutional positions from 13F+COT+SI; solve the calibrated L2
MFG; measure W₂ distance between model equilibrium μ* and observed μ̂ each quarter.
A small, stable W₂ is direct evidence for the Nash-equilibrium description;
spikes in W₂ should align with the crisis episodes of §1.7.

### 1.5 Level 0 — cross-market Γ_t (§VI, notebook Day 15)

| Quantity | Data needed | Sources | Status | Priority |
|---|---|---|---|---|
| Dollar cycle | DXY, broad dollar index | 🔌 FRED (DTWEXBGS) | stub | **P0** |
| FX rates | G10 + EM daily | 🆓 ECB reference rates · 🆓 BIS | 🆓 | **P0** |
| Cross-border securities flows | Monthly by country | 🆓 US Treasury TIC | 🆓 | P1 |
| Fund-level country flows | Weekly EM/DM equity+bond flows | 💰 **EPFR** (the gold standard; academic access via some schools) | 💰 | P1 |
| Banking flows | Cross-border claims | 🆓 BIS locational statistics (quarterly) | 🆓 | P1 |
| Reserve composition | COFER currency shares | 🆓 IMF | 🆓 | P2 |
| CB balance sheets | Fed/ECB/BoJ/PBoC assets | 🆓 each CB publishes | 🆓 | **P0** |
| Country equity indices | Investable proxies | 🆓 ETF closes (EWJ, FXI, EEM…) · 💰 MSCI | 🆓 | **P0** |
| Sovereign risk | CDS spreads | 💰 Markit · 🆓 proxy: bond yield spreads (FRED) | 🆓 proxy | P2 |

**Experiment E6 (P1).** *L0 transmission*: re-run notebook Day 15 on real data —
does adding DXY + flow factors raise OOS R² for national equity indices, and does
cross-market correlation rise in high-DXY regimes as the model predicts?

### 1.6 Retail AI — μ_retail (§IX, `agents/retail_ai.py`, notebook Day 16)

This is the frontier component: it models data that mostly **does not exist yet**.

| Quantity | Data needed | Sources | Status | Priority |
|---|---|---|---|---|
| Historical retail holdings | Per-ticker popularity | 🆓 **Robintrack** archive (2018–2020, discontinued — a natural experiment frozen in amber) | 🆓 | P1 |
| Retail order flow | Off-exchange/PFOF share | 🆓 SEC Rule 606 filings (quarterly per broker) · 🆓 FINRA OTC transparency | 🆓 | P1 |
| Social coordination | WSB/StockTwits message volume | 🆓 Reddit academic dumps (Pushshift successors) · 🔌 StockTwits API | 🆓 | P1 |
| Search attention | Ticker-level search volume | 🆓 Google Trends | 🆓 | P1 |
| AI-tool adoption rate a_t | % of retail using LLM advisors | 🧪 no source — survey (Prolific/Qualtrics, n≈2 000, ~$1 500) | 🧪 | P1 |
| **LLM recommendation kernel R(q)** | What advice LLMs actually give retail investors | 🧪 **does not exist — Experiment E5 creates it** | 🧪 | **P0 (novelty)** |

**Experiment E5 (P0 — the workshop paper's novel dataset).** *The LLM query atlas*:
systematically audit the major consumer LLMs with a stratified corpus of retail-style
investment prompts (the 5 archetypes of `agents/retail_ai.py` × tickers × market
conditions × dates), parse the recommended allocations, and publish the response
kernel R̂(q) with its concentration, day-to-day drift, and cross-model herding
coefficient ĥ. Cost: ≈ $200 of API calls (prompt templates are already stubbed in
[retail_ai.py](agents/retail_ai.py) — `stub_llm_response`). Privacy-clean (no human
subjects), fully reproducible, and to our knowledge **no such public dataset exists**.
This single experiment turns Day 16 from a thesis into a measurement.

### 1.7 Crisis detector Λ_t (§VIII, `online/regime_detector.py`)

| Quantity | Data needed | Sources | Status | Priority |
|---|---|---|---|---|
| System leverage | Margin debt | 🆓 FINRA monthly | 🆓 | **P0** |
| Funding stress | Repo volumes/rates, OFR indices | 🆓 OFR (free, daily) | 🆓 | **P0** |
| Vol geometry | VIX term structure | 🆓 CBOE EOD | 🆓 | **P0** |
| Positioning crowding | From §1.4 sources | see above | — | **P0** |
| Correlation structure | From price panel §1.3 | see above | — | **P0** |
| **Labeled crisis episodes** | Ground-truth event windows | 🆓 self-constructed (below) | ✅ done | **P0** |

Curated episode table (the backtest ground truth): 1987-10 crash · 1994 bond massacre ·
1997 Asia · 1998 LTCM · 2000-03 dot-com · **2007-08 Quant Quake** · 2008-09 GFC ·
2010-05 Flash Crash · 2011-08 US downgrade · 2013 Taper Tantrum · 2015-08 CNY deval ·
2018-02 Volmageddon · 2018-12 · **2020-02 COVID** · 2021-01 GME · 2021 Archegos ·
2022 UK gilts · 2023-03 SVB · 2024-08 yen-carry unwind.

**Experiment E4 (P0).** *Λ_t backtest 1990–2025*: compute the indicator daily from
free data only (margin debt, OFR, CBOE, FINRA, 13F), measure lead time and
false-alarm rate on the episode table against VIX-threshold and credit-spread
baselines. The claim to beat: **positive median lead time with < 2 false alarms/year**.

### 1.8 Controller C + backtest (`controller/`, `backtest/`)

| Quantity | Data needed | Sources | Status | Priority |
|---|---|---|---|---|
| Survivorship-free returns | Delisting-adjusted panel | 🎓 CRSP (the only clean source) | 🎓 | **P0** |
| Trading costs | Spread + impact curves | 🎓 TAQ-estimated · 💰 broker TCA | 🎓 | P1 |
| Borrow costs | Stock loan fees | 💰 Markit · 🆓 iBorrowDesk scrape | 🆓 | P1 |
| Paper execution | Broker API | 🔌 Alpaca paper trading (free key) | stub | P1 |

---

## 2 · The full source catalogue by access tier

**Tier F — free, no key (34 sources).** FRED · EDGAR (13F/8-K/XBRL/full-text) ·
FINRA short interest & margin debt · CFTC COT · US Treasury TIC · BIS (locational,
derivatives, credit gaps) · IMF (COFER, IFS, WEO) · OFR monitors · NY Fed (primary
dealers, SCE) · CBOE EOD indices · ECB SDW · BoJ · PBoC · Fed H.4.1/H.8/Z.1 ·
NASDAQ Trader halts · Jay Ritter IPO · LoPucki BRD · ICI flows · GDELT 2.0 ·
Google Trends · Reddit academic dumps · SEC Rule 606 · iBorrowDesk · ETF.com ·
Stooq · Yahoo Finance · OECD MEI · World Bank · Eurostat · investing.com calendars ·
S&P/Russell press releases · exchange holiday calendars · UCSD/academic replication
archives · Kenneth French data library.

**Tier K — free tier with API key (stubs wired or trivial) (9).**
Polygon.io 🔌 · Alpaca 🔌-ready · FRED key 🔌 · NewsAPI 🔌 · Tiingo · AlphaVantage ·
Financial Modeling Prep · StockTwits · Quandl/Nasdaq Data Link community.

**Tier W — WRDS academic license (🎓 the decisive unlock) (10).**
CRSP · Compustat · **TAQ (millisecond)** · I/B/E/S · Thomson-Reuters 13F (s34) ·
SDC Platinum (M&A + issuance) · OptionMetrics · Audit Analytics · BoardEx · TRACE.
*HKUST and Stanford both subscribe. Joining a group with WRDS access converts
eight P0 rows above from blocked to available on day one — this is the concrete,
non-hand-wavy reason an academic collaboration is on the critical path.*

**Tier P — commercial / partnership (10).** EPFR · Markit (CDS, securities lending) ·
RavenPack · Bloomberg/Refinitiv terminals · Databento · SensorTower · HFR ·
WhaleWisdom Pro · Ortex · broker/prime-broker internal flow (partnership only).

**Tier X — must be created (2).** LLM recommendation kernel R(q) (**E5**) ·
retail AI adoption survey a_t.

---

## 3 · Priority roadmap

| Phase | Goal | Data unlocked by | Experiments |
|---|---|---|---|
| **P0 — NeurIPS workshop MVP** | 4 empirical results + 1 novel dataset | Free tier + FRED/Polygon keys + **WRDS via academic group** | E1 noise atlas · E2 operator tables · E4 Λ_t backtest · E5 LLM query atlas |
| **P1 — full paper** | Equilibrium test + L0 + retail | + I/B/E/S depth, EPFR (or TIC proxy), Robintrack, surveys | E3 MFG consistency · E6 L0 transmission |
| **P2 — industrial product** | Live daily pipeline | + Databento/RavenPack/Markit, execution APIs | walk-forward live paper-trading |

**The one-paragraph pitch this document supports:** the theory is closed, the code
runs, the experiments are specified to the level of individual data files. What is
missing is (i) WRDS-grade data access and (ii) ~$200 of LLM audit budget. E1+E2+E4+E5
are executable within one quarter by one person inside a group that has both.

---

## 4 · Repo stub inventory (activation map)

| File | Env var | Activates | Free tier? |
|---|---|---|---|
| [data/sources/polygon.py](data/sources/polygon.py) | `POLYGON_KEY` | Intraday bars, corporate actions | ✅ 5 req/min |
| [data/sources/fred.py](data/sources/fred.py) | `FRED_KEY` | All macro series | ✅ unlimited |
| [data/sources/news.py](data/sources/news.py) | `NEWSAPI_KEY` | Headline stream | ✅ 100 req/day |
| [data/scrapers/sec_13f.py](data/scrapers/sec_13f.py) | — (EDGAR is keyless) | 13F holdings | ✅ |
| [data/kafka/producer.py](data/kafka/producer.py) | `KAFKA_BOOTSTRAP` | Streaming ingest | local |
| [agents/retail_ai.py](agents/retail_ai.py) | `OPENAI_KEY` (commented) | E5 LLM audit | ~$200 total |
| [controller/execution.py](controller/execution.py) | `ALPACA_KEY` | Paper trading | ✅ |

Committed keys: **zero**, by policy. See repo README § Security.
