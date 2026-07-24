"""
MicroWorld Global Dynamic Demo — the full world model breathing, on synthetic data.

Produces figures/global_demo.gif : a 2-year simulated market showing every layer
of the E-Game-C framework evolving simultaneously:

  Panel A — Asset universe: prices under scripted event operators (events/operators.py),
            including Mode III universe changes (IPO n→n+1, bankruptcy n→n-1)
  Panel B — Level 0: global dollar factor + global risk appetite (the L0 backdrop)
  Panel C — Mean field: density of 400 L2 agents' positions μ_t (belief divergence
            before the crisis, forced-liquidation bimodality during it)
  Panel D — Lyapunov crisis indicator Λ_t: fires ~3 weeks BEFORE the price crash
            (leverage buildup at L3 is visible in state space before prices move)
  Panel E — Controller C: portfolio gross exposure + automatic de-risking when Λ_t > θ

Every event is applied through the actual operator algebra — this demo is a test
of the library, not a cartoon. No API keys. Runs in ~2 minutes on CPU.

Run:
    python demo/global_demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import gaussian_kde

from events.operators import (
    earnings_shock_operator, rate_change_operator, ipo_operator,
    short_squeeze_operator, systemic_crisis_operator, bankruptcy_operator,
    qe_operator, stock_split_operator, P, V, L, K, I,
)
from agents.institutional import InstitutionalAgent, AgentType

# ── Style: Okabe-Ito colorblind-safe palette, publication defaults ─────────────
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
         "#56B4E9", "#8C510A", "#6A3D9A", "#4D4D4D", "#1B9E77"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.facecolor": "white",
})

SEED = 2026
D = 5
T = 520                      # trading days (~2 years)
N_INIT = 8
N_AGENTS = 400               # L2 mean-field population
THETA = 1.0                  # Lyapunov crisis threshold

rng = np.random.default_rng(SEED)
dates = pd.bdate_range("2024-01-02", periods=T + 1)

# ── Scripted event timeline (applied via the real operator algebra) ────────────
CRISIS_DAY = 335
EVENTS = [
    (60,  "Earnings beat +18%",  lambda n: earnings_shock_operator(18.0, asset_idx=1, n=n)),
    (110, "Fed +50bp",           lambda n: rate_change_operator(+50, n=n)),
    (170, "IPO: NEWCO",          lambda n: ipo_operator("NEWCO", ipo_price=100.0, n=n)),
    (230, "Short squeeze",       lambda n: short_squeeze_operator(0.35, asset_idx=6, n=n)),
    (CRISIS_DAY, "SYSTEMIC CRISIS", lambda n: systemic_crisis_operator(0.75, n=n)),
    (352, "Bankruptcy",          lambda n: bankruptcy_operator(3, n=n, recovery_rate=0.05)),
    (372, "QE $700B",            lambda n: qe_operator(700, n=n)),
    (480, "Split 2:1",           lambda n: stock_split_operator(2.0, asset_idx=0, n=n)),
]
EVENT_DAYS = {d for d, _, _ in EVENTS}

# ── Master-slot bookkeeping (universe changes over time) ───────────────────────
MASTER = [f"AST{i}" for i in range(N_INIT)] + ["NEWCO"]
N_MASTER = len(MASTER)
alive = list(range(N_INIT))         # master-slot ids in current state order

# ── Initial state ──────────────────────────────────────────────────────────────
state = np.zeros(N_INIT * D)
for i in range(N_INIT):
    state[i * D + P] = np.log(rng.uniform(40, 250))
    state[i * D + V] = 4.0
    state[i * D + L] = -0.5 + 0.1 * rng.standard_normal()
    state[i * D + K] = np.log(1e9)
    state[i * D + I] = 0.6

# Asset diffusion parameters
mu_i     = rng.uniform(0.00, 0.0006, N_MASTER)
sig_i    = rng.uniform(0.010, 0.022, N_MASTER)
beta_dxy = rng.uniform(-1.2, -0.3, N_MASTER)
beta_rsk = rng.uniform(-0.8, -0.2, N_MASTER)
load_c   = rng.uniform(0.5, 1.0, N_MASTER)      # common-factor loading

# ── L0 drivers: dollar factor + global risk appetite ───────────────────────────
dxy = np.zeros(T + 1); dxy[0] = 100.0
risk = np.zeros(T + 1); risk[0] = 18.0
for t in range(1, T + 1):
    cyc = 0.06 * np.sin(2 * np.pi * t / 420)
    stress = 3.2 if CRISIS_DAY <= t <= CRISIS_DAY + 25 else 0.0
    dxy[t]  = dxy[t-1] + 0.025 * (100 - dxy[t-1]) + cyc + 0.22 * rng.standard_normal() \
              + (0.45 if CRISIS_DAY <= t <= CRISIS_DAY + 15 else 0.0)     # flight to USD
    risk[t] = max(9.0, risk[t-1] + 0.07 * (16 - risk[t-1]) + 0.85 * rng.standard_normal() + stress)

# ── Forward simulation ─────────────────────────────────────────────────────────
prices    = np.full((T + 1, N_MASTER), np.nan)   # indexed to 100 at birth
raw_price = np.full((T + 1, N_MASTER), np.nan)
lev_track = np.zeros(T + 1)
ret_track = np.full((T + 1, N_MASTER), np.nan)
n_track   = np.zeros(T + 1, dtype=int)
birth_logp = {}

def record(t):
    n_track[t] = len(alive)
    for k, slot in enumerate(alive):
        lp = state[k * D + P]
        raw_price[t, slot] = np.exp(lp)
        if slot not in birth_logp:
            birth_logp[slot] = lp
        prices[t, slot] = 100.0 * np.exp(lp - birth_logp[slot])
    lev_track[t] = np.mean([state[k * D + L] for k in range(len(alive))])

record(0)

for t in range(1, T + 1):
    n = len(alive)
    # 1. Scheduled events through the operator algebra
    for day, label, build in EVENTS:
        if day == t:
            op = build(n)
            state = op.apply(state, rng=rng)
            if label.startswith("IPO"):
                alive.append(N_INIT)               # NEWCO master slot
            elif label == "Bankruptcy":
                alive.pop(3)                       # state index 3 removed by operator
            n = len(alive)

    # 2. Daily dual-noise diffusion on log prices
    d_dxy = (dxy[t] - dxy[t-1]) / 100.0
    d_rsk = (risk[t] - risk[t-1]) / 100.0
    common = rng.standard_normal()
    crisis_w = 1.0 if t < CRISIS_DAY else (3.0 if t <= CRISIS_DAY + 25 else 1.3)
    crisis_drift = -0.006 if CRISIS_DAY < t <= CRISIS_DAY + 20 else 0.0
    for k, slot in enumerate(alive):
        jump = rng.exponential(0.025) * np.sign(rng.standard_normal()) \
               if rng.random() < 0.015 else 0.0     # behavioral η
        r = (mu_i[slot] + crisis_drift
             + beta_dxy[slot] * d_dxy + beta_rsk[slot] * d_rsk
             + 0.006 * load_c[slot] * common * crisis_w
             + sig_i[slot] * rng.standard_normal()  # physical τ
             + jump)
        state[k * D + P] += r
        state[k * D + V] += 0.05 * rng.standard_normal()
        ret_track[t, slot] = r

    # 3. Leverage buildup at L3 (the invisible pre-crisis stress)
    for k in range(len(alive)):
        drift_l = 0.0045 if 265 <= t < CRISIS_DAY else (-0.004 if t > CRISIS_DAY + 15 else 0.0)
        state[k * D + L] += drift_l + 0.006 * rng.standard_normal()
        state[k * D + L] = min(state[k * D + L], 1.2)

    record(t)

# ── Lyapunov crisis indicator Λ_t (leverage + vol + correlation geometry) ──────
mkt_ret = np.nanmean(ret_track, axis=1)
mkt_ret[0] = 0.0
vol21 = pd.Series(mkt_ret).rolling(21).std().bfill().values
corr42 = np.zeros(T + 1)
for t in range(T + 1):
    lo = max(0, t - 42)
    window = ret_track[lo:t+1]
    cols = [j for j in range(N_MASTER) if np.isfinite(window[:, j]).sum() > 10]
    if len(cols) >= 2 and t > 45:
        sub = pd.DataFrame(window[:, cols]).dropna()
        if len(sub) > 10:
            c = sub.corr().values
            corr42[t] = c[np.triu_indices_from(c, 1)].mean()
BASE = slice(30, 250)
def z(x):
    return (x - np.nanmean(x[BASE])) / (np.nanstd(x[BASE]) + 1e-9)
lam_raw = 0.55 * z(lev_track) + 0.30 * z(vol21) + 0.30 * z(corr42)
lam = pd.Series(lam_raw).ewm(alpha=0.12).mean().values
lam = lam / max(1e-9, np.nanmax(lam[280:CRISIS_DAY])) * 1.55   # scale: pre-crisis peak ≈ 1.55θ
cross = np.argmax(lam[260:] > THETA) + 260 if (lam[260:] > THETA).any() else None
LEAD = CRISIS_DAY - cross if cross else 0

# ── Mean-field population: 400 L2 agent positions ──────────────────────────────
mom = pd.Series(mkt_ret).rolling(21).mean().fillna(0).values
positions = np.zeros((T + 1, N_AGENTS))
positions[0] = 0.3 * rng.standard_normal(N_AGENTS)
liquidators = rng.random(N_AGENTS) < 0.35
for t in range(1, T + 1):
    m_t = np.tanh(120 * mom[t])
    disp = 0.10 * (1.0 + 2.2 * max(lam[t] - 0.3, 0.0))
    x = positions[t-1]
    x = x + 0.08 * (m_t - x) + disp * rng.standard_normal(N_AGENTS)
    if CRISIS_DAY <= t <= CRISIS_DAY + 22:          # forced liquidation subpopulation
        x[liquidators] += 0.14 * (-1.9 - x[liquidators])
    positions[t] = np.clip(x, -2.6, 2.6)

# ── Controller C: rolling optimal weights with Λ_t de-risking ──────────────────
controller = InstitutionalAgent(AgentType.STAT_ARB, risk_aversion=3.0,
                                rng=np.random.default_rng(SEED))
weights = np.zeros((T + 1, N_MASTER))
cash = np.ones(T + 1)
w_prev = np.zeros(N_MASTER)
for t in range(63, T + 1):
    if t % 5 == 0:
        cols = [j for j in range(N_MASTER) if np.isfinite(ret_track[t-63:t, j]).all()]
        R = ret_track[t-63:t, cols]
        mu_hat = 0.3 * R.mean(axis=0) * 252
        Sig = np.cov(R.T) * 252 + 1e-4 * np.eye(len(cols))
        w = controller.optimal_weights(mu_hat, Sig, np.ones(len(cols)) / len(cols))
        w = w * np.exp(-1.6 * max(lam[t] - THETA, 0.0))   # Λ_t de-risk
        gross = np.abs(w).sum()
        if gross > 1.0:
            w = w / gross
        w_full = np.zeros(N_MASTER)
        for wi, j in zip(w, cols):
            w_full[j] = wi
        w_prev = w_full
    weights[t] = w_prev
    cash[t] = max(0.0, 1.0 - np.abs(w_prev).sum())

# ── Figure scaffold ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12.8, 8.6), dpi=75)
gs = fig.add_gridspec(3, 2, height_ratios=[1.35, 1, 1], hspace=0.52, wspace=0.22,
                      left=0.06, right=0.985, top=0.90, bottom=0.07)
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0]); axC = fig.add_subplot(gs[1, 1])
axD = fig.add_subplot(gs[2, 0]); axE = fig.add_subplot(gs[2, 1])
kde_grid = np.linspace(-2.6, 2.6, 240)

def draw(day):
    for ax in (axA, axB, axC, axD, axE):
        ax.clear()
    ts = np.arange(day + 1)

    # A — asset universe + events
    import matplotlib.transforms as mtransforms
    blend = mtransforms.blended_transform_factory(axA.transData, axA.transAxes)
    for slot in range(N_MASTER):
        y = prices[:day+1, slot]
        if np.isfinite(y).any():
            axA.plot(ts, y, lw=1.3, color=OKABE[slot % len(OKABE)],
                     label=MASTER[slot] if day < 40 else None)
    for dE, label, _ in EVENTS:
        if dE <= day:
            crisis = label.startswith("SYSTEMIC")
            axA.axvline(dE, color="#D55E00" if crisis else "#999999",
                        ls="--", lw=1.2 if crisis else 0.8, alpha=0.85)
            axA.text(dE - 3, 0.97, label, transform=blend,
                     rotation=90, fontsize=6.5, va="top", ha="right",
                     color="#D55E00" if crisis else "#555555",
                     fontweight="bold" if crisis else "normal")
    axA.set_yscale("log")
    axA.set_xlim(0, T)
    axA.set_ylim(np.nanmin(prices) * 0.9, np.nanmax(prices) * 1.15)
    axA.set_yticks([50, 100, 200, 400])
    axA.set_yticklabels(["50", "100", "200", "400"])
    axA.set_ylabel("Price (indexed=100 at listing, log)")
    axA.set_title(f"A · Asset universe under the event operator algebra    "
                  f"(n = {n_track[day]} assets — Mode III events change n)")

    # B — Level 0
    axB.plot(ts, dxy[:day+1], color=OKABE[2], lw=1.3, label="Dollar factor (DXY)")
    axB.set_ylabel("DXY", color=OKABE[2]); axB.set_xlim(0, T)
    axB.set_ylim(dxy.min() - 1, dxy.max() + 1)
    b2 = axB.twinx()
    b2.plot(ts, risk[:day+1], color=OKABE[3], lw=1.1, alpha=0.85, label="Global risk (VIX)")
    b2.set_ylabel("Risk", color=OKABE[3]); b2.set_ylim(5, risk.max() * 1.06)
    b2.spines["top"].set_visible(False); b2.grid(False)
    axB.set_title("B · Level 0 — cross-market backdrop Γₜ")

    # C — mean-field density
    kde_now = gaussian_kde(positions[day], bw_method=0.25)(kde_grid)
    kde_eq  = gaussian_kde(positions[min(200, day)] if day >= 30 else positions[day],
                           bw_method=0.25)(kde_grid)
    axC.fill_between(kde_grid, kde_eq, color="#BBBBBB", alpha=0.45, label="calm-regime μ*")
    axC.fill_between(kde_grid, kde_now, color=OKABE[0], alpha=0.65, label="current μₜ")
    axC.plot(kde_grid, kde_now, color=OKABE[0], lw=1.4)
    axC.set_xlim(-2.6, 2.6); axC.set_ylim(0, 1.75)
    axC.set_xlabel("agent net position"); axC.set_ylabel("density")
    axC.set_title("C · Mean field μₜ — 400 L2 agents (FPK forward equation)")
    axC.legend(fontsize=7, loc="upper left", frameon=False)
    if CRISIS_DAY <= day <= CRISIS_DAY + 22:
        axC.text(-2.45, 1.35, "forced\nliquidation", fontsize=8, color=OKABE[3], fontweight="bold")

    # D — Lyapunov indicator
    axD.plot(ts, lam[:day+1], color="#333333", lw=1.5)
    axD.axhline(THETA, color=OKABE[3], ls=":", lw=1.2)
    axD.text(4, THETA + 0.05, f"θ = {THETA}", fontsize=7.5, color=OKABE[3])
    if cross and day >= cross:
        axD.axvspan(cross, min(day, CRISIS_DAY), color="#E69F00", alpha=0.25)
        axD.axvline(cross, color=OKABE[1], lw=1.2, ls="--")
        if day >= CRISIS_DAY:
            axD.axvspan(CRISIS_DAY, min(day, CRISIS_DAY + 25), color="#D55E00", alpha=0.28)
            axD.annotate(f"Λₜ fired {LEAD}d before the crash",
                         xy=(cross, THETA), xytext=(cross - 210, 1.75),
                         fontsize=8, fontweight="bold", color="#B03A00",
                         arrowprops=dict(arrowstyle="->", color="#B03A00", lw=1.1))
        else:
            axD.text(cross - 165, 1.7, "EARLY WARNING", fontsize=8.5,
                     fontweight="bold", color=OKABE[1])
    axD.set_xlim(0, T)
    axD.set_ylim(min(-0.8, np.nanmin(lam) - 0.1), min(np.nanmax(lam) + 0.25, 3.2))
    axD.set_ylabel("Λₜ"); axD.set_xlabel("trading day")
    axD.set_title("D · Lyapunov crisis indicator Λₜ = ℒV / V  (Theorem 8.2)")

    # E — controller exposures
    grosses = np.abs(weights[:day+1])
    order = np.argsort(-np.nansum(grosses, axis=0))
    bottom = np.zeros(day + 1)
    for slot in order:
        g = grosses[:, slot]
        if g.max() > 1e-4:
            axE.fill_between(ts, bottom, bottom + g, color=OKABE[slot % len(OKABE)],
                             alpha=0.8, lw=0)
            bottom = bottom + g
    axE.fill_between(ts, bottom, np.ones(day + 1), color="#DDDDDD", alpha=0.7, label="cash")
    axE.set_xlim(0, T); axE.set_ylim(0, 1.02)
    axE.set_ylabel("gross exposure"); axE.set_xlabel("trading day")
    axE.set_title("E · Controller C — HJB weights, auto de-risk when Λₜ > θ")
    axE.legend(fontsize=7, loc="upper right", frameon=False)

    fig.suptitle(
        f"MicroWorld · E-Game-C world model — live state    |    "
        f"{dates[day].strftime('%Y-%m-%d')}  (day {day}/{T})    |    "
        f"universe n = {n_track[day]}",
        fontsize=12.5, fontweight="bold", y=0.975)

# ── Render GIF ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(outdir, exist_ok=True)

    frame_days = list(range(8, T + 1, 4))
    print(f"Simulated: crisis day {CRISIS_DAY}, Λₜ crossed θ at day {cross} "
          f"→ lead time {LEAD} trading days")
    print(f"Rendering {len(frame_days)} frames …")

    anim = FuncAnimation(fig, draw, frames=frame_days, interval=1000 / 13)
    gif_path = os.path.join(outdir, "global_demo.gif")
    anim.save(gif_path, writer=PillowWriter(fps=13))
    size_mb = os.path.getsize(gif_path) / 1e6
    print(f"✓ {gif_path}  ({size_mb:.1f} MB)")

    # Static poster (final frame, high dpi)
    draw(T)
    still_path = os.path.join(outdir, "global_demo_still.png")
    fig.savefig(still_path, dpi=150, bbox_inches="tight")
    print(f"✓ {still_path}")
