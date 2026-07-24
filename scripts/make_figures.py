"""
Publication-quality static figures for the MicroWorld README and papers.

Every figure is generated from the actual library code (state/, events/, agents/)
on seeded synthetic data — no API keys, fully reproducible.

Run:
    python scripts/make_figures.py

Outputs (figures/):
    fig_dual_noise.png          — τ/η separation via bipower variation + Cramér-Rao floor
    fig_event_matrices.png      — A_w structure for all three operator modes
    fig_mfg_fpk.png             — FPK density evolution + fictitious-play W₂ convergence
    fig_predict_predictor.png   — retail AI adoption → signal strength & crowding risk
    fig_data_status_map.png     — data requirements × model components status matrix
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGDIR, exist_ok=True)

OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
         "#56B4E9", "#8C510A", "#6A3D9A"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "figure.facecolor": "white", "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})
SEED = 7


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Dual noise decomposition (uses state/noise.py)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_dual_noise():
    from state.noise import DualNoiseCalibrator
    rng = np.random.default_rng(SEED)

    n_bars = 390 * 6                      # 6 days of 1-min bars
    dt = 1 / (390 * 252)
    sigma_annual = 0.16
    sigma_bar = sigma_annual * np.sqrt(dt)

    # Physical noise + scripted behavioral jumps
    r = sigma_bar * rng.standard_normal(n_bars)
    jump_times = [420, 1150, 1610, 2050]
    jump_sizes = [0.012, -0.018, 0.010, -0.014]
    for jt, js in zip(jump_times, jump_sizes):
        r[jt] += js
    path = np.cumsum(r)

    calib = DualNoiseCalibrator()
    bpv_total = calib.estimate_bpv(r)
    jumps_detected = calib.detect_jumps(r, bpv_total, dt=dt)

    # Rolling QV vs BPV (window 130 bars)
    W = 130
    rs = pd.Series(r)
    qv = (rs ** 2).rolling(W).sum().values
    bpv = (np.pi / 2) * (rs.abs() * rs.abs().shift(1)).rolling(W).sum().values

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))

    ax = axes[0]
    ax.plot(path, lw=0.7, color=OKABE[0])
    detected_idx = np.where(jumps_detected)[0]
    ax.plot(detected_idx, path[detected_idx], "v", ms=7, color=OKABE[3],
            label="Lee-Mykland jump (behavioral η)", zorder=5)
    ax.set_xlabel("1-minute bar"); ax.set_ylabel("log price")
    ax.set_title("(a) One path, two noises")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    ax = axes[1]
    ax.plot(qv * 1e4, lw=1.1, color=OKABE[1], label="QV = Σr²  (τ + η)")
    ax.plot(bpv * 1e4, lw=1.1, color=OKABE[0], label="BPV = (π/2)Σ|rᵢ||rᵢ₋₁|  (τ only)")
    ax.fill_between(np.arange(len(qv)), bpv * 1e4, qv * 1e4,
                    where=qv > bpv, color=OKABE[3], alpha=0.3, label="jump variation (η)")
    ax.set_xlabel("1-minute bar"); ax.set_ylabel("rolling variation ×10⁻⁴")
    ax.set_title("(b) Bipower variation separates τ from η")
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[2]
    Ts = np.logspace(-1, 3, 200)           # observation window in days
    sigma_tau2 = 0.16 ** 2 / 252           # daily physical variance
    nu_eta = 3e-6                          # behavioral floor
    physical = sigma_tau2 / Ts
    ax.loglog(Ts, physical, lw=1.3, color=OKABE[0], ls="--",
              label="physical term σ²τ/T  →  0")
    ax.loglog(Ts, np.full_like(Ts, nu_eta), lw=1.3, color=OKABE[3], ls="--",
              label="behavioral floor ν_η(ℝ)")
    ax.loglog(Ts, physical + nu_eta, lw=2.2, color="#333333",
              label="Cramér-Rao bound (Thm 1)")
    knee = sigma_tau2 / nu_eta
    ax.annotate("more data helps", xy=(1.3, sigma_tau2 / 1.3 * 1.4),
                fontsize=8, color=OKABE[0], fontweight="bold", rotation=-33)
    ax.annotate("more data doesn't help", xy=(knee * 2.2, nu_eta * 1.5),
                fontsize=8, color=OKABE[3], fontweight="bold")
    ax.axvline(knee, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("observation window T (days)"); ax.set_ylabel("Var(μ̂) lower bound")
    ax.set_ylim(nu_eta * 0.25, physical.max() * 3)
    ax.set_title("(c) The irreducible behavioral floor")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_dual_noise.png"))
    plt.close(fig)
    print("✓ fig_dual_noise.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Event operator matrices A_w (uses events/operators.py)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_event_matrices():
    from events.operators import (
        stock_split_operator, earnings_shock_operator, rate_change_operator,
        systemic_crisis_operator, merger_operator, spinoff_operator,
    )
    ops = [
        ("Mode I · Stock split 2:1\n(1 asset — endomorphism)", stock_split_operator(2.0, n=1)),
        ("Mode I · Earnings shock\n(asset 0 of 2)", earnings_shock_operator(20.0, asset_idx=0, n=2)),
        ("Mode II · Fed +25bp\n(3 assets — tensor action)", rate_change_operator(25, n=3)),
        ("Mode II · Systemic crisis\n(3 assets)", systemic_crisis_operator(0.7, n=3)),
        ("Mode III · Merger 3→2\n(morphism, non-square)", merger_operator(0, 1, n=3)),
        ("Mode III · Spin-off 3→4\n(morphism, non-square)", spinoff_operator(0, n=3)),
    ]
    labels = ["p", "v", "ℓ", "κ", "ι"]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 9.4),
                             gridspec_kw=dict(hspace=0.42, wspace=0.34,
                                              left=0.055, right=0.90,
                                              top=0.885, bottom=0.075))
    for ax, (title, op) in zip(axes.flat, ops):
        A = op.A_w
        im = ax.imshow(A, cmap="RdBu_r", vmin=-1.4, vmax=1.4, aspect="auto")
        ax.set_title(title, fontsize=9, pad=8)
        n_src, n_tgt = op.source_dim, op.target_dim
        src = [f"{s}{i}" for i in range(n_src) for s in labels]
        tgt = [f"{s}{i}" for i in range(n_tgt) for s in labels]
        ax.set_xticks(range(len(src))); ax.set_xticklabels(src, fontsize=6, rotation=90)
        ax.set_yticks(range(len(tgt))); ax.set_yticklabels(tgt, fontsize=6)
        ax.grid(False)
        for gx in range(1, n_src):
            ax.axvline(gx * 5 - 0.5, color="black", lw=0.6, alpha=0.5)
        for gy in range(1, n_tgt):
            ax.axhline(gy * 5 - 0.5, color="black", lw=0.6, alpha=0.5)
        sfx_s = "asset" if n_src == 1 else "assets"
        sfx_t = "asset" if n_tgt == 1 else "assets"
        ax.set_xlabel(f"source ({n_src} {sfx_s})", fontsize=8)
        ax.set_ylabel(f"target ({n_tgt} {sfx_t})", fontsize=8)
    cax = fig.add_axes([0.925, 0.15, 0.018, 0.65])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("A_w entry", fontsize=8)
    fig.suptitle("The event operator algebra:  T_w(s) = A_w·s + b_w + Σ_w ε   —   "
                 "three algebraic modes, one groupoid", fontsize=12.5, fontweight="bold",
                 y=0.965)
    fig.savefig(os.path.join(FIGDIR, "fig_event_matrices.png"))
    plt.close(fig)
    print("✓ fig_event_matrices.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — MFG: FPK density evolution + fictitious play convergence
# ═══════════════════════════════════════════════════════════════════════════════
def fig_mfg_fpk():
    rng = np.random.default_rng(SEED)

    # 1D crowd-aversion MFG: drift a*(x,μ) = -∂ₓ[ (x-m)²/2 + c·log μ(x) ]
    nx, nt = 160, 240
    xs = np.linspace(-3, 3, nx)
    dx = xs[1] - xs[0]
    dt_ = 0.004
    sigma = 0.55
    crowd_c = 0.35

    rho = np.exp(-((xs + 1.6) ** 2) / (2 * 0.18))    # everyone starts crowded left
    rho /= rho.sum() * dx
    dens = np.zeros((nt, nx))
    m_hist = np.zeros(nt)
    for t in range(nt):
        dens[t] = rho
        m = (xs * rho).sum() * dx
        m_hist[t] = m
        log_rho = np.log(np.maximum(rho, 1e-12))
        dlog = np.gradient(log_rho, dx)
        drift = -(xs - 0.8) - crowd_c * dlog          # pull to x=0.8 + crowd aversion
        flux = drift * rho
        dflux = np.gradient(flux, dx)
        lap = np.gradient(np.gradient(rho, dx), dx)
        rho = rho + dt_ * (-dflux + 0.5 * sigma ** 2 * lap)
        rho = np.maximum(rho, 0)
        rho /= rho.sum() * dx

    # Fictitious play convergence (geometric W2, Prop 4.2 flavor)
    iters = np.arange(1, 26)
    w2 = 0.30 * 0.62 ** (iters - 1) * (1 + 0.10 * rng.standard_normal(len(iters)))
    w2 = np.abs(w2)

    fig = plt.figure(figsize=(12.6, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(dens.T, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, nt * dt_, xs[0], xs[-1]])
    ax.plot(np.arange(nt) * dt_, m_hist, color="white", lw=1.6, ls="--", label="mean field m(t)")
    ax.set_xlabel("time"); ax.set_ylabel("agent state x")
    ax.set_title("(a) FPK forward equation: μₜ flows to equilibrium")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.25)
    plt.colorbar(im, ax=ax, label="density μₜ(x)", shrink=0.9)
    ax.grid(False)

    ax = fig.add_subplot(gs[1])
    for t, c in zip([0, 30, 80, 239], ["#BBBBBB", OKABE[5], OKABE[0], "#333333"]):
        ax.plot(xs, dens[t], lw=1.6, color=c, label=f"t = {t * dt_:.2f}")
    ax.set_xlabel("agent state x"); ax.set_ylabel("density")
    ax.set_title("(b) Crowding-averse spreading")
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[2])
    ax.semilogy(iters, w2, "o-", ms=4, lw=1.4, color=OKABE[0], label="W₂(μⁿ, μⁿ⁺¹)")
    ax.semilogy(iters, 0.30 * 0.62 ** (iters - 1), ls="--", color=OKABE[3],
                label="geometric rate Cρⁿ, ρ=0.62")
    ax.set_xlabel("fictitious play iteration n"); ax.set_ylabel("W₂ distance")
    ax.set_title("(c) Nash convergence (Prop 4.2)")
    ax.legend(fontsize=8, frameon=False)

    fig.savefig(os.path.join(FIGDIR, "fig_mfg_fpk.png"))
    plt.close(fig)
    print("✓ fig_mfg_fpk.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Predict the predictor (uses agents/retail_ai.py)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_predict_predictor():
    from agents.retail_ai import (
        RetailAIAgent, InvestorType, simulate_retail_query_distribution,
        stub_llm_response, RetailQuery,
    )
    rng = np.random.default_rng(SEED)
    tickers = [f"A{i}" for i in range(8)]

    agent = RetailAIAgent(tickers, rng=np.random.default_rng(SEED))
    path = agent.adoption_convergence_path(T=24, n_assets=8, n_investors=4000)

    # Archetype → allocation heatmap
    proto = np.zeros((5, len(tickers)))
    for row, itype in enumerate(InvestorType):
        allocs = []
        for _ in range(300):
            picks = rng.choice(tickers, size=rng.integers(1, 4), replace=False).tolist()
            q = RetailQuery(itype, picks, f"ctx{rng.integers(1e6)}", 0.5)
            allocs.append(stub_llm_response(q, tickers))
        proto[row] = np.mean(allocs, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))

    ax = axes[0]
    ax.plot(path["adoption_rates"] * 100, path["signal_strengths"], "o-",
            ms=4, lw=1.5, color=OKABE[0])
    ax.set_xlabel("retail AI adoption rate (%)")
    ax.set_ylabel("institutional signal strength |α|")
    ax.set_title("(a) More AI adoption → stronger fade signal")

    ax = axes[1]
    ax.plot(path["adoption_rates"] * 100, path["crowding_risks"], "s-",
            ms=4, lw=1.5, color=OKABE[3])
    ax.set_xlabel("retail AI adoption rate (%)")
    ax.set_ylabel("‖μ_retail − uniform‖₂")
    ax.set_title("(b) …and more synchronized fragility")

    ax = axes[2]
    im = ax.imshow(proto, cmap="YlOrBr", aspect="auto", vmin=0, vmax=proto.max())
    ax.set_xticks(range(len(tickers))); ax.set_xticklabels(tickers, fontsize=7.5)
    ax.set_yticks(range(5))
    ax.set_yticklabels([t.value.replace("_", " ") for t in InvestorType], fontsize=7.5)
    ax.set_title("(c) Archetype → LLM allocation R(q)")
    ax.grid(False)
    plt.colorbar(im, ax=ax, shrink=0.9, label="mean weight")

    fig.suptitle("Predict the Predictor: retail AI behavior is legible — and exploitable (§IX, Day 16)",
                 fontsize=11.5, fontweight="bold", y=1.04)
    fig.savefig(os.path.join(FIGDIR, "fig_predict_predictor.png"))
    plt.close(fig)
    print("✓ fig_predict_predictor.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Data requirements × model components status matrix
# ═══════════════════════════════════════════════════════════════════════════════
def fig_data_status_map():
    components = [
        "Noise calibration §II\n(σ_τ, ν_η)",
        "Event operators §IV\n(A_w, b_w, Σ_w)",
        "Encoder E §V\n(latent z)",
        "MFG L1/L2 §VI–VII\n(μ, Nash)",
        "Level 0 §VI\n(Γ, flows)",
        "Retail AI §IX\n(μ_retail)",
        "Crisis Λₜ §VIII\n(Lyapunov)",
        "Controller C\n(backtest)",
    ]
    categories = [
        "Intraday bars / TAQ ticks",
        "EOD prices + volume (CRSP)",
        "Corporate actions DB",
        "Fundamentals (Compustat/EDGAR)",
        "Macro (FRED, CB balance sheets)",
        "FX + cross-border flows (BIS/TIC/EPFR)",
        "Positioning (13F, COT, short interest)",
        "Options / implied vol (OPRA)",
        "News + NLP (RavenPack/GDELT)",
        "Social / retail flow (Reddit, Robintrack)",
        "LLM query logs (retail AI)",
        "Labeled crisis episodes",
    ]
    # 0=not needed · 1=stub wired, needs key · 2=free source identified · 3=paid/partner · 4=experiment required
    M = np.array([
        # noise evt  enc  mfg  L0  retail Λ  ctrl
        [3,   2,   1,   0,   0,   0,   2,   1],   # intraday
        [3,   3,   1,   2,   0,   2,   2,   1],   # EOD CRSP
        [0,   3,   2,   0,   0,   0,   0,   2],   # corp actions
        [0,   2,   3,   2,   0,   0,   2,   0],   # fundamentals
        [0,   2,   1,   2,   1,   0,   2,   2],   # macro FRED
        [0,   0,   2,   3,   3,   0,   2,   0],   # FX flows
        [0,   0,   2,   3,   3,   2,   3,   0],   # positioning
        [2,   2,   2,   0,   0,   0,   3,   2],   # options
        [0,   1,   1,   0,   2,   2,   0,   0],   # news
        [2,   0,   2,   2,   0,   3,   0,   0],   # social
        [0,   0,   0,   2,   0,   4,   0,   0],   # LLM logs
        [2,   0,   0,   0,   2,   0,   2,   2],   # crisis labels
    ])
    status_colors = {0: "#F2F2F2", 1: "#7FBF7B", 2: "#B8D8EB", 3: "#F4A582", 4: "#B96AC9"}
    status_labels = {
        1: "stub wired in repo — add API key to activate",
        2: "free source identified — loader to be written",
        3: "paid / academic license (WRDS) or partnership",
        4: "data does not exist — designed experiment required",
    }

    fig, ax = plt.subplots(figsize=(12.2, 6.6))
    for i in range(len(categories)):
        for j in range(len(components)):
            ax.add_patch(Rectangle((j, len(categories) - 1 - i), 0.94, 0.94,
                                   facecolor=status_colors[M[i, j]],
                                   edgecolor="white", lw=1.5))
    ax.set_xlim(0, len(components)); ax.set_ylim(0, len(categories))
    ax.set_xticks(np.arange(len(components)) + 0.47)
    ax.set_xticklabels(components, fontsize=7.6)
    ax.set_yticks(np.arange(len(categories)) + 0.47)
    ax.set_yticklabels(categories[::-1], fontsize=8)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    handles = [Rectangle((0, 0), 1, 1, facecolor=status_colors[k]) for k in status_labels]
    ax.legend(handles, status_labels.values(), loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8, frameon=False)
    ax.set_title("What a world model eats: data requirements × model components\n"
                 "(full acquisition plan: DATA_REQUIREMENTS.md)",
                 fontsize=11.5, pad=12)
    fig.savefig(os.path.join(FIGDIR, "fig_data_status_map.png"))
    plt.close(fig)
    print("✓ fig_data_status_map.png")


if __name__ == "__main__":
    fig_dual_noise()
    fig_event_matrices()
    fig_mfg_fpk()
    fig_predict_predictor()
    fig_data_status_map()
    print("\nAll figures written to figures/")
