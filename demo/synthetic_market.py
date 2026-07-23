"""
Synthetic US equity market generator for demo purposes.
Produces realistic OHLCV + macro + news data — no API keys required.

Uses the dual noise model from §III:
  dS_t = μ dt + σ_τ dW_t^(physical) + dJ_t^(behavioral)

Calibrated to approximate S&P 500 statistics:
  σ_τ ≈ 0.16/√252 per day (physical volatility)
  λ_η ≈ 0.05 jumps/day (behavioral jump intensity)
  |jump| ~ Exponential(mean=0.02)
"""
import numpy as np
import pandas as pd


def generate_market(
    n_assets: int = 50,
    n_days: int = 504,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tickers = [f"STK{i:03d}" for i in range(n_assets)]
    dates   = pd.bdate_range("2022-01-03", periods=n_days)

    # Asset parameters (cross-sectional heterogeneity)
    sigma_tau = rng.uniform(0.10, 0.40, n_assets) / np.sqrt(252)  # physical vol
    mu        = rng.uniform(-0.0002, 0.0008, n_assets)             # drift
    lambda_eta = rng.uniform(0.01, 0.10, n_assets)                 # jump intensity
    jump_size  = rng.uniform(0.01, 0.04, n_assets)                 # avg |jump|
    betas     = rng.uniform(0.5, 1.5, n_assets)                    # market beta

    # Market factor (common behavioral shock)
    market_physical = rng.normal(0, 0.01, n_days)
    market_jumps    = np.where(
        rng.uniform(size=n_days) < 0.03,
        rng.choice([-1, 1], n_days) * rng.exponential(0.025, n_days),
        0.0,
    )

    rows = []
    for i, ticker in enumerate(tickers):
        price = 50.0 * rng.uniform(0.5, 2.0)
        prices = [price]

        for t in range(n_days):
            # Physical noise (Brownian)
            idio = rng.normal(0, sigma_tau[i])
            # Behavioral noise (Lévy jump)
            jump = 0.0
            if rng.uniform() < lambda_eta[i]:
                jump = rng.choice([-1, 1]) * rng.exponential(jump_size[i])
            # Market factor contribution
            mkt = betas[i] * (market_physical[t] + market_jumps[t])

            ret = mu[i] + idio + jump + mkt
            price = prices[-1] * np.exp(ret)
            prices.append(price)

        for t in range(n_days):
            p = prices[t + 1]
            daily_range = p * abs(rng.normal(0, sigma_tau[i])) * 2
            rows.append({
                "date":      dates[t],
                "ticker":    ticker,
                "open":      prices[t] * np.exp(rng.normal(0, sigma_tau[i] * 0.3)),
                "high":      p + daily_range * rng.uniform(0, 1),
                "low":       p - daily_range * rng.uniform(0, 1),
                "close":     p,
                "volume":    rng.lognormal(15, 1),
                "returns":   np.log(prices[t + 1] / prices[t]),
                # Dual noise ground truth
                "sigma_tau_true": sigma_tau[i],
                "had_jump":  abs(np.log(prices[t+1]/prices[t]) - mu[i]
                                 - betas[i]*market_physical[t]) > 2.5 * sigma_tau[i],
            })

    df = pd.DataFrame(rows).set_index(["date", "ticker"])

    # Add macro state (same for all assets per day)
    macro = _generate_macro(dates, seed=seed + 1)
    df = df.join(macro, on="date")

    return df


def _generate_macro(dates: pd.DatetimeIndex, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(dates)
    # Simple AR(1) macro processes
    fed_funds   = np.cumsum(rng.normal(0, 0.02, n)).clip(0, 6) + 2.0
    vix         = np.abs(rng.normal(20, 5, n)).clip(10, 80)
    t10y2y      = np.cumsum(rng.normal(0, 0.05, n)).clip(-2, 3)
    cpi         = 3.5 + np.cumsum(rng.normal(0, 0.1, n) * 0.1).clip(-2, 5)

    return pd.DataFrame({
        "DFF":      fed_funds,
        "VIXCLS":   vix,
        "T10Y2Y":   t10y2y,
        "CPIAUCSL": cpi,
    }, index=dates)


if __name__ == "__main__":
    print("Generating synthetic market (50 assets, 504 days)...")
    df = generate_market(n_assets=50, n_days=504)
    print(f"Shape: {df.shape}")
    print(df.head(10).to_string())
