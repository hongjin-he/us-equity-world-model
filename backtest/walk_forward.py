"""
§8 — Walk-forward backtesting protocol.
Simulates the daily retrain cycle to avoid lookahead bias.

train_window = 504 trading days (~2yr)
test_window  = 21  trading days (~1mo)
retrain_freq = 21  trading days
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class BacktestResult:
    dates: list
    pnl: list
    weights: list
    sharpe: float
    max_drawdown: float
    hit_rate: float
    turnover: float
    mfg_pred_r2: float


def walk_forward_backtest(
    data: pd.DataFrame,
    train_fn,
    train_window: int = 504,
    test_window: int = 21,
    retrain_freq: int = 21,
    tc_bps: float = 10.0,
) -> BacktestResult:
    """
    data: MultiIndex DataFrame (date × ticker), columns = features + 'returns'
    train_fn: callable(train_data) → fitted E-Game-C model
    tc_bps: one-way transaction cost in basis points
    """
    results = []
    prev_weights = None

    for train_end in range(train_window, len(data) - test_window, retrain_freq):
        train_data = data.iloc[train_end - train_window: train_end]
        test_data  = data.iloc[train_end: train_end + test_window]

        model = train_fn(train_data)

        for t in range(len(test_data) - 1):
            obs  = test_data.iloc[:t + 1]
            z_t  = model.encode(obs)
            w_t  = model.get_weights(z_t)
            r_t1 = test_data.iloc[t + 1]["returns"]

            pnl = float((w_t * r_t1).sum())
            if prev_weights is not None:
                tc = (tc_bps / 1e4) * float(np.abs(w_t - prev_weights).sum())
                pnl -= tc
            prev_weights = w_t.copy()

            results.append({
                "date":    test_data.index[t],
                "pnl":     pnl,
                "weights": w_t,
            })

    df = pd.DataFrame(results)
    return _compute_metrics(df)


def _compute_metrics(df: pd.DataFrame) -> BacktestResult:
    pnl = np.array(df["pnl"])
    sharpe = float(np.mean(pnl) / (np.std(pnl) + 1e-8) * np.sqrt(252))

    equity = np.cumprod(1 + pnl)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = float(drawdowns.min())

    hit_rate = float((pnl > 0).mean())

    weights = np.array(df["weights"].tolist())
    turnover = float(np.abs(np.diff(weights, axis=0)).sum(axis=1).mean() / 2)

    return BacktestResult(
        dates=df["date"].tolist(),
        pnl=pnl.tolist(),
        weights=weights.tolist(),
        sharpe=sharpe,
        max_drawdown=max_dd,
        hit_rate=hit_rate,
        turnover=turnover,
        mfg_pred_r2=float("nan"),   # filled separately after MFG evaluation
    )


PERFORMANCE_TARGETS = {
    "sharpe":       ("Annualized Sharpe",  "> 1.5"),
    "max_drawdown": ("Max Drawdown",       "< 20%"),
    "hit_rate":     ("Hit Rate",           "> 52%"),
    "turnover":     ("Daily Turnover",     "< 30%"),
    "mfg_pred_r2":  ("MFG Pred R²",        "> 0.05"),
}
