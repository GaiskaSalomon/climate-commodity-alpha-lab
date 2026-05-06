"""
Transaction cost utilities for commodity strategy backtesting.

The core cost logic (cost = turnover * bps / 10_000) lives in
performance.py::compute_strategy_returns(). This module provides
standalone helpers for cost analysis and sensitivity testing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cost_per_trade(cost_bps: float = 5.0) -> float:
    """Convert basis points to a fractional cost per unit of turnover."""
    return cost_bps / 10_000


def compute_turnover(positions: pd.Series) -> pd.Series:
    """
    Compute daily turnover as absolute change in lagged position.

    Turnover = |position_t - position_{t-1}|

    A full position flip (1 -> -1) produces turnover = 2.
    An unchanged position produces turnover = 0.
    """
    lagged = positions.shift(1).fillna(0)
    return lagged.diff().abs().fillna(0)


def apply_transaction_costs(
    gross_returns: pd.Series,
    positions: pd.Series,
    cost_bps: float = 5.0,
) -> pd.Series:
    """
    Subtract transaction costs from gross strategy returns.

    Parameters
    ----------
    gross_returns: Daily gross strategy returns (position * realized return).
    positions:     Raw signal positions before lagging (+1, 0, -1).
    cost_bps:      Round-trip cost assumption in basis points per unit turnover.

    Returns
    -------
    Net returns after costs.
    """
    turnover = compute_turnover(positions.shift(1).fillna(0))
    costs = turnover * cost_per_trade(cost_bps)
    return gross_returns - costs


def cost_sensitivity(
    gross_returns: pd.Series,
    positions: pd.Series,
    cost_range_bps: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compute net Sharpe ratio across a range of cost assumptions.

    Useful for robustness checks — if the strategy only survives at 1 bps,
    it is not production-viable.

    Parameters
    ----------
    cost_range_bps: List of cost assumptions in basis points.
                    Defaults to [0, 2, 5, 10, 20].

    Returns
    -------
    DataFrame with columns: cost_bps, ann_return, sharpe_net, net_total_return.
    """
    if cost_range_bps is None:
        cost_range_bps = [0, 2, 5, 10, 20]

    rows = []
    for bps in cost_range_bps:
        net = apply_transaction_costs(gross_returns, positions, cost_bps=bps)
        ann_ret = net.mean() * 252
        std = net.std() * np.sqrt(252)
        sharpe = ann_ret / std if std > 0 else np.nan
        rows.append({
            "cost_bps":        bps,
            "ann_return":      round(ann_ret, 4),
            "sharpe_net":      round(sharpe, 3),
            "net_total_return": round((1 + net).prod() - 1, 4),
        })
    return pd.DataFrame(rows)


def avg_holding_period(positions: pd.Series) -> float:
    """
    Estimate average holding period in days from turnover.

    avg_holding_period = 1 / avg_daily_turnover
    (approximate; assumes symmetric position changes)
    """
    to = compute_turnover(positions)
    avg_to = to[to > 0].mean()
    return 1.0 / avg_to if avg_to > 0 else np.inf
