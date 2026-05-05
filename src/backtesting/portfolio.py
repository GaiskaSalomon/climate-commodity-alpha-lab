import pandas as pd
import numpy as np


def equal_weight_portfolio(
    signals: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "ticker",
    position_col: str = "position",
    return_col: str = "actual",
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Aggregate individual asset signals into an equal-weight portfolio.
    """
    result = signals.copy()
    result["position_lagged"] = result.groupby(asset_col)[position_col].shift(1).fillna(0)
    result["turnover"] = result.groupby(asset_col)["position_lagged"].diff().abs().fillna(0)
    result["transaction_cost"] = result["turnover"] * (cost_bps / 10_000)
    result["asset_net_return"] = result["position_lagged"] * result[return_col] - result["transaction_cost"]

    portfolio = (
        result.groupby(date_col)["asset_net_return"]
        .mean()
        .reset_index()
        .rename(columns={"asset_net_return": "portfolio_return"})
    )

    portfolio["equity_curve"] = (1 + portfolio["portfolio_return"]).cumprod()
    return portfolio
