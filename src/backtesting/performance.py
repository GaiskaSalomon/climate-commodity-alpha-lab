import numpy as np
import pandas as pd
from src.models.evaluation import sharpe_ratio, sortino_ratio, max_drawdown, information_coefficient, hit_rate


def compute_strategy_returns(
    df: pd.DataFrame,
    prediction_col: str = "prediction",
    realized_return_col: str = "actual",
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    result = df.copy()

    result["position"] = np.where(result[prediction_col] > 0, 1, np.where(result[prediction_col] < 0, -1, 0))
    result["position_lagged"] = result["position"].shift(1).fillna(0)

    result["turnover"] = result["position_lagged"].diff().abs().fillna(0)
    result["transaction_cost"] = result["turnover"] * (cost_bps / 10_000)

    result["gross_return"] = result["position_lagged"] * result[realized_return_col]
    result["net_return"] = result["gross_return"] - result["transaction_cost"]
    result["equity_curve"] = (1 + result["net_return"]).cumprod()

    return result


def summarize_performance(
    result: pd.DataFrame,
    periods_per_year: int = 252,
    cost_bps: float = 5.0,
) -> dict:
    net = result["net_return"]
    gross = result["gross_return"]
    ec = result["equity_curve"]

    return {
        "sharpe_gross": sharpe_ratio(gross, periods_per_year),
        "sharpe_net": sharpe_ratio(net, periods_per_year),
        "sortino_net": sortino_ratio(net, periods_per_year),
        "max_drawdown": max_drawdown(ec),
        "total_net_return": ec.iloc[-1] - 1 if not ec.empty else np.nan,
        "avg_annual_net_return": net.mean() * periods_per_year,
        "hit_rate": hit_rate(result["prediction"], result["actual"]),
        "ic": information_coefficient(result["prediction"], result["actual"]),
        "avg_turnover": result["turnover"].mean(),
        "n_observations": len(result),
        "cost_bps": cost_bps,
    }
