import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if returns.std() == 0:
        return np.nan
    return np.sqrt(periods_per_year) * returns.mean() / returns.std()


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    downside = returns[returns < 0]
    if downside.std() == 0:
        return np.nan
    return np.sqrt(periods_per_year) * returns.mean() / downside.std()


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    return (equity_curve / running_max - 1).min()


def information_coefficient(predictions: pd.Series, realized: pd.Series) -> float:
    valid = pd.concat([predictions, realized], axis=1).dropna()
    if valid.empty:
        return np.nan
    return spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic


def hit_rate(predictions: pd.Series, realized: pd.Series) -> float:
    valid = pd.concat([predictions, realized], axis=1).dropna()
    return (np.sign(valid.iloc[:, 0]) == np.sign(valid.iloc[:, 1])).mean()


def compute_all_metrics(results_df: pd.DataFrame) -> dict:
    preds = results_df["prediction"]
    actual = results_df["actual"]

    return {
        "ic": information_coefficient(preds, actual),
        "hit_rate": hit_rate(preds, actual),
    }
