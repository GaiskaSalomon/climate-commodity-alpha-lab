import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    return returns.dropna().quantile(1 - confidence)


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    var = historical_var(returns, confidence)
    tail = returns[returns <= var]
    return tail.mean() if not tail.empty else np.nan
