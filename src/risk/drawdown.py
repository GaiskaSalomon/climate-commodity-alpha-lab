import pandas as pd
import numpy as np


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1


def max_drawdown(equity_curve: pd.Series) -> float:
    return drawdown_series(equity_curve).min()


def drawdown_duration(equity_curve: pd.Series) -> int:
    dd = drawdown_series(equity_curve)
    in_drawdown = dd < 0
    streaks = in_drawdown.ne(in_drawdown.shift()).cumsum()
    return in_drawdown.groupby(streaks).sum().max()
