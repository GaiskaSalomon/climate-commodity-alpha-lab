import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rank_ic(predictions: pd.Series, realized: pd.Series) -> float:
    valid = pd.concat([predictions, realized], axis=1).dropna()
    if valid.empty:
        return np.nan
    return spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic


def ic_series(
    df: pd.DataFrame,
    pred_col: str = "prediction",
    actual_col: str = "actual",
    date_col: str = "date",
    freq: str = "ME",
) -> pd.Series:
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.set_index(date_col)

    monthly_ic = result.resample(freq).apply(
        lambda g: rank_ic(g[pred_col], g[actual_col])
    )
    return monthly_ic
