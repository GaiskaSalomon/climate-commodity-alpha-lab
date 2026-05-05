import pandas as pd
import numpy as np


def add_price_features(
    df: pd.DataFrame,
    price_col: str = "close",
    group_col: str = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    result = df.sort_values([group_col, date_col]).copy()
    g = result.groupby(group_col)[price_col]

    result["return_1d"] = g.pct_change(1)
    result["return_5d"] = g.pct_change(5)
    result["return_20d"] = g.pct_change(20)

    result["volatility_20d"] = g.transform(lambda x: x.pct_change().rolling(20).std())
    result["volatility_60d"] = g.transform(lambda x: x.pct_change().rolling(60).std())

    result["momentum_20d"] = g.transform(lambda x: x / x.shift(20) - 1)
    result["momentum_60d"] = g.transform(lambda x: x / x.shift(60) - 1)

    rolling_max = g.transform(lambda x: x.rolling(60).max())
    result["drawdown_60d"] = result[price_col] / rolling_max - 1

    def trend_strength(x):
        r = x.pct_change()
        return r.rolling(20).mean() / (r.rolling(20).std() + 1e-9)

    result["trend_strength"] = g.transform(trend_strength)

    def rolling_skew(x):
        return x.pct_change().rolling(20).skew()

    def rolling_kurt(x):
        return x.pct_change().rolling(20).kurt()

    result["realized_skewness"] = g.transform(rolling_skew)
    result["realized_kurtosis"] = g.transform(rolling_kurt)

    return result
