import pandas as pd
import numpy as np


def add_forward_returns(
    df: pd.DataFrame,
    price_col: str = "close",
    group_col: str = "ticker",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    if horizons is None:
        horizons = [1, 5, 10, 20]

    result = df.sort_values([group_col, "date"]).copy()

    for h in horizons:
        result[f"future_return_{h}d"] = (
            result.groupby(group_col)[price_col]
            .transform(lambda x: x.shift(-h) / x - 1)
        )
        result[f"future_direction_{h}d"] = (
            result[f"future_return_{h}d"] > 0
        ).astype(float)

    return result


def add_cross_sectional_rank(
    df: pd.DataFrame,
    target_col: str = "future_return_5d",
    date_col: str = "date",
) -> pd.DataFrame:
    result = df.copy()
    result["cs_rank"] = (
        result.groupby(date_col)[target_col]
        .rank(pct=True, ascending=True)
    )
    return result
