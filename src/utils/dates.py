import pandas as pd


def get_trading_dates(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end)


def lag_safe(df: pd.DataFrame, col: str, n: int, group_col: str | None = None) -> pd.Series:
    if group_col:
        return df.groupby(group_col)[col].shift(n)
    return df[col].shift(n)
