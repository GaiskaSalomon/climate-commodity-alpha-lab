import pandas as pd


def align_to_trading_calendar(
    df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    date_col: str = "date",
    method: str = "ffill",
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df = df.reindex(trading_dates, method=method)
    df.index.name = date_col
    return df.reset_index()


def merge_on_date(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: list[str],
    how: str = "left",
) -> pd.DataFrame:
    for df in [left, right]:
        for col in on:
            if col == "date":
                df[col] = pd.to_datetime(df[col])
    return pd.merge(left, right, on=on, how=how)
