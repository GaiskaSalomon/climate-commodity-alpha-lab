import pandas as pd


def check_no_future_leakage(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str = "date",
) -> bool:
    """
    Sanity check: features must not have higher correlation with the target
    than with its lagged version (a proxy for leakage detection).
    """
    corr_direct = df[feature_cols].corrwith(df[target_col]).abs().max()
    lagged_target = df[target_col].shift(1)
    corr_lagged = df[feature_cols].corrwith(lagged_target).abs().max()

    if corr_direct > corr_lagged * 2:
        print(
            f"Warning: max feature-target correlation ({corr_direct:.3f}) is more than 2x "
            f"the lagged correlation ({corr_lagged:.3f}). Check for leakage."
        )
        return False
    return True


def assert_sorted_by_date(df: pd.DataFrame, date_col: str = "date") -> None:
    dates = pd.to_datetime(df[date_col])
    assert dates.is_monotonic_increasing, "DataFrame must be sorted by date ascending."
