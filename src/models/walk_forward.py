import pandas as pd
from sklearn.base import clone


def walk_forward_validation(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    model,
    date_col: str = "date",
    train_window: int = 756,
    test_window: int = 21,
    step_size: int = 21,
) -> pd.DataFrame:
    """
    Walk-forward validation for time-series commodity forecasting.

    Avoids temporal leakage by ensuring the test set is always strictly
    after the training window. No shuffling, no random splits.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    predictions = []
    start = train_window

    while start + test_window <= len(df):
        train_start = start - train_window
        train = df.iloc[train_start:start]
        test = df.iloc[start : start + test_window]

        X_train = train[features].dropna()
        y_train = train.loc[X_train.index, target]
        X_test = test[features]

        valid_train = y_train.notna()
        if valid_train.sum() < 10:
            start += step_size
            continue

        fitted = clone(model)
        fitted.fit(X_train[valid_train], y_train[valid_train])

        y_pred = fitted.predict(X_test.fillna(0))

        fold = test[[date_col]].copy()
        fold["prediction"] = y_pred
        fold["actual"] = test[target].values
        fold["fold_train_start"] = train[date_col].iloc[0]
        fold["fold_train_end"] = train[date_col].iloc[-1]
        predictions.append(fold)

        start += step_size

    return pd.concat(predictions, ignore_index=True)


def expanding_window_validation(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    model,
    date_col: str = "date",
    min_train: int = 252,
    test_window: int = 21,
    step_size: int = 21,
) -> pd.DataFrame:
    df = df.sort_values(date_col).reset_index(drop=True)
    predictions = []
    start = min_train

    while start + test_window <= len(df):
        train = df.iloc[:start]
        test = df.iloc[start : start + test_window]

        X_train = train[features].fillna(0)
        y_train = train[target].dropna()
        X_test = test[features].fillna(0)

        fitted = clone(model)
        fitted.fit(X_train.loc[y_train.index], y_train)

        fold = test[[date_col]].copy()
        fold["prediction"] = fitted.predict(X_test)
        fold["actual"] = test[target].values
        predictions.append(fold)

        start += step_size

    return pd.concat(predictions, ignore_index=True)
