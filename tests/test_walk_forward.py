import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from src.models.walk_forward import walk_forward_validation


def make_synthetic_panel(n=500):
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    np.random.seed(0)
    X = np.random.randn(n, 3)
    y = X[:, 0] * 0.3 + X[:, 1] * 0.1 + np.random.randn(n) * 0.5
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["date"] = dates
    df["target"] = y
    return df


def test_walk_forward_no_future_data():
    df = make_synthetic_panel()
    results = walk_forward_validation(
        df,
        features=["f1", "f2", "f3"],
        target="target",
        model=Ridge(),
        train_window=200,
        test_window=21,
        step_size=21,
    )
    assert "prediction" in results.columns
    assert "actual" in results.columns

    # Ensure test dates are always strictly after training end
    assert (results["date"] > results["fold_train_end"]).all()


def test_walk_forward_returns_dataframe():
    df = make_synthetic_panel()
    out = walk_forward_validation(df, ["f1", "f2", "f3"], "target", Ridge())
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
