import pandas as pd
import numpy as np
from src.backtesting.performance import compute_strategy_returns, summarize_performance
from src.backtesting.signal_to_position import signal_to_position


def make_backtest_df(n=200):
    np.random.seed(1)
    return pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=n, freq="B"),
        "prediction": np.random.randn(n),
        "actual": np.random.randn(n) * 0.01,
    })


def test_strategy_returns_columns():
    df = make_backtest_df()
    result = compute_strategy_returns(df)
    for col in ["position", "position_lagged", "gross_return", "net_return", "equity_curve"]:
        assert col in result.columns


def test_equity_curve_starts_near_one():
    df = make_backtest_df()
    result = compute_strategy_returns(df)
    assert 0.9 < result["equity_curve"].iloc[1] < 1.1


def test_signal_to_position_values():
    pred = pd.Series([0.5, -0.5, 0.0, 1.0, -1.0])
    pos = signal_to_position(pred, threshold=0.0)
    assert set(pos.unique()).issubset({-1, 0, 1})


def test_summarize_performance_keys():
    df = make_backtest_df()
    result = compute_strategy_returns(df)
    summary = summarize_performance(result)
    for key in ["sharpe_net", "max_drawdown", "hit_rate", "ic"]:
        assert key in summary
