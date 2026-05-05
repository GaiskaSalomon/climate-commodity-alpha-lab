import pandas as pd
import numpy as np
from src.models.evaluation import sharpe_ratio, max_drawdown, information_coefficient, hit_rate


def test_sharpe_positive_returns():
    r = pd.Series([0.001] * 252)
    assert sharpe_ratio(r) > 0


def test_sharpe_zero_vol():
    r = pd.Series([0.0] * 100)
    assert np.isnan(sharpe_ratio(r))


def test_max_drawdown_monotone_growth():
    ec = pd.Series((1.001 ** np.arange(100)).cumprod())
    assert max_drawdown(ec) == 0.0


def test_ic_perfect_prediction():
    pred = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    real = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    assert abs(information_coefficient(pred, real) - 1.0) < 1e-9


def test_hit_rate_all_correct():
    pred = pd.Series([1.0, -1.0, 1.0])
    real = pd.Series([0.5, -0.3, 0.2])
    assert hit_rate(pred, real) == 1.0
