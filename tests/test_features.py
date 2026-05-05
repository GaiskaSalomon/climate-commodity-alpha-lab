import pandas as pd
import numpy as np
import pytest
from src.features.price_features import add_price_features
from src.features.weather_features import add_weather_anomalies
from src.features.target_builder import add_forward_returns
from src.features.hurricane_features import haversine_distance_km


def make_price_df(n=100):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(42)
    prices = 100 * (1 + np.random.randn(n) * 0.01).cumprod()
    return pd.DataFrame({"date": dates, "close": prices, "ticker": "TEST"})


def test_price_features_shape():
    df = make_price_df()
    result = add_price_features(df)
    assert "return_1d" in result.columns
    assert "volatility_20d" in result.columns
    assert len(result) == len(df)


def test_forward_returns_no_look_ahead():
    df = make_price_df(60)
    result = add_forward_returns(df, horizons=[5])
    last_5 = result["future_return_5d"].iloc[-5:]
    assert last_5.isna().all(), "Last 5 rows should have NaN future returns (no lookahead data)"


def test_weather_anomalies():
    n = 365 * 3
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "temperature": 20 + 10 * np.sin(np.linspace(0, 6 * np.pi, n)) + np.random.randn(n),
        "precipitation": np.abs(np.random.randn(n) * 5),
    })
    result = add_weather_anomalies(df)
    assert "temperature_anomaly" in result.columns
    assert "precipitation_anomaly" in result.columns
    assert result["temperature_anomaly"].notna().sum() > 0


def test_haversine_distance():
    # New York to Los Angeles: ~3940 km
    dist = haversine_distance_km(40.71, -74.01, 34.05, -118.24)
    assert 3800 < dist < 4100
