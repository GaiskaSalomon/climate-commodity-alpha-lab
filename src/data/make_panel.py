import pandas as pd


def build_research_panel(
    prices: pd.DataFrame,
    weather_features: pd.DataFrame,
    hurricane_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build aligned commodity research panel.
    One row per date-asset pair.
    """
    prices = prices.copy()
    weather_features = weather_features.copy()

    prices["date"] = pd.to_datetime(prices["date"])
    weather_features["date"] = pd.to_datetime(weather_features["date"])

    panel = prices.merge(weather_features, on=["date", "region"], how="left")

    if hurricane_features is not None:
        hurricane_features = hurricane_features.copy()
        hurricane_features["date"] = pd.to_datetime(hurricane_features["date"])
        panel = panel.merge(hurricane_features, on=["date", "region"], how="left")

    return panel.sort_values(["asset", "date"]).reset_index(drop=True)
