import pandas as pd


def add_weather_anomalies(
    df: pd.DataFrame,
    date_col: str = "date",
    group_cols: list[str] | None = None,
    temp_col: str = "temperature",
    precip_col: str = "precipitation",
) -> pd.DataFrame:
    """
    Add weather anomaly features using day-of-year climatology.

    Transforms raw weather variables into tradable signals:
    unusually hot, cold, dry or wet periods relative to seasonal norms.
    """
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result["day_of_year"] = result[date_col].dt.dayofyear

    if group_cols is None:
        group_cols = []

    climatology_cols = group_cols + ["day_of_year"]

    result["temp_climatology"] = (
        result.groupby(climatology_cols)[temp_col].transform("mean")
    )
    result["precip_climatology"] = (
        result.groupby(climatology_cols)[precip_col].transform("mean")
    )

    result["temperature_anomaly"] = result[temp_col] - result["temp_climatology"]
    result["precipitation_anomaly"] = result[precip_col] - result["precip_climatology"]

    return result


def add_rolling_weather_features(
    df: pd.DataFrame,
    date_col: str = "date",
    group_col: str = "region",
    temp_col: str = "temperature_anomaly",
    precip_col: str = "precipitation_anomaly",
    temp_threshold_high: float = 35.0,
    temp_threshold_low: float = 0.0,
) -> pd.DataFrame:
    result = df.sort_values([group_col, date_col]).copy()

    for window in [7, 14, 30, 60]:
        result[f"temp_anomaly_{window}d"] = (
            result.groupby(group_col)[temp_col]
            .transform(lambda x: x.rolling(window).mean())
        )
        result[f"precip_anomaly_{window}d"] = (
            result.groupby(group_col)[precip_col]
            .transform(lambda x: x.rolling(window).sum())
        )

    if "temperature" in result.columns:
        result["heat_stress_days_30d"] = (
            result.groupby(group_col)["temperature"]
            .transform(lambda x: (x > temp_threshold_high).rolling(30).sum())
        )
        result["cold_stress_days_30d"] = (
            result.groupby(group_col)["temperature"]
            .transform(lambda x: (x < temp_threshold_low).rolling(30).sum())
        )

    result["drought_proxy_60d"] = (
        result.groupby(group_col)[precip_col]
        .transform(lambda x: x.rolling(60).sum())
    )

    result["extreme_weather_flag"] = (
        (result["temperature_anomaly"].abs() > 2 * result["temperature_anomaly"].std()) |
        (result["precipitation_anomaly"].abs() > 2 * result["precipitation_anomaly"].std())
    ).astype(int)

    return result
