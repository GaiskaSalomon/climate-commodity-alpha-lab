import pandas as pd
import numpy as np


def haversine_distance_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return radius * 2 * np.arcsin(np.sqrt(a))


def build_hurricane_risk_features(
    storm_df: pd.DataFrame,
    reference_points: list[dict],
    date_col: str = "date",
    storm_lat_col: str = "lat",
    storm_lon_col: str = "lon",
    wind_col: str = "max_wind",
    proximity_threshold_km: float = 500.0,
) -> pd.DataFrame:
    """
    Build hurricane risk features around commodity-relevant infrastructure regions.

    Uses inverse-distance-weighted wind intensity as risk score, which captures
    both storm severity and proximity to energy/agricultural production areas.
    """
    storm_df = storm_df.copy()
    storm_df[date_col] = pd.to_datetime(storm_df[date_col])

    ref_df = pd.DataFrame(reference_points)
    records = []

    for _, storm in storm_df.iterrows():
        for _, point in ref_df.iterrows():
            distance = haversine_distance_km(
                storm[storm_lat_col],
                storm[storm_lon_col],
                point["lat"],
                point["lon"],
            )
            records.append({
                "date": storm[date_col],
                "region": point["region"],
                "distance_km": distance,
                "max_wind": storm[wind_col],
                "risk_score": storm[wind_col] / (1 + distance),
            })

    risk_df = pd.DataFrame(records)

    daily_risk = (
        risk_df.groupby(["date", "region"])
        .agg(
            hurricane_risk_score=("risk_score", "max"),
            min_storm_distance_km=("distance_km", "min"),
            max_storm_wind=("max_wind", "max"),
        )
        .reset_index()
    )

    daily_risk["hurricane_active_near_region"] = (
        daily_risk["min_storm_distance_km"] <= proximity_threshold_km
    ).astype(int)

    daily_risk["gulf_disruption_proxy"] = (
        daily_risk["hurricane_risk_score"] *
        daily_risk["hurricane_active_near_region"]
    )

    return daily_risk


def add_lagged_hurricane_features(
    df: pd.DataFrame,
    group_col: str = "region",
    date_col: str = "date",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    if lags is None:
        lags = [1, 3, 5, 7]

    result = df.sort_values([group_col, date_col]).copy()

    for lag in lags:
        result[f"hurricane_risk_score_lag{lag}d"] = (
            result.groupby(group_col)["hurricane_risk_score"].shift(lag)
        )

    result["storm_count_7d"] = (
        result.groupby(group_col)["hurricane_active_near_region"]
        .transform(lambda x: x.rolling(7).sum())
    )

    return result
