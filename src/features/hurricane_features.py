"""
Hurricane risk features for commodity return forecasting.

Uses QC-filtered IBTrACS track data with ERA5 atmospheric covariates,
SST, OHC, and z26 (26°C isotherm depth) interpolated along the track.
"""
from __future__ import annotations

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


REFERENCE_POINTS = [
    {"region": "gulf_of_mexico",  "lat": 25.0,  "lon": -90.0},
    {"region": "us_gulf_coast",   "lat": 29.5,  "lon": -90.5},
    {"region": "us_corn_belt",    "lat": 41.5,  "lon": -93.0},
    {"region": "us_wheat_plains", "lat": 38.0,  "lon": -99.0},
]

SAFFIR_SIMPSON = {
    "TD": 0, "TS": 1, "HU": 2, "TY": 2,
    "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5,
}


def build_hurricane_risk_features(
    storm_df: pd.DataFrame,
    reference_points: list[dict] | None = None,
    proximity_threshold_km: float = 500.0,
    basins: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build daily hurricane risk features for each commodity-relevant region.

    Uses inverse-distance-weighted wind intensity as primary risk score.
    Also aggregates OHC and wind shear from the track when available (thesis data).

    Parameters
    ----------
    storm_df:
        IBTrACS QC track data. Expected columns: date, lat, lon, vmax.
        Optional: ohc, shear_200_850, sst, z26, usa_status.
    reference_points:
        List of dicts with 'region', 'lat', 'lon'.
    proximity_threshold_km:
        Distance threshold to flag hurricane as active near a region.
    basins:
        Filter to specific basins (e.g. ['NA', 'EP']). None = all.
    """
    if reference_points is None:
        reference_points = REFERENCE_POINTS

    df = storm_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if basins is not None:
        df = df[df["basin"].isin(basins)]

    ref_df = pd.DataFrame(reference_points)
    has_ohc = "ohc" in df.columns
    has_shear = "shear_200_850" in df.columns
    has_sst = "sst" in df.columns

    records = []
    for _, storm in df.iterrows():
        for _, point in ref_df.iterrows():
            distance = haversine_distance_km(
                storm["lat"], storm["lon"],
                point["lat"], point["lon"],
            )
            rec = {
                "date": storm["date"],
                "region": point["region"],
                "distance_km": distance,
                "vmax": storm["vmax"],
                "risk_score": storm["vmax"] / (1 + distance),
            }
            if has_ohc:
                rec["ohc"] = storm["ohc"]
            if has_shear:
                rec["shear_200_850"] = storm["shear_200_850"]
            if has_sst:
                rec["sst"] = storm["sst"]
            records.append(rec)

    risk_df = pd.DataFrame(records)

    agg = {"risk_score": "max", "distance_km": "min", "vmax": "max"}
    if has_ohc:
        agg["ohc"] = "mean"
    if has_shear:
        agg["shear_200_850"] = "mean"
    if has_sst:
        agg["sst"] = "mean"

    daily = (
        risk_df.groupby(["date", "region"])
        .agg(**{k: (k, v) for k, v in agg.items()})
        .reset_index()
        .rename(columns={
            "risk_score": "hurricane_risk_score",
            "distance_km": "min_storm_distance_km",
            "vmax": "max_storm_wind_kt",
        })
    )

    daily["hurricane_active"] = (
        daily["min_storm_distance_km"] <= proximity_threshold_km
    ).astype(int)

    # Intensity-weighted proximity: high OHC + close storm = amplified risk
    if has_ohc:
        ohc_norm = daily["ohc"] / (daily["ohc"].max() + 1e-9)
        daily["ohc_weighted_risk"] = daily["hurricane_risk_score"] * (1 + ohc_norm)

    # Gulf energy disruption proxy (energy regions only)
    energy_mask = daily["region"].isin(["gulf_of_mexico", "us_gulf_coast"])
    daily["gulf_disruption_proxy"] = (
        daily["hurricane_risk_score"] * daily["hurricane_active"] * energy_mask.astype(int)
    )

    return daily.sort_values(["region", "date"]).reset_index(drop=True)


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
        result[f"hurricane_risk_lag{lag}d"] = (
            result.groupby(group_col)["hurricane_risk_score"].shift(lag)
        )

    result["storm_active_7d"] = (
        result.groupby(group_col)["hurricane_active"]
        .transform(lambda x: x.rolling(7, min_periods=1).sum())
    )

    return result


def hurricane_season_flag(dates: pd.Series) -> pd.Series:
    """Returns 1 during Atlantic hurricane season (June–November)."""
    return dates.dt.month.between(6, 11).astype(int)
