"""
Loaders for pre-processed datasets from the PhD thesis on tropical cyclone poleward migration.

IBTrACS QC (74,792 track points, 1980-2025) and IBTrACS with atmospheric/oceanic
covariates (60,234 points with ERA5 track-interpolated variables + OHC/z26/SST).
Climate indices (AMO, ONI, PDO with lags, monthly 1980-2025).
SHIPS intensity database (AL and EP basins, 1982-2023).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

THESIS_DATA = Path(r"D:\Avances Doctorado\data")

IBTRACS_QC = THESIS_DATA / "results/processed/ibtracs/ibtracs_qc_1980_2025.parquet"
IBTRACS_OHC = THESIS_DATA / "results/processed/ibtracs/ibtracs_qc_1980_2025_with_ohc.parquet"
CLIMATE_INDICES = THESIS_DATA / "results/processed/covariates/climate_indices.parquet"
SHIPS_DB = THESIS_DATA / "external/ships/ships_dev_database.csv"


def load_ibtracs(with_ohc: bool = True) -> pd.DataFrame:
    """
    Load QC-filtered IBTrACS track data (1980-2025).

    with_ohc=True returns 60,234 rows with ERA5 atmospheric covariates,
    SST, OHC, and z26 interpolated along the track.
    with_ohc=False returns 74,792 rows with basic position/intensity only.
    """
    path = IBTRACS_OHC if with_ohc else IBTRACS_QC
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.normalize()
    return df


def load_climate_indices() -> pd.DataFrame:
    """
    Monthly AMO, ONI, PDO with 1- and 3-month lags (1980-2025).
    Returns a daily-indexed DataFrame for easy merging via forward-fill.
    """
    df = pd.read_parquet(CLIMATE_INDICES)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    return df.sort_values("date").reset_index(drop=True)


def load_ships() -> pd.DataFrame:
    """
    SHIPS intensity database (AL and EP, 1982-2023).
    VMXI: maximum intensity (kt). I24H: 24-hour intensity change (kt).
    """
    df = pd.read_csv(SHIPS_DB)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    df["date"] = df["DATE_TIME"].dt.normalize()
    df = df.rename(columns={"ATCF_ID": "storm_id", "BASIN": "basin",
                             "VMXI": "vmxi", "I24H": "i24h"})
    return df.sort_values(["storm_id", "DATE_TIME"]).reset_index(drop=True)


def climate_indices_daily(
    start: str = "2005-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Forward-fill monthly climate indices to daily frequency for panel merging.
    """
    monthly = load_climate_indices()
    monthly = monthly[monthly["date"] >= start]
    daily_idx = pd.date_range(start=start, end=end, freq="D")
    daily = (
        monthly.set_index("date")[["amo", "oni", "pdo"]]
        .reindex(daily_idx, method="ffill")
    )
    daily.index.name = "date"
    return daily.reset_index()
