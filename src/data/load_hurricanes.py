import pandas as pd
from src.config import DATA_RAW_DIR


IBTRACS_COLS = {
    "SID": "storm_id",
    "NAME": "name",
    "ISO_TIME": "datetime",
    "LAT": "lat",
    "LON": "lon",
    "WMO_WIND": "max_wind",
    "WMO_PRES": "min_pressure",
    "BASIN": "basin",
    "SUBBASIN": "subbasin",
    "NATURE": "nature",
    "USA_SSHS": "saffir_simpson_cat",
}


def load_ibtracs(path: str | None = None, basins: list[str] | None = None) -> pd.DataFrame:
    """
    Load IBTrACS tropical cyclone track data.

    Download from: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/
    Recommended file: ibtracs.NA.list.v04r00.csv  (North Atlantic basin)
    """
    if path is None:
        path = DATA_RAW_DIR / "hurricanes" / "ibtracs.NA.list.v04r00.csv"

    df = pd.read_csv(path, skiprows=[1], low_memory=False)
    df = df.rename(columns={k: v for k, v in IBTRACS_COLS.items() if k in df.columns})

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["date"] = df["datetime"].dt.normalize()

    for col in ["lat", "lon", "max_wind"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if basins is not None:
        df = df[df["basin"].isin(basins)]

    return df.dropna(subset=["lat", "lon", "max_wind"]).reset_index(drop=True)
