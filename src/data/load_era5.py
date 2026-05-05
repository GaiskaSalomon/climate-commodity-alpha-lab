"""
ERA5 reanalysis data download and processing via CDS API.

Variables downloaded:
- 2m temperature (t2m)
- Total precipitation (tp)

Regions correspond to commodity-relevant infrastructure and production areas.
"""
from __future__ import annotations

import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import DATA_RAW_DIR, REFERENCE_REGIONS


REGION_BOXES = {
    "gulf_of_mexico": {
        "north": 30.0, "west": -97.0, "south": 20.0, "east": -83.0,
        "lat": 25.0, "lon": -90.0,
    },
    "us_corn_belt": {
        "north": 46.0, "west": -100.0, "south": 37.0, "east": -83.0,
        "lat": 41.5, "lon": -93.0,
    },
    "us_wheat_plains": {
        "north": 43.0, "west": -105.0, "south": 33.0, "east": -93.0,
        "lat": 38.0, "lon": -99.0,
    },
    "us_gulf_coast": {
        "north": 31.0, "west": -95.0, "south": 27.0, "east": -87.0,
        "lat": 29.5, "lon": -90.5,
    },
}


def download_era5_daily(
    region_name: str,
    years: list[int],
    output_dir: Path | None = None,
    variables: list[str] | None = None,
) -> Path:
    """
    Download ERA5 daily aggregated data for a single region box.

    Uses ERA5 hourly data on single levels, aggregated to daily by CDS.
    Returns path to the downloaded NetCDF file.
    """
    if output_dir is None:
        output_dir = DATA_RAW_DIR / "weather" / "era5"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if variables is None:
        variables = ["2m_temperature", "total_precipitation"]

    box = REGION_BOXES[region_name]
    area = [box["north"], box["west"], box["south"], box["east"]]  # N/W/S/E

    year_str = [str(y) for y in years]
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]

    out_path = output_dir / f"era5_{region_name}_{'_'.join(year_str)}.nc"

    if out_path.exists():
        print(f"[ERA5] Already exists: {out_path.name} — skipping download.")
        return out_path

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": variables,
            "year": year_str,
            "month": months,
            "day": days,
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": area,
            "format": "netcdf",
        },
        str(out_path),
    )
    print(f"[ERA5] Downloaded: {out_path.name}")
    return out_path


def _open_nc_from_zip_or_file(path: Path) -> xr.Dataset:
    """
    Open an xarray Dataset from either a plain NetCDF or a ZIP containing NetCDFs.
    The new CDS API (post-2024) delivers data as ZIP archives with multiple NetCDF files.
    We extract to a temp directory and merge instant (t2m) and accumulated (tp) streams.
    """
    import zipfile
    import tempfile

    if not zipfile.is_zipfile(path):
        return xr.open_dataset(path, engine="netcdf4")

    datasets = []
    # Extract to a persistent sibling directory to avoid temp cleanup issues
    extract_dir = path.parent / (path.stem + "_extracted")
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".nc"):
                zf.extract(name, extract_dir)
                nc_file = extract_dir / name
                ds = xr.open_dataset(nc_file, engine="netcdf4")
                datasets.append(ds)

    if not datasets:
        raise ValueError(f"No NetCDF files found inside {path}")

    if len(datasets) == 1:
        return datasets[0]

    # Merge instant (t2m) + accumulated (tp) streams on shared coordinates
    return xr.merge(datasets, compat="override")


def nc_to_daily_dataframe(
    nc_path: Path,
    region_name: str,
) -> pd.DataFrame:
    """
    Open ERA5 file (NetCDF or ZIP), spatially average over the region box,
    aggregate to daily, and return a tidy DataFrame:
    columns: date, region, temperature (°C), precipitation (mm/day).
    """
    ds = _open_nc_from_zip_or_file(Path(nc_path))

    # Detect spatial dimension names (latitude/longitude or lat/lon)
    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"
    spatial_dims = [d for d in [lat_dim, lon_dim] if d in ds.dims]

    ds_mean = ds.mean(dim=spatial_dims) if spatial_dims else ds

    # Detect time dimension
    time_dim = next((d for d in ["valid_time", "time"] if d in ds_mean.dims), None)
    if time_dim is None:
        raise ValueError(f"No time dimension found in {nc_path}. Dims: {list(ds_mean.dims)}")

    ds_daily = ds_mean.resample({time_dim: "1D"}).mean()

    df = ds_daily.to_dataframe().reset_index()
    df = df.rename(columns={time_dim: "date"})

    rename_map = {}
    if "t2m" in df.columns:
        rename_map["t2m"] = "temperature"
    if "tp" in df.columns:
        rename_map["tp"] = "precipitation"

    df = df.rename(columns=rename_map)

    if "temperature" in df.columns:
        df["temperature"] = df["temperature"] - 273.15  # K → °C

    if "precipitation" in df.columns:
        df["precipitation"] = df["precipitation"] * 1000  # m → mm/day

    df["region"] = region_name
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    keep = ["date", "region", "temperature", "precipitation"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].sort_values("date").reset_index(drop=True)


def load_all_era5_regions(
    nc_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load all downloaded ERA5 NetCDF files and concatenate into a single panel.
    """
    if nc_dir is None:
        nc_dir = DATA_RAW_DIR / "weather" / "era5"

    frames = []
    for region_name in REGION_BOXES:
        pattern = list(Path(nc_dir).glob(f"era5_{region_name}_*.nc"))
        if not pattern:
            continue
        dfs = [nc_to_daily_dataframe(p, region_name) for p in sorted(pattern)]
        frames.append(pd.concat(dfs, ignore_index=True))

    if not frames:
        raise FileNotFoundError(f"No ERA5 NetCDF files found in {nc_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["region", "date"]).reset_index(drop=True)
    return combined
