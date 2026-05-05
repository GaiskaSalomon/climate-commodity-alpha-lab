import pandas as pd
import requests
from src.config import NOAA_API_KEY, DATA_RAW_DIR


def load_weather_from_file(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = DATA_RAW_DIR / "weather" / "weather_data.parquet"
    return pd.read_parquet(path)


def fetch_noaa_daily(
    station_id: str,
    start: str,
    end: str,
    datatypes: list[str] | None = None,
) -> pd.DataFrame:
    if datatypes is None:
        datatypes = ["TMAX", "TMIN", "PRCP"]

    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": NOAA_API_KEY}
    records = []

    for datatype in datatypes:
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "datatypeid": datatype,
            "startdate": start,
            "enddate": end,
            "limit": 1000,
            "units": "metric",
        }
        resp = requests.get(base_url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json().get("results", [])
        records.extend(data)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.date
    pivot = df.pivot_table(index="date", columns="datatype", values="value", aggfunc="mean")
    pivot = pivot.reset_index()
    pivot.columns.name = None
    return pivot
