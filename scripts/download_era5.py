"""
Download ERA5 reanalysis data for all regions and years.
Run: python scripts/download_era5.py

Downloads 1 year per request (CDS limit). Skips already-downloaded files.
Estimated time: ~3-5 min per year-region pair (~60-100 min total for 2005-2025).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_era5 import download_era5_daily
from src.config import DATA_RAW_DIR

OUT_DIR = DATA_RAW_DIR / "weather" / "era5"
REGIONS = ["gulf_of_mexico", "us_corn_belt", "us_wheat_plains", "us_gulf_coast"]
YEARS = list(range(2005, 2026))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(REGIONS) * len(YEARS)
    done = 0

    for year in YEARS:
        for region in REGIONS:
            done += 1
            print(f"[{done}/{total}] {region} {year}", flush=True)
            try:
                download_era5_daily(region, years=[year], output_dir=OUT_DIR)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)


if __name__ == "__main__":
    main()
