from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = ROOT_DIR / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_INTERIM_DIR = ROOT_DIR / "data/interim"
DATA_PROCESSED_DIR = ROOT_DIR / os.getenv("DATA_PROCESSED_DIR", "data/processed")
REPORTS_DIR = ROOT_DIR / os.getenv("REPORTS_DIR", "reports")
FIGURES_DIR = REPORTS_DIR / "figures"

NOAA_API_KEY = os.getenv("NOAA_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

ASSETS = {
    "UNG": "natural_gas",
    "USO": "crude_oil",
    "DBA": "agriculture_broad",
    "CORN": "corn",
    "WEAT": "wheat",
    "SOYB": "soybeans",
    "GLD": "gold",
}

REFERENCE_REGIONS = [
    {"region": "gulf_of_mexico", "lat": 25.0, "lon": -90.0},
    {"region": "us_corn_belt", "lat": 41.5, "lon": -93.0},
    {"region": "us_wheat_plains", "lat": 38.0, "lon": -99.0},
    {"region": "us_gulf_coast", "lat": 29.5, "lon": -90.5},
]

TRANSACTION_COST_BPS = 5.0
PERIODS_PER_YEAR = 252
