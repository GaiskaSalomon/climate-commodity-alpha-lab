import pandas as pd
import yfinance as yf
from src.config import ASSETS, DATA_RAW_DIR


def download_commodity_prices(
    tickers: list[str] | None = None,
    start: str = "2005-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    if tickers is None:
        tickers = list(ASSETS.keys())

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
    close = raw["Close"].reset_index()
    long = close.melt(id_vars="Date", var_name="ticker", value_name="close")
    long = long.rename(columns={"Date": "date"})
    long["asset"] = long["ticker"].map(ASSETS)
    return long.dropna(subset=["close"]).sort_values(["ticker", "date"])


def load_prices_from_file(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = DATA_RAW_DIR / "prices" / "commodity_prices.parquet"
    return pd.read_parquet(path)
