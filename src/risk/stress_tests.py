import pandas as pd
import numpy as np


def performance_by_regime(
    results: pd.DataFrame,
    regime_col: str,
    return_col: str = "net_return",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Break down strategy performance across climate or volatility regimes.
    """
    rows = []
    for regime, group in results.groupby(regime_col):
        r = group[return_col].dropna()
        rows.append({
            "regime": regime,
            "n_obs": len(r),
            "mean_return_annualized": r.mean() * periods_per_year,
            "sharpe": np.sqrt(periods_per_year) * r.mean() / r.std() if r.std() > 0 else np.nan,
            "hit_rate": (r > 0).mean(),
        })
    return pd.DataFrame(rows)


def hurricane_season_performance(
    results: pd.DataFrame,
    date_col: str = "date",
    return_col: str = "net_return",
) -> pd.DataFrame:
    result = results.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result["month"] = result[date_col].dt.month
    result["hurricane_season"] = result["month"].between(6, 11).astype(int)
    return performance_by_regime(result, "hurricane_season", return_col)
