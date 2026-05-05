import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def add_volatility_regime(
    df: pd.DataFrame,
    vol_col: str = "volatility_20d",
    group_col: str = "ticker",
    n_regimes: int = 3,
) -> pd.DataFrame:
    result = df.copy()

    def assign_regime(x):
        valid = x.dropna()
        if len(valid) < n_regimes * 10:
            return pd.Series(np.nan, index=x.index)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(valid.values.reshape(-1, 1))
        km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        centers = km.cluster_centers_.flatten()
        order = np.argsort(centers)
        rank_map = {old: new for new, old in enumerate(order)}
        ranked = np.array([rank_map[l] for l in labels])
        out = pd.Series(np.nan, index=x.index)
        out[valid.index] = ranked
        return out

    result["volatility_regime"] = result.groupby(group_col)[vol_col].transform(assign_regime)

    result["high_vol_regime"] = (result["volatility_regime"] == n_regimes - 1).astype(int)
    result["low_vol_regime"] = (result["volatility_regime"] == 0).astype(int)

    return result


def add_climate_stress_regime(
    df: pd.DataFrame,
    temp_col: str = "temp_anomaly_30d",
    precip_col: str = "precip_anomaly_30d",
    hurricane_col: str = "hurricane_risk_score",
) -> pd.DataFrame:
    result = df.copy()

    cols = [c for c in [temp_col, precip_col, hurricane_col] if c in result.columns]
    if not cols:
        return result

    result["climate_stress_index"] = result[cols].apply(
        lambda row: np.nanmean(np.abs(row.values)), axis=1
    )

    q33 = result["climate_stress_index"].quantile(0.33)
    q66 = result["climate_stress_index"].quantile(0.66)

    result["climate_regime"] = pd.cut(
        result["climate_stress_index"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["low_stress", "moderate_stress", "high_stress"],
    )

    return result
