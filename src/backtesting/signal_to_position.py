import numpy as np
import pandas as pd


def signal_to_position(
    prediction: pd.Series,
    threshold: float = 0.0,
    long_only: bool = False,
) -> pd.Series:
    """
    Convert model forecasts into trading positions.

    +1 = long, -1 = short, 0 = flat.
    threshold controls the dead-band around zero to reduce excessive turnover.
    """
    if long_only:
        return np.where(prediction > threshold, 1, 0)

    return pd.Series(
        np.where(prediction > threshold, 1, np.where(prediction < -threshold, -1, 0)),
        index=prediction.index,
    )


def rank_to_position(
    cs_rank: pd.Series,
    top_pct: float = 0.3,
    bottom_pct: float = 0.3,
) -> pd.Series:
    """
    Convert cross-sectional rank into long/short positions.

    Long top_pct, short bottom_pct, flat the rest.
    """
    return pd.Series(
        np.where(
            cs_rank >= (1 - top_pct),
            1,
            np.where(cs_rank <= bottom_pct, -1, 0),
        ),
        index=cs_rank.index,
    )
