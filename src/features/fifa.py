"""M9 — FIFA-rating derivatives.

`fifa_potential_gap` (potential − overall) is a growth-headroom signal for young players
(D-17). `delta_fifa_rating` (overall − last-season overall) is a development-speed signal,
reusing `fifa_rating_lag1` produced by the lag module (M6 runs first). FC25 has no
`potential` → `fifa_potential_gap` is NaN for 2024-25 by construction.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def add_fifa_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add fifa_potential_gap and delta_fifa_rating (needs M6's fifa_rating_lag1)."""
    df = df.copy()
    df["fifa_potential_gap"] = df["fifa_potential"] - df["fifa_rating"]

    if "fifa_rating_lag1" in df.columns:
        df["delta_fifa_rating"] = df["fifa_rating"] - df["fifa_rating_lag1"]
    else:  # lag module must run first; fail loud rather than silently skip
        raise KeyError("fifa.add_fifa_features requires fifa_rating_lag1 (run add_lag_features first)")

    logger.info("fifa: added 2 columns")
    return df
