"""M4 — age-curve features.

Age is a top value driver with a position-specific peak (Phase-3 Fig10: GK ~30, outfield
~25-26). Adds polynomial age terms, position-conditional distance-from-peak, and career-stage
bands. `age_at_season_end` (the raw age) is already in the panel and is left untouched.
"""

from __future__ import annotations

import pandas as pd

from src.utils.constants import PEAK_AGE_BY_POSITION
from src.utils.logging import get_logger

logger = get_logger(__name__)


def add_age_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add age², age³, position-conditional distance_from_peak, and is_young/peak/declining."""
    df = df.copy()
    age = df["age_at_season_end"]

    df["age_squared"] = age ** 2
    df["age_cubed"] = age ** 3
    peak = df["primary_position"].map(PEAK_AGE_BY_POSITION)
    df["distance_from_peak"] = (age - peak).abs()

    df["is_young"] = (age < 24).fillna(False).astype(bool)
    df["is_peak"] = age.between(24, 29).fillna(False).astype(bool)
    df["is_declining"] = (age > 29).fillna(False).astype(bool)

    logger.info("age_curve: added 6 columns")
    return df
