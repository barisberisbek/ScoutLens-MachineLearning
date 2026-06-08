"""M8 — one-hot categorical encodings.

Position and continent one-hots so linear models (and SHAP) read them cleanly; tree models
also benefit. League one-hot is intentionally skipped (tree models learn it from `league`;
add in Phase 5 if needed). `worldcup_25_squad` is already a boolean in the panel (D-06).
"""

from __future__ import annotations

import pandas as pd

from src.utils.constants import POSITION_PRIMARY_MAP
from src.utils.logging import get_logger

logger = get_logger(__name__)

# The 7 continent buckets present in the panel (continent_map + 'Other' fallback).
CONTINENT_GROUPS = ["Europe", "Africa", "SouthAmerica", "NorthAmerica", "Asia", "Oceania", "Other"]
POSITIONS = list(dict.fromkeys(POSITION_PRIMARY_MAP.values()))  # GK, DEF, MID, FWD


def add_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_<POS> (4) and is_continent_<X> (7) booleans."""
    df = df.copy()
    for pos in POSITIONS:
        df[f"is_{pos}"] = (df["primary_position"] == pos).astype(bool)
    for cont in CONTINENT_GROUPS:
        df[f"is_continent_{cont}"] = (df["continent_group"] == cont).astype(bool)

    logger.info("categorical: added %d columns", len(POSITIONS) + len(CONTINENT_GROUPS))
    return df
