"""M1 — per-90 (and per-match) rate features.

Guarantees a `<stat>_per_90` column for every stat in `PER_90_STATS` (the Phase-3/P2-D6
survivor set). Match-based goalkeeper counts (clean sheets) get a `_per_match` rate instead.
NaN-safe: `minutes_played <= 0` (or `matches_played <= 0`) → NaN, never division-by-zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.constants import PER_90_STATS
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _rate(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    """`numerator / denominator`, returning NaN wherever denominator <= 0 or is null."""
    num = numerator.to_numpy(dtype="float64")
    den = denominator.to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def add_per_90_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `_per_90` for any `PER_90_STATS` stat lacking one, plus `clean_sheets_per_match`."""
    df = df.copy()
    nineties = df["minutes_played"] / 90.0

    added = []
    for stat in PER_90_STATS:
        col = f"{stat}_per_90"
        if stat not in df.columns or col in df.columns:
            continue  # base missing, or per-90 already provided by Phase 2
        df[col] = _rate(df[stat], nineties)
        added.append(col)

    # Match-based GK rate (clean sheets are per-appearance, not per-90).
    if "clean_sheets" in df.columns and "clean_sheets_per_match" not in df.columns:
        df["clean_sheets_per_match"] = _rate(df["clean_sheets"], df["matches_played"])
        added.append("clean_sheets_per_match")

    logger.info("per_90: added %d rate column(s): %s", len(added), added)
    return df
