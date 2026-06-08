"""M7 — lookup-based anchor multipliers.

`year_inflation_multiplier` encodes the empirical YoY market inflation (Phase-3 Fig03 /
D-07) so models can subtract the year effect directly. `league_value_multiplier` is already
in the panel (from Phase 2); it is reaffirmed from the committed CSV only if absent. These
are the Stage-2 anchors (league premium + year inflation).
"""

from __future__ import annotations

import pandas as pd

from src.utils.constants import YEAR_INFLATION
from src.utils.io import load_lookup_csv
from src.utils.logging import get_logger

logger = get_logger(__name__)


def add_multiplier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year_inflation_multiplier; backfill league_value_multiplier from CSV if missing."""
    df = df.copy()
    df["year_inflation_multiplier"] = df["season"].map(YEAR_INFLATION)

    added = 1
    if "league_value_multiplier" not in df.columns:
        df["league_value_multiplier"] = df["league"].map(load_lookup_csv("league_value_multipliers"))
        added += 1

    logger.info("multipliers: added %d column(s)", added)
    return df
