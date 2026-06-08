"""M3 — position- and league-conditional z-scores.

The same per-90 value means different things by position (Phase-3 Fig05: median rates differ
by position) and by league, so models need standardized rates. Each stat gets `_z_pos`
(within season × primary_position) and `_z_league` (within season × league). Mean/std use
non-null values; a group with `<15` non-null values is too small to standardize → NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

MIN_GROUP = 15

# Stats worth standardizing (per-90 rates + key counts + composites).
Z_SCORE_STATS = [
    "goals_per_90", "assists_per_90", "shots_per_90",
    "xg_per_90", "xag_per_90", "npxg_per_90",
    "tackles_won_per_90", "interceptions_per_90",
    "saves_per_90", "clean_sheets_per_match",
    "minutes_played", "matches_played",
    "goal_threat", "defensive_actions", "creative_threat",
]


def _zscore(x: pd.Series) -> pd.Series:
    """Standardize a group, returning all-NaN if fewer than MIN_GROUP non-null values."""
    if x.notna().sum() < MIN_GROUP:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / (x.std() + 1e-6)


def add_z_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `_z_pos` and `_z_league` for each `Z_SCORE_STATS` column present."""
    df = df.copy()
    groupings = {"_z_pos": ["season", "primary_position"], "_z_league": ["season", "league"]}

    added = 0
    thin = 0
    for stat in Z_SCORE_STATS:
        if stat not in df.columns:
            continue
        for suffix, keys in groupings.items():
            z = df.groupby(keys, observed=True)[stat].transform(_zscore)
            df[f"{stat}{suffix}"] = z
            added += 1
            # count groups that were too thin to standardize (all-NaN where base had values)
            sizes = df.groupby(keys, observed=True)[stat].transform(lambda s: s.notna().sum())
            thin += int(((sizes < MIN_GROUP) & df[stat].notna()).sum() > 0)

    if thin:
        logger.warning("z_scores: %d stat×grouping combos had <%d-member groups (→ NaN there)",
                       thin, MIN_GROUP)
    logger.info("z_scores: added %d columns", added)
    return df
