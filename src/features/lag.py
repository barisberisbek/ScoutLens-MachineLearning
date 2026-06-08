"""M6 — within-player lag / trajectory features (the most intricate module).

For each `LAG_STATS` column, attach the previous-season value (`_lag1`) and the
season-over-season change (`delta_`). Relies on `player_id` being stable across seasons
(Phase-2 deterministic synthetic ids guarantee this for unmatched players); a
synthetic→TM transition legitimately changes the id, so `has_lag1=False` there — handled,
not a bug. `consecutive_seasons` is gap-aware: it counts an uninterrupted run and resets
when a season is skipped.

CRITICAL: sort by (player_id, season) before shifting, or the lag is wrong.
"""

from __future__ import annotations

import pandas as pd

from src.utils.constants import LAG_STATS, SEASON_END_YEAR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `<stat>_lag1` + `delta_<stat>` for LAG_STATS, plus has_lag1 + consecutive_seasons."""
    df = df.copy().sort_values(["player_id", "season"]).reset_index(drop=True)

    added = 0
    for stat in LAG_STATS:
        if stat not in df.columns:
            continue
        lag = df.groupby("player_id", sort=False)[stat].shift(1)
        df[f"{stat}_lag1"] = lag
        df[f"delta_{stat}"] = df[stat] - lag
        added += 2

    # has_lag1: a previous row exists for this player (regardless of any season gap).
    df["has_lag1"] = df.groupby("player_id", sort=False).cumcount() > 0

    # consecutive_seasons: gap-aware run length. A "break" begins a new streak whenever the
    # season-end year is not exactly one greater than the previous row's (or there is none).
    end_year = df["season"].map(SEASON_END_YEAR)
    prev_end_year = df.groupby("player_id", sort=False)["season"].shift(1).map(SEASON_END_YEAR)
    df["_break"] = (end_year - prev_end_year != 1)  # NaN (first row) != 1 → True → new streak
    df["_streak"] = df.groupby("player_id", sort=False)["_break"].cumsum()
    df["consecutive_seasons"] = df.groupby(["player_id", "_streak"], sort=False).cumcount() + 1
    df = df.drop(columns=["_break", "_streak"])

    logger.info("lag: added %d columns (+ has_lag1, consecutive_seasons)", added)
    return df
