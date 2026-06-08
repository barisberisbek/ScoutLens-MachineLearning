"""Stage-1 training-pair construction + leakage-safe feature/target separation.

A pair = (player, season) → same player's next season, where both seasons meet the
min-minutes bar, the seasons are consecutive (gap-aware), and the primary position is
unchanged. The label is `next_<target>`; the player's CURRENT-season stats are features
(past info, not leakage). Market-value columns are excluded (Stage-2's domain).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.stage1.target_specs import ALL_TARGETS
from src.utils.constants import SEASON_END_YEAR, TRAIN_SEASONS, TEST_SEASON
from src.utils.io import load_parquet, project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

# current-season → next-season target maps to these train/val seasons:
_TRAIN_CURRENT = ["2021-22", "2022-23"]   # → targets in 22-23, 23-24
_VAL_CURRENT = ["2023-24"]                # → target in 24-25

# Columns never used as Stage-1 features (leakage / metadata / Stage-2's market value).
_EXCLUDE_FEATURES: frozenset[str] = frozenset({
    "market_value_eur", "log_market_value", "log_market_value_lag1", "delta_log_market_value",
    "resolve_score", "min_minutes_threshold", "meets_min_minutes", "season_end_year",
})


def _features_path() -> str:
    return str(project_root() / "data" / "processed" / "features.parquet")


def build_all_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Return all valid (current→next) pairs with `next_<target>` columns and a `split` label."""
    df = df.sort_values(["player_id", "season"]).reset_index(drop=True)
    g = df.groupby("player_id", sort=False)
    out = df.copy()
    out["next_season"] = g["season"].shift(-1)
    out["next_meets"] = g["meets_min_minutes"].shift(-1)
    out["next_pos"] = g["primary_position"].shift(-1)
    for t in ALL_TARGETS:
        if t in out.columns:
            out[f"next_{t}"] = g[t].shift(-1)

    ey = out["season"].map(SEASON_END_YEAR)
    ney = out["next_season"].map(SEASON_END_YEAR)
    valid = (
        out["next_season"].notna()
        & (ney - ey == 1)
        & out["meets_min_minutes"].fillna(False)
        & out["next_meets"].fillna(False)
        & (out["primary_position"] == out["next_pos"])
    )
    pairs = out[valid].copy()
    pairs["split"] = np.select(
        [pairs["season"].isin(_TRAIN_CURRENT), pairs["season"].isin(_VAL_CURRENT)],
        ["train", "val"], default="other",
    )
    return pairs[pairs["split"].isin(["train", "val"])].reset_index(drop=True)


def feature_columns(pairs: pd.DataFrame) -> list[str]:
    """Numeric/bool feature columns, excluding targets, next_*, market value, and metadata."""
    next_cols = {c for c in pairs.columns if c.startswith("next_")}
    cols = []
    for c in pairs.columns:
        if c in _EXCLUDE_FEATURES or c in next_cols or c == "split":
            continue
        if pd.api.types.is_bool_dtype(pairs[c]) or pd.api.types.is_numeric_dtype(pairs[c]):
            cols.append(c)
    return cols


def load(position: str, features_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load (train_pairs, val_pairs, feature_cols) for one position."""
    df = load_parquet(features_path or _features_path())
    pairs = build_all_pairs(df)
    pairs = pairs[pairs["primary_position"] == position].reset_index(drop=True)
    feats = feature_columns(pairs)
    train = pairs[pairs["split"] == "train"].reset_index(drop=True)
    val = pairs[pairs["split"] == "val"].reset_index(drop=True)
    # Drop features that are entirely null in this position's train set (no signal; also
    # avoids the median-imputer "all-missing column" warning, e.g. xag z-scores for GK).
    feats = [c for c in feats if train[c].notna().any()]
    logger.info("%s: %d train / %d val pairs, %d features", position, len(train), len(val), len(feats))
    return train, val, feats


def assert_no_leakage(feature_cols: list[str]) -> None:
    """Guard: no future (`next_*`) or market-value column may be a feature."""
    bad = [c for c in feature_cols if c.startswith("next_") or "market_value" in c]
    if bad:
        raise AssertionError(f"leakage: forbidden feature columns present: {bad}")
