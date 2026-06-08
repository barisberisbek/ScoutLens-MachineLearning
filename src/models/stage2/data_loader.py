"""Stage-2 dataset construction (single-row, not pairs) + leakage-safe features.

One row per (player, season) with `market_value_eur` observed and min-minutes met; target
is `log_market_value`. All market-value columns are excluded from features — including
`log_market_value_lag1` (start-excluded: it would let the model copy last year's value
instead of learning the stat→value mapping; revisit via ablation).
"""

from __future__ import annotations

import pandas as pd

from src.utils.constants import TEST_SEASON, TRAIN_SEASONS
from src.utils.io import load_parquet, project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

TARGET = "log_market_value"

_EXCLUDE_FEATURES: frozenset[str] = frozenset({
    "log_market_value", "log_market_value_lag1", "delta_log_market_value",
    "market_value_eur", "market_value_date",
    "resolve_score", "min_minutes_threshold", "meets_min_minutes", "season_end_year",
})


def _features_path() -> str:
    return str(project_root() / "data" / "processed" / "features.parquet")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric/bool features, excluding the target, all market-value cols, and metadata."""
    cols = []
    for c in df.columns:
        if c in _EXCLUDE_FEATURES:
            continue
        if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_dataset(df: pd.DataFrame, position: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Pure: filter to MV-observed + min-minutes + position, split by season, pick features."""
    df = df[df["market_value_eur"].notna() & df["meets_min_minutes"].fillna(False)
            & (df["primary_position"] == position)].reset_index(drop=True)
    feats = feature_columns(df)
    train = df[df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    val = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    feats = [c for c in feats if len(train) and train[c].notna().any()]  # drop all-null-in-train
    return train, val, feats


def load(position: str, features_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(train, val, feature_cols) for one position: MV-observed + min-minutes rows."""
    train, val, feats = build_dataset(load_parquet(features_path or _features_path()), position)
    logger.info("%s: %d train / %d val rows, %d features", position, len(train), len(val), len(feats))
    return train, val, feats


def assert_no_leakage(feature_cols: list[str]) -> None:
    """Guard: no market-value column (target, lag, or raw) may be a feature."""
    bad = [c for c in feature_cols if "market_value" in c]
    if bad:
        raise AssertionError(f"leakage: market-value column(s) in features: {bad}")
