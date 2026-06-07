"""Loader for the Kaggle "Hubert" Football Players Stats 2024-25 dataset.

The raw CSV is a merge of FBref's 10 stat tables, so ~118 of 267 columns are
duplicate re-exports (every table re-emits Player/Nation/Pos/Comp/Age/Born/90s
and any shared stat). This module de-duplicates, renames to snake_case, parses
league/nationality/position, and flags mid-season transfers — but does NOT merge
split-season rows or engineer features (those belong to Phases 2 and 4).

Output: a tidy one-row-per-(player, club-stint) DataFrame for 2024-25 top-5.
"""

from __future__ import annotations

import re

import pandas as pd

from src.utils.constants import (
    FBREF_COLUMN_RENAME,
    LEAGUE_NAME_PREFIX_MAP,
    STATS_PRESERVE_SUFFIXED,
    TOP5_LEAGUES,
    normalize_position,
)
from src.utils.io import project_root
from src.utils.logging import get_logger

# FBref stat-table suffixes, sorted longest-first so e.g. "_stats_passing_types"
# is matched before "_stats_passing" and "_stats_keeper_adv" before "_stats_keeper".
_STAT_SUFFIXES: list[str] = sorted(
    [
        "_stats_shooting", "_stats_passing", "_stats_passing_types", "_stats_gca",
        "_stats_defense", "_stats_possession", "_stats_playing_time", "_stats_misc",
        "_stats_keeper", "_stats_keeper_adv",
    ],
    key=len,
    reverse=True,
)


def load_kaggle_2024_25() -> pd.DataFrame:
    """Load and normalize the Kaggle top-5 2024-25 dataset into a tidy DataFrame."""
    logger = get_logger(__name__)
    csv_path = project_root() / "data" / "raw" / "kaggle" / "players_data-2024_2025.csv"

    df = pd.read_csv(csv_path)
    logger.info(f"Read raw: shape={df.shape}")

    df = _deduplicate_stat_columns(df, logger)
    df = df.drop(columns=["Rk"], errors="ignore")

    rename_dict = {
        col: FBREF_COLUMN_RENAME.get(col, _auto_normalize(col)) for col in df.columns
    }
    df = df.rename(columns=rename_dict)
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names after rename: {dupes}")

    # Numeric coercions (replace existing columns in place)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce").astype("Int64")
    df["minutes_played"] = pd.to_numeric(df["minutes_played"], errors="coerce")
    if "matches_90s" in df.columns:
        df["matches_90s"] = pd.to_numeric(df["matches_90s"], errors="coerce")

    # Derived columns, built and attached in one concat to avoid frame fragmentation.
    # Nationality is "<fbref_code> <ISO3>" with codes of varying length (2 or 3 chars),
    # so split on whitespace rather than fixed slicing.
    parts = df["nationality_raw"].astype("string").str.split(n=1)
    league = df["league_raw"].map(LEAGUE_NAME_PREFIX_MAP).fillna(df["league_raw"])
    derived = pd.DataFrame(
        {
            "league": league,
            "nationality_fbref": parts.str[0].str.lower(),
            "nationality_iso3": parts.str[-1].str.upper(),
            "position_detail": df["position_raw"],
            "position_primary": df["position_raw"].apply(normalize_position),
            "season": "2024-25",
            "season_end_year": 2025,
            "data_source": "kaggle_hubert",
        },
        index=df.index,
    )
    df = pd.concat([df, derived], axis=1)

    n_unmapped = int((df["league"] == df["league_raw"]).sum())
    if n_unmapped > 0:
        unmapped = sorted(df.loc[df["league"] == df["league_raw"], "league_raw"].unique())
        logger.warning(f"{n_unmapped} rows have unmapped league_raw values: {unmapped}")

    df = _flag_split_season_players(df, logger)

    _validate(df, logger)
    logger.info(f"Final clean shape={df.shape}")
    return df


def _deduplicate_stat_columns(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Drop FBref stat-table duplicate columns.

    For each "_stats_X"-suffixed column: if the bare base name already exists,
    drop the suffixed duplicate; otherwise rename it to the base. Columns listed
    in STATS_PRESERVE_SUFFIXED are renamed to a distinct name instead of dropped
    (their bare base means a different stat).
    """
    to_drop: list[str] = []
    to_rename: dict[str, str] = {}
    n_preserve = 0

    for col in df.columns:
        if col in STATS_PRESERVE_SUFFIXED:
            to_rename[col] = STATS_PRESERVE_SUFFIXED[col]
            n_preserve += 1
            continue
        for suffix in _STAT_SUFFIXES:
            if col.endswith(suffix):
                base = col[: -len(suffix)]
                if base in df.columns:
                    to_drop.append(col)
                else:
                    to_rename[col] = base
                break

    logger.info(
        f"Dedup: dropping {len(to_drop)} duplicate, renaming {len(to_rename) - n_preserve} "
        f"unique, preserving {n_preserve} distinct suffixed columns"
    )
    df = df.drop(columns=to_drop).rename(columns=to_rename)
    logger.info(f"After dedup: shape={df.shape}")
    return df


def _auto_normalize(col_name: str) -> str:
    """Fallback normalizer for columns absent from FBREF_COLUMN_RENAME.

    Symbols are encoded meaningfully BEFORE the generic non-alphanumeric pass so
    that a stat and its percentage / per-90 / plus-minus variant don't collapse to
    the same name (e.g. 'Succ' vs 'Succ%', 'xG' vs 'xG+/-', '/90' vs '+/-90').
    Conventions match FBREF_COLUMN_RENAME (_pct, _per_90, _plus_minus).
    """
    s = col_name
    s = s.replace("+/-", " plus_minus ")
    s = s.replace("%", " pct ")
    s = s.replace("/90", " per_90 ")
    s = s.replace("/", " per ")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _flag_split_season_players(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Flag (do NOT merge) players appearing in multiple rows (mid-season transfer).

    Grouped on (player_name, birth_year, nationality_iso3) so same-named distinct
    players are not conflated. Merging is Phase 2's responsibility.
    """
    grp_cols = ["player_name", "birth_year", "nationality_iso3"]
    counts = df.groupby(grp_cols, dropna=False).size().reset_index(name="_n")
    splits = counts[counts["_n"] > 1]
    df = df.merge(
        splits[grp_cols].assign(is_split_season=True), on=grp_cols, how="left"
    )
    df["is_split_season"] = df["is_split_season"].fillna(False).astype(bool)
    logger.info(
        f"Split-season records: {int(df['is_split_season'].sum())} "
        f"from {len(splits)} players"
    )
    return df


def _validate(df: pd.DataFrame, logger) -> None:
    """Loud sanity checks; failing fast beats silently shipping bad data."""
    assert len(df) >= 2700, f"Expected >=2700 rows, got {len(df)}"
    assert df["league"].nunique() == 5, f"Expected 5 leagues, got {df['league'].nunique()}"
    actual_leagues = set(df["league"].unique())
    assert actual_leagues == set(TOP5_LEAGUES), f"League mismatch: {actual_leagues}"
    assert df["minutes_played"].max() > 3000, "Top minutes look wrong"
    assert (
        df["position_primary"].isin(["GK", "DEF", "MID", "FWD"]).all()
    ), "Unknown position values exist"
    logger.info("All validations passed")
