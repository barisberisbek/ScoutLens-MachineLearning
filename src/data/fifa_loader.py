"""Loader for FIFA / EA FC player ratings across 4 seasons (D-17, §10.11).

Provides overall + potential as features (high signal for young players). Sources, after
pre-research, are consolidated to TWO files:
  - `data/raw/fifa/EA FC 24/male_players.csv` (stefanoleone992 schema) holds one clean
    snapshot per fifa_version 15-24 → we take 22/23/24 from it (skips the redundant 5.3 GB
    weekly-update mega-file and the legacy file).
  - `data/raw/fifa/EA FC 25/male_players.csv` (nyagami schema) → FC 25 (2024-25). It has NO
    `potential` / `dob` / sofifa_id.

Output is RAW-unified: name resolution to FBref/TM is Phase 2. Reads only data/raw/fifa/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.constants import (
    FIFA_NYAGAMI_COLUMN_RENAME,
    FIFA_POSITION_MAP,
    FIFA_STEFANO_COLUMN_RENAME,
    FIFA_YEAR_TO_SEASON,
)
from src.utils.io import project_root
from src.utils.logging import get_logger

_UNIFIED_COLS = [
    "fifa_year", "season", "source", "player_name", "age", "date_of_birth",
    "nationality", "club", "league", "primary_position", "position_detail",
    "overall", "potential", "sofifa_id",
]


def _fifa_primary_position(pos_str) -> str:
    """First token of a FIFA position string ("ST, CF" / "ST") → GK/DEF/MID/FWD/Other."""
    if pos_str is None or (isinstance(pos_str, float) and pd.isna(pos_str)):
        return "Other"
    first = str(pos_str).split(",")[0].strip().upper()
    return FIFA_POSITION_MAP.get(first, "Other")


def _select_unified(df: pd.DataFrame) -> pd.DataFrame:
    for c in _UNIFIED_COLS:
        if c not in df.columns:
            df[c] = None
    return df[_UNIFIED_COLS]


def load_stefano_versions(path: Path, versions=(22, 23, 24)) -> pd.DataFrame:
    """Load FIFA 22/23/24 rows from the stefano EA-FC-24 multi-version file."""
    logger = get_logger(__name__)
    df = pd.read_csv(path, low_memory=True)
    df["fifa_version"] = pd.to_numeric(df["fifa_version"], errors="coerce").astype("Int64")
    df = df[df["fifa_version"].isin(versions)].copy()
    df = df.rename(columns=FIFA_STEFANO_COLUMN_RENAME)
    df["fifa_year"] = df["fifa_version"].astype(int)
    df["season"] = df["fifa_year"].map(FIFA_YEAR_TO_SEASON)
    df["source"] = "stefanoleone992_fc24file"
    df["primary_position"] = df["position_detail"].apply(_fifa_primary_position)
    logger.info(
        "stefano versions loaded: "
        + df.groupby("fifa_year").size().to_dict().__str__()
    )
    return _select_unified(df)


def load_nyagami_fc25(path: Path) -> pd.DataFrame:
    """Load EA FC 25 (2024-25) from the nyagami file (no potential/dob/sofifa_id)."""
    logger = get_logger(__name__)
    df = pd.read_csv(path, low_memory=True)
    df = df.rename(columns=FIFA_NYAGAMI_COLUMN_RENAME)
    df["fifa_year"] = 25
    df["season"] = FIFA_YEAR_TO_SEASON[25]
    df["source"] = "nyagami_fc25"
    df["primary_position"] = df["position_detail"].apply(_fifa_primary_position)
    for missing in ("potential", "date_of_birth", "sofifa_id"):
        if missing not in df.columns:
            df[missing] = None
    logger.info(f"nyagami FC25 loaded: {len(df)} players (potential absent → null)")
    return _select_unified(df)


def load_all_fifa() -> pd.DataFrame:
    """Concatenate FIFA 22/23/24 (stefano) + FC 25 (nyagami) into the unified schema."""
    root = project_root() / "data" / "raw" / "fifa"
    stefano = load_stefano_versions(root / "EA FC 24" / "male_players.csv")
    nyagami = load_nyagami_fc25(root / "EA FC 25" / "male_players.csv")
    out = pd.concat([stefano, nyagami], ignore_index=True)
    # numeric coercions
    for col in ("overall", "potential", "age"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
