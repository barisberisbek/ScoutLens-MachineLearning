"""Coverage report for the FBref + Understat raw scrape.

Scans data/raw/fbref/ and data/raw/understat/, records row counts per combination,
flags missing combos and out-of-range player counts, and writes a committed CSV to
reports/. Run after scraping (works on partial data too):

    python scripts/validate_coverage.py
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from src.data.understat_loader import _league_slug
from src.utils.constants import (
    FBREF_LEAGUE_IDS,
    FBREF_SEASONS,
    FBREF_STAT_TYPES,
    LOWER4_LEAGUES,
    TOP5_LEAGUES,
)
from src.utils.io import project_root

# Sanity ranges for the 'standard' table (one row per player-season).
_TOP5_RANGE = (450, 750)
_LOWER_RANGE = (250, 650)


def _num_rows(path) -> int | None:
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    root = project_root()
    fbref_base = root / "data" / "raw" / "fbref"
    understat_base = root / "data" / "raw" / "understat"

    rows = []
    # FBref: league x season x stat_type
    for league in FBREF_LEAGUE_IDS:
        slug = _league_slug(league)
        for season in FBREF_SEASONS:
            for stat_type in FBREF_STAT_TYPES:
                path = fbref_base / slug / season / f"{stat_type}.parquet"
                exists = path.exists()
                rows.append(
                    {
                        "source": "fbref",
                        "league": league,
                        "season": season,
                        "stat_type": stat_type,
                        "row_count": _num_rows(path) if exists else 0,
                        "file_exists": exists,
                    }
                )
    # Understat: top-5 x season (single table)
    for league in TOP5_LEAGUES:
        slug = _league_slug(league)
        for season in FBREF_SEASONS:
            path = understat_base / slug / f"{season}.parquet"
            exists = path.exists()
            rows.append(
                {
                    "source": "understat",
                    "league": league,
                    "season": season,
                    "stat_type": "player_season",
                    "row_count": _num_rows(path) if exists else 0,
                    "file_exists": exists,
                }
            )

    df = pd.DataFrame(rows)
    out_path = root / "reports" / "fbref_coverage_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # ── Console summary ──
    total = len(df)
    present = int(df["file_exists"].sum())
    print(f"Coverage summary written to {out_path}")
    print(f"\nCombinations: {present}/{total} present, {total - present} missing")
    print("\nBy source:")
    print(df.groupby("source")["file_exists"].agg(["sum", "count"]).to_string())

    missing = df[~df["file_exists"]]
    if not missing.empty:
        print(f"\nMISSING ({len(missing)}):")
        for _, r in missing.iterrows():
            print(f"  {r['source']}: {r['league']}/{r['season']}/{r['stat_type']}")

    # Sanity: standard-table player counts per league/season
    std = df[(df["source"] == "fbref") & (df["stat_type"] == "standard") & df["file_exists"]]
    print("\nStandard-table player counts (sanity):")
    flags = []
    for _, r in std.iterrows():
        lo, hi = _TOP5_RANGE if r["league"] in TOP5_LEAGUES else _LOWER_RANGE
        flag = "" if lo <= r["row_count"] <= hi else "  <-- OUT OF RANGE"
        if flag:
            flags.append((r["league"], r["season"], r["row_count"]))
        print(f"  {r['league']}/{r['season']}: {r['row_count']}{flag}")
    if flags:
        print(f"\n{len(flags)} out-of-range counts flagged above.")


if __name__ == "__main__":
    main()
