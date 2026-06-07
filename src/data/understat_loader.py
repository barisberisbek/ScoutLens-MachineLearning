"""Understat player-season scraper via soccerdata (xG source for top-5 leagues).

Understat covers only the top-5 European leagues, but provides the expected-goals
metrics that soccerdata's FBref tables lack: ``xg``, ``np_xg``, ``xa``, ``xg_chain``,
``xg_buildup`` (plus goals/assists/shots/key_passes/minutes). This fills the historical
top-5 xG gap (Stage-1 FWD targets, §6.2). Lower leagues are NOT covered by Understat.

Output is RAW (one parquet per (league, season)); name-based reconciliation to FBref /
Kaggle happens in Phase 2 (no shared player_id across sources).
"""

from __future__ import annotations

import pandas as pd
import soccerdata as sd

from src.utils.constants import FBREF_LEAGUE_IDS, FBREF_SEASONS, TOP5_LEAGUES
from src.utils.io import project_root, save_parquet
from src.utils.logging import get_logger


def _league_slug(league: str) -> str:
    """Filesystem-safe slug (matches the FBref scraper's convention)."""
    return league.lower().replace("ü", "u").replace(" ", "_")


def _fetch_one(league: str, season: str) -> pd.DataFrame:
    """Fetch one (league, season) of Understat player-season stats as a flat DataFrame."""
    understat_id = FBREF_LEAGUE_IDS[league]  # soccerdata keys are shared for the top-5
    reader = sd.Understat(leagues=[understat_id], seasons=[season])
    return reader.read_player_season_stats().reset_index()


def scrape_all_understat(
    force: bool = False,
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
) -> dict:
    """Scrape Understat player-season stats for the (filtered) top-5 × seasons plan.

    Resumable (skip existing unless ``force``); error-tolerant (log + continue).

    Returns
    -------
    dict with 'completed' / 'skipped' / 'failed' lists.
    """
    logger = get_logger(__name__)
    leagues = leagues or TOP5_LEAGUES
    seasons = seasons or FBREF_SEASONS

    status: dict[str, list] = {"completed": [], "skipped": [], "failed": []}
    total = len(leagues) * len(seasons)
    idx = 0
    base = project_root() / "data" / "raw" / "understat"

    for league in leagues:
        slug = _league_slug(league)
        for season in seasons:
            idx += 1
            out_path = base / slug / f"{season}.parquet"
            tag = f"{league}/{season}"
            if out_path.exists() and not force:
                logger.info(f"[{idx}/{total}] SKIP (cached): {tag}")
                status["skipped"].append((league, season))
                continue
            logger.info(f"[{idx}/{total}] FETCH: {tag}")
            try:
                df = _fetch_one(league, season)
                if df is None or df.empty:
                    logger.warning(f"  Empty result: {tag}")
                    status["failed"].append((league, season, "empty"))
                    continue
                save_parquet(df, out_path)
                logger.info(f"  Saved: {df.shape} -> {out_path.relative_to(base)}")
                status["completed"].append((league, season))
            except Exception as e:  # noqa: BLE001 — tolerate one combo, keep going
                logger.exception(f"  FAILED: {tag}: {e}")
                status["failed"].append((league, season, str(e)))

    logger.info(
        f"Done: {len(status['completed'])} completed, "
        f"{len(status['skipped'])} skipped, {len(status['failed'])} failed"
    )
    return status
