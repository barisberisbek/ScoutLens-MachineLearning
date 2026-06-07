"""FBref multi-season / lower-league scraper built on soccerdata 1.9.0.

soccerdata's ``FBref.read_player_season_stats`` whitelists only 5 player-season stat
types (``fbref.py:546``: standard, shooting, playing_time, keeper, misc), yet its fetch
+ parse logic is type-agnostic. We monkeypatch that one method with an expanded
allow-list so we can also pull passing / defense / gca / possession, reusing soccerdata's
polite rate-limiting, on-disk cache, and comment-wrapped-table parsing.

Why monkeypatch instead of subclass: soccerdata resolves league IDs via
``LEAGUE_DICT[...][cls.__name__]`` (``_common.py:378``), so a subclass with a different
``__name__`` breaks league validation. Patching the method on ``sd.FBref`` itself keeps
the class identity (and all name-based internals) intact.

NOTE: the patched body is a faithful copy of soccerdata 1.9.0 ``fbref.py:512-609`` (only
the allow-list differs) — COUPLED to that pinned version. Output is RAW (no column
normalization; that is Phase 2).
"""

from __future__ import annotations

import pandas as pd
import soccerdata as sd
import soccerdata.fbref as _sdfb
from lxml import etree, html

from src.utils.constants import (
    FBREF_LEAGUE_IDS,
    FBREF_SEASONS,
    FBREF_STAT_TYPES,
    FBREF_STAT_TYPES_EXTENDED,
    FBREF_STAT_TYPES_NATIVE,
)
from src.utils.io import project_root, save_parquet
from src.utils.logging import get_logger

_EXTENDED_ALLOWED = FBREF_STAT_TYPES_NATIVE + FBREF_STAT_TYPES_EXTENDED


def _read_player_season_stats_extended(self, stat_type: str = "standard") -> pd.DataFrame:
    """Patched ``FBref.read_player_season_stats`` — soccerdata 1.9.0 body, wider allow-list."""
    player_stats = _EXTENDED_ALLOWED
    filemask = "players_{}_{}_{}.html"

    if stat_type not in player_stats:
        raise TypeError(f"Invalid argument: stat_type should be in {player_stats}")

    if stat_type == "standard":
        page = "stats"
    elif stat_type == "playing_time":
        page = "playingtime"
    elif stat_type == "keeper":
        page = "keepers"
    elif stat_type == "keeper_adv":
        page = "keepersadv"  # FBref's advanced-GK URL segment (table id stays stats_keeper_adv)
    else:
        page = stat_type

    seasons = self.read_seasons()
    players = []
    for (lkey, skey), season in seasons.iterrows():
        big_five = lkey == "Big 5 European Leagues Combined"
        filepath = self.data_dir / filemask.format(lkey, skey, stat_type)
        url = (
            _sdfb.FBREF_API
            + "/".join(season.url.split("/")[:-1])
            + f"/{page}"
            + ("/players/" if big_five else "/")
            + season.url.split("/")[-1]
        )
        reader = self.get(url, filepath)
        tree = html.parse(reader)
        for elem in tree.xpath("//td[@data-stat='comp_level']//span"):
            elem.getparent().remove(elem)
        if big_five:
            (html_table,) = tree.xpath(f"//table[@id='stats_{stat_type}']")
            df_table = _sdfb._parse_table(html_table)
            df_table[("Unnamed: league", "league")] = (
                df_table.xs("Comp", axis=1, level=1).squeeze().map(_sdfb.BIG_FIVE_DICT)
            )
            df_table[("Unnamed: season", "season")] = skey
            df_table.drop("Comp", axis=1, level=1, inplace=True)
        else:
            (el,) = tree.xpath(f"//comment()[contains(.,'div_stats_{stat_type}')]")
            parser = etree.HTMLParser(recover=True)
            (html_table,) = etree.fromstring(el.text, parser).xpath(
                f"//table[contains(@id, 'stats_{stat_type}')]"
            )
            df_table = _sdfb._parse_table(html_table)
            df_table[("Unnamed: league", "league")] = lkey
            df_table[("Unnamed: season", "season")] = skey
        df_table = _sdfb._fix_nation_col(df_table)
        players.append(df_table)

    df = _sdfb._concat(players, key=["league", "season"])
    df = df[df.Player != "Player"]
    return (
        df.drop("Matches", axis=1, level=0)
        .drop("Rk", axis=1, level=0)
        .rename(columns={"Squad": "team"})
        .replace({"team": _sdfb.TEAMNAME_REPLACEMENTS})
        .pipe(_sdfb.standardize_colnames, cols=["Player", "Nation", "Pos", "Age", "Born"])
        .set_index(["league", "season", "team", "player"])
        .sort_index()
    )


# Apply the patch once, at import. Widens only the stat-type allow-list; native types
# behave identically. Affects sd.FBref process-wide (intended for this scraper).
sd.FBref.read_player_season_stats = _read_player_season_stats_extended


def _league_slug(league: str) -> str:
    """Filesystem-safe slug: 'Süper Lig' → 'super_lig', 'Premier League' → 'premier_league'."""
    return league.lower().replace("ü", "u").replace(" ", "_")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a MultiIndex column axis to underscore-joined names and de-duplicate.

    RAW output: names kept verbatim (Phase 2 normalizes). Duplicate flattened names get a
    numeric suffix so parquet round-trips cleanly.
    """
    if isinstance(df.columns, pd.MultiIndex):
        flat = ["_".join(filter(None, map(str, col))).strip("_") for col in df.columns]
    else:
        flat = [str(c) for c in df.columns]
    seen: dict[str, int] = {}
    unique = []
    for name in flat:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}__{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)
    df = df.copy()
    df.columns = unique
    return df


def _fetch_one(league: str, season: str, stat_type: str) -> pd.DataFrame:
    """Fetch one (league, season, stat_type) as a flat RAW DataFrame."""
    fbref_id = FBREF_LEAGUE_IDS[league]
    reader = sd.FBref(leagues=[fbref_id], seasons=[season])
    df = reader.read_player_season_stats(stat_type=stat_type)
    df = df.reset_index()
    return _flatten_columns(df)


def scrape_all_fbref(
    force: bool = False,
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    stat_types: list[str] | None = None,
) -> dict:
    """Scrape the (filtered) plan of (league, season, stat_type) combinations.

    Resumable: an existing output parquet is skipped unless ``force``. Error-tolerant:
    a failed combo is logged and the loop continues.

    Returns
    -------
    dict with 'completed' / 'skipped' / 'failed' lists.
    """
    logger = get_logger(__name__)
    leagues = leagues or list(FBREF_LEAGUE_IDS.keys())
    seasons = seasons or FBREF_SEASONS
    stat_types = stat_types or FBREF_STAT_TYPES

    status: dict[str, list] = {"completed": [], "skipped": [], "failed": []}
    total = len(leagues) * len(seasons) * len(stat_types)
    idx = 0
    base = project_root() / "data" / "raw" / "fbref"

    for league in leagues:
        slug = _league_slug(league)
        for season in seasons:
            for stat_type in stat_types:
                idx += 1
                out_path = base / slug / season / f"{stat_type}.parquet"
                tag = f"{league}/{season}/{stat_type}"
                if out_path.exists() and not force:
                    logger.info(f"[{idx}/{total}] SKIP (cached): {tag}")
                    status["skipped"].append((league, season, stat_type))
                    continue
                logger.info(f"[{idx}/{total}] FETCH: {tag}")
                try:
                    df = _fetch_one(league, season, stat_type)
                    if df is None or df.empty:
                        logger.warning(f"  Empty result: {tag}")
                        status["failed"].append((league, season, stat_type, "empty"))
                        continue
                    save_parquet(df, out_path)
                    logger.info(f"  Saved: {df.shape} -> {out_path.relative_to(base)}")
                    status["completed"].append((league, season, stat_type))
                except Exception as e:  # noqa: BLE001 — tolerate one combo, keep going
                    logger.exception(f"  FAILED: {tag}: {e}")
                    status["failed"].append((league, season, stat_type, str(e)))

    logger.info(
        f"Done: {len(status['completed'])} completed, "
        f"{len(status['skipped'])} skipped, {len(status['failed'])} failed"
    )
    return status
