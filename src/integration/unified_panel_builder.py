"""Unified panel construction (Phase 2, roadmap §5.3, §6.3).

Fuses the six source families into one row-per-(player_id, season) panel. The FBref
stat tables are the backbone (only source spanning all 9 leagues × 4 seasons); every
other source attaches by the Transfermarkt ``player_id`` assigned via
:class:`src.data.name_resolver.PlayerIDResolver`.

Pipeline: ``load_fbref_stats`` (11-table collision-safe merge) → ``resolve_backbone``
(→ player_id) → ``collapse_split_season`` (→ unique (player_id, season)) →
``attach_*`` (xG / MV / contract / FIFA / league meta) → ``finalize_panel``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src.data.name_resolver import (
    PlayerIDResolver,
    initials_form,
    normalize_club,
    normalize_name,
    synthetic_player_id,
)
from src.utils.constants import (
    FBREF_DROP_DUP_COLS,
    FBREF_LEAGUE_IDS,
    FBREF_SEASON_TO_CANONICAL,
    FBREF_SEASONS,
    FBREF_TABLE_PRIORITY,
    FBREF_TABLE_STAT_RENAME,
    MIN_MINUTES_LOWER,
    MIN_MINUTES_TOP5,
    POSITION_PRIMARY_MAP,
    SEASON_END_DATES,
    TM_LEAGUE_COMPETITION_IDS,
    TOP5_LEAGUES,
)
from src.utils.io import load_lookup_csv, load_parquet, project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Identity columns carried from `standard` (the authoritative table). The 11 tables are
# joined on a composite string key built from these (NaN-safe; `born` disambiguates
# same-name same-club namesakes, e.g. two "João Mendes" at Vitória Guimarães).
_ID_COLS = ["league", "season", "team", "player", "nation", "pos", "age", "born"]
_KEY_FIELDS = ["league", "season", "team", "player", "nation", "born"]

# Per-90 columns whose value is recomputed from a summed counting base during
# split-season collapse (per the "recompute from totals, not averaged per-90s" rule).
_PER90_BASE: dict[str, str] = {
    "goals_per_90": "goals",
    "assists_per_90": "assists",
    "goals_plus_assists_per_90": "goals_plus_assists",
    "non_penalty_goals_per_90": "non_penalty_goals",
    "shots_per_90": "shots",
    "shots_on_target_per_90": "shots_on_target",
    "sca_per_90": "shot_creating_actions",
    "gca_per_90": "goal_creating_actions",
    "goals_against_per_90": "goals_against",
    "psxg_per_90": "psxg",
    "gk_sweeper_per_90": "gk_sweeper_actions",
    "plus_minus_per_90": "plus_minus",
}


def league_slug(canonical: str) -> str:
    """Filesystem slug used by the FBref scraper: ``'Süper Lig'`` → ``'super_lig'``."""
    return canonical.lower().replace("ü", "u").replace(" ", "_")


def _snake(s: str) -> str:
    """Snake-case a raw FBref column for the namespaced fallback name."""
    s = (
        s.replace("+/-", " plus_minus ").replace("%", " pct ")
        .replace("/90", " per_90 ").replace("/", " per ").replace("#", " num ")
    )
    s = s.lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _primary_position(raw) -> str | None:
    """First-token position → GK/DEF/MID/FWD; ``None`` (logged by caller) if unknown."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw).strip() == "":
        return None
    return POSITION_PRIMARY_MAP.get(str(raw).split(",")[0].strip())


def _merge_key(df: pd.DataFrame) -> pd.Series:
    """NaN-safe composite join key from the identity fields."""
    parts = [df[c].astype("string").fillna("") for c in _KEY_FIELDS]
    key = parts[0]
    for p in parts[1:]:
        key = key.str.cat(p, sep="|")
    return key


def _load_one_stat_table(path: Path, table: str) -> pd.DataFrame:
    """Load one FBref stat table; rename curated cols, namespace the rest, drop dups.

    Returns ``_mkey`` + (identity cols, standard only) + renamed stats. Deduplicates on
    ``_mkey`` (keeps first, logged) so a truly identical duplicate row can never explode
    the downstream left-join.
    """
    df = load_parquet(path)
    rename = FBREF_TABLE_STAT_RENAME.get(table, {})
    drop = FBREF_DROP_DUP_COLS.get(table, frozenset())
    new_names: dict[str, str] = {}
    for col in df.columns:
        if col in _ID_COLS or col in drop:
            continue  # identity cols + expected dups handled separately
        new_names[col] = rename.get(col) or f"{table}__{_snake(col)}"

    out = df.copy()
    out["_mkey"] = _merge_key(out)
    keep = ["_mkey"]
    if table == "standard":
        keep += [c for c in _ID_COLS if c in out.columns]
    keep += list(new_names.keys())
    out = out[keep].rename(columns=new_names)

    n_dup = int(out["_mkey"].duplicated().sum())
    if n_dup:
        logger.warning("%s: %d duplicate identity row(s) collapsed (kept first)", table, n_dup)
        out = out.drop_duplicates("_mkey", keep="first")
    return out


def load_fbref_stats(
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    fbref_root: Path | None = None,
) -> pd.DataFrame:
    """Merge the 11 FBref stat tables into one wide frame per (player, team, season).

    ``standard`` is the authoritative backbone; the other 10 tables left-join onto it
    (so the player set = standard's). Curated columns get clean canonical names; the
    long tail is namespaced ``<table>__<col>``. Raises on any UNEXPECTED duplicate
    column (no silent overwrite); expected duplicates are pre-dropped and logged.
    """
    leagues = leagues or list(FBREF_LEAGUE_IDS.keys())
    seasons = seasons or list(FBREF_SEASONS)
    root = fbref_root or (project_root() / "data" / "raw" / "fbref")

    frames: list[pd.DataFrame] = []
    missing_tables: list[str] = []
    excluded_playing_time = 0
    for lg in leagues:
        slug = league_slug(lg)
        for sc in seasons:
            std_path = root / slug / sc / "standard.parquet"
            if not std_path.exists():
                missing_tables.append(f"{slug}/{sc}/standard")
                continue
            wide = _load_one_stat_table(std_path, "standard")
            std_keys = set(wide["_mkey"])

            for table in FBREF_TABLE_PRIORITY[1:]:
                path = root / slug / sc / f"{table}.parquet"
                if not path.exists():
                    missing_tables.append(f"{slug}/{sc}/{table}")
                    continue
                sub = _load_one_stat_table(path, table)
                if table == "playing_time":
                    excluded_playing_time += len(set(sub["_mkey"]) - std_keys)
                overlap = (set(sub.columns) - {"_mkey"}) & (set(wide.columns) - {"_mkey"})
                if overlap:
                    raise ValueError(
                        f"Unexpected duplicate column(s) {sorted(overlap)} merging "
                        f"'{table}' for {slug}/{sc} — add to FBREF_DROP_DUP_COLS or curate"
                    )
                n_before = len(wide)
                wide = wide.merge(sub, on="_mkey", how="left")
                assert len(wide) == n_before, f"row explosion merging {table} {slug}/{sc}"

            wide = wide.drop(columns="_mkey")
            wide["season"] = FBREF_SEASON_TO_CANONICAL.get(sc, sc)
            wide["league"] = lg
            frames.append(wide)

    if missing_tables:
        logger.warning("FBref tables missing for %d (league-season-table): %s%s",
                       len(missing_tables), ", ".join(missing_tables[:8]),
                       " ..." if len(missing_tables) > 8 else "")
    if excluded_playing_time:
        logger.info("Excluded %d playing_time-only player-rows absent from standard (left-join)",
                    excluded_playing_time)
    if not frames:
        raise FileNotFoundError("No FBref standard.parquet files found for the requested slice")

    stats = pd.concat(frames, ignore_index=True)
    stats["primary_position"] = stats["pos"].map(_primary_position)
    stats = stats.rename(columns={"pos": "detailed_position"})
    n_bad = int(stats["primary_position"].isna().sum())
    if n_bad:
        logger.warning("primary_position null for %d row(s) (unmapped FBref pos code)", n_bad)

    logger.info("FBref stats merged: %d rows × %d cols (%d league × %d season)",
                len(stats), stats.shape[1], stats["league"].nunique(), stats["season"].nunique())
    return stats


def resolve_backbone(stats: pd.DataFrame, resolver: PlayerIDResolver) -> pd.DataFrame:
    """Resolve every backbone row to a ``player_id`` (synthetic id on no-TM-match)."""
    resolved = resolver.resolve_frame(
        stats, name_col="player", birth_year_col="born", nationality_col="nation",
        team_col="team", season_col="season", source="fbref",
    )
    mask = resolved["player_id"].isna()
    if mask.any():
        resolved.loc[mask, "player_id"] = [
            synthetic_player_id(normalize_name(nm), by, nat)
            for nm, by, nat in zip(
                resolved.loc[mask, "player"], resolved.loc[mask, "born"], resolved.loc[mask, "nation"]
            )
        ]
        resolved.loc[mask, "resolve_method"] = "synthetic"
    resolved["data_richness"] = np.where(
        resolved["resolve_method"].eq("synthetic"), "synthetic_basic", "full"
    )
    return resolved


def split_id_collisions(resolved: pd.DataFrame) -> pd.DataFrame:
    """Un-merge distinct players that share one Transfermarkt id (mononym magnet).

    A single-token TM name like "Gabriel" scores 100 under token_set_ratio against every
    "Gabriel <surname>", so multiple distinct players can collapse onto one id and be
    wrongly summed. Here, any real player_id mapping to >1 distinct normalized name keeps
    the id for the dominant-minutes name; the others are re-assigned deterministic
    synthetic ids (kept, flagged, never summed into the wrong player).
    """
    resolved = resolved.copy()
    nname = resolved["player"].map(normalize_name)
    is_real = ~resolved["player_id"].astype("string").str.startswith("synthetic_")
    distinct = nname[is_real].groupby(resolved.loc[is_real, "player_id"]).nunique()
    contaminated = set(distinct[distinct > 1].index)
    if not contaminated:
        return resolved

    minutes = resolved["minutes_played"].fillna(0)
    n_reassigned = 0
    for pid in contaminated:
        rows = resolved[(resolved["player_id"] == pid)].index
        names = nname.loc[rows]
        dominant = minutes.loc[rows].groupby(names).sum().idxmax()
        for idx in rows:
            if names.loc[idx] != dominant:
                resolved.at[idx, "player_id"] = synthetic_player_id(
                    names.loc[idx], resolved.at[idx, "born"], resolved.at[idx, "nation"]
                )
                resolved.at[idx, "resolve_method"] = "synthetic_split"
                resolved.at[idx, "data_richness"] = "synthetic_basic"
                n_reassigned += 1
    logger.info("ID-collision guard: %d player_id(s) shared by distinct names → %d rows re-assigned synthetic ids",
                len(contaminated), n_reassigned)
    return resolved


def split_minutes_overflow(resolved: pd.DataFrame, cap: int = 3800) -> pd.DataFrame:
    """Un-merge same-name namesakes that the name guard misses (identical mononyms).

    No real player exceeds ~3,800 minutes in a season (≤38 games even across a mid-season
    transfer), so a (player_id, season) group above the cap is two distinct same-name
    players (e.g. two Portuguese "Vitinha" born 2000). Keep the max-minutes stint with the
    id; re-assign the rest club-salted synthetic ids (kept, flagged for the override list).
    """
    resolved = resolved.copy()
    minutes = resolved["minutes_played"].fillna(0)
    grp_min = minutes.groupby([resolved["player_id"], resolved["season"]]).transform("sum")
    over = grp_min > cap
    if not over.any():
        return resolved

    nname = resolved["player"].map(normalize_name)
    n_reassigned = 0
    for (pid, ssn), idxs in resolved[over].groupby(["player_id", "season"]).groups.items():
        keep = minutes.loc[idxs].idxmax()  # max-minutes stint keeps the id
        for idx in idxs:
            if idx == keep:
                continue
            salted = f"{nname.loc[idx]}|{normalize_club(resolved.at[idx, 'team'])}"
            resolved.at[idx, "player_id"] = synthetic_player_id(
                salted, resolved.at[idx, "born"], resolved.at[idx, "nation"]
            )
            resolved.at[idx, "resolve_method"] = "synthetic_split"
            resolved.at[idx, "data_richness"] = "synthetic_basic"
            n_reassigned += 1
    logger.info("Minutes-overflow guard: %d row(s) >%d-min/season re-assigned (same-name namesakes)",
                n_reassigned, cap)
    return resolved


def _classify_stat_columns(cols: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Partition numeric stat columns into (sum, weighted-avg, per-90) for collapse."""
    summ, wavg, per90 = [], [], []
    avg_markers = ("_pct", "_per_sot", "avg", "per_match", "_per_90")
    avg_exact = {"goals_per_shot", "goals_per_shot_on_target", "on_off"}
    for c in cols:
        if c.endswith("_per_90"):
            per90.append(c)
        elif c.endswith("_pct") or c.endswith("_per_sot") or c in avg_exact or "avg" in c or "per_match" in c:
            wavg.append(c)
        else:
            summ.append(c)
    return summ, wavg, per90


def collapse_split_season(resolved: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-stint players to one row per (player_id, season) (D-15).

    Counting stats are summed; percentages/averages are minutes-weighted; per-90s are
    recomputed from summed totals where a base exists (else minutes-weighted); club /
    league / position come from the max-minutes stint. ``is_split_season`` flags the
    collapsed rows; ``meets_min_minutes`` is recomputed from total season minutes.
    """
    key = ["player_id", "season"]
    non_stat = set(_ID_COLS) | {
        "player_id", "resolve_method", "resolve_score", "data_richness",
        "primary_position", "detailed_position", "minutes_played",
    }
    numeric_cols = [c for c in resolved.columns
                    if c not in non_stat and pd.api.types.is_numeric_dtype(resolved[c])]
    summ, wavg, per90 = _classify_stat_columns(numeric_cols)

    dup_mask = resolved.duplicated(key, keep=False)
    singles = resolved[~dup_mask].copy()
    singles["is_split_season"] = False

    multi = resolved[dup_mask]
    n_groups = multi.groupby(key, sort=False).ngroups
    logger.info("Split-season collapse: %d stint-rows → %d players", len(multi), n_groups)

    collapsed: list[dict] = []
    for _, g in multi.groupby(key, sort=False):
        g = g.sort_values("minutes_played", ascending=False, na_position="last")
        rep = g.iloc[0].to_dict()  # max-minutes stint → context columns
        minutes = g["minutes_played"].fillna(0).to_numpy(dtype=float)
        total_min = float(minutes.sum())

        rep["minutes_played"] = g["minutes_played"].sum(min_count=1)
        for c in summ:
            rep[c] = g[c].sum(min_count=1)
        for c in wavg:
            vals = g[c].to_numpy(dtype=float)
            ok = ~np.isnan(vals)
            rep[c] = (np.average(vals[ok], weights=minutes[ok])
                      if ok.any() and minutes[ok].sum() > 0 else np.nan)
        nineties = rep.get("matches_90s")
        for c in per90:
            base = _PER90_BASE.get(c)
            if base is not None and base in rep and nineties and nineties > 0:
                rep[c] = rep[base] / nineties
            else:  # minutes-weighted fallback
                vals = g[c].to_numpy(dtype=float)
                ok = ~np.isnan(vals)
                rep[c] = (np.average(vals[ok], weights=minutes[ok])
                          if ok.any() and minutes[ok].sum() > 0 else np.nan)
        rep["is_split_season"] = True
        collapsed.append(rep)

    out = pd.concat([singles, pd.DataFrame(collapsed)], ignore_index=True) if collapsed else singles
    out = out.copy()  # de-fragment after the dict→frame concat

    # recompute the min-minutes flag from TOTAL season minutes
    is_top5 = out["league"].isin(TOP5_LEAGUES)
    out["min_minutes_threshold"] = np.where(is_top5, MIN_MINUTES_TOP5, MIN_MINUTES_LOWER)
    out["meets_min_minutes"] = out["minutes_played"].fillna(0) >= out["min_minutes_threshold"]

    dup_remaining = int(out.duplicated(key).sum())
    assert dup_remaining == 0, f"{dup_remaining} duplicate (player_id, season) after collapse"
    out = out.rename(columns={"team": "club"})
    logger.info("Collapsed panel: %d unique (player_id, season) rows", len(out))
    return out


# ── secondary-source matching (against the resolved panel, for consistency) ──────
class _PanelMatcher:
    """Match secondary-source rows to panel player_id by name+year or season+name.

    Carries an initials index so short names (FIFA "K. Mbappé") match full FBref names
    ("Kylian Mbappé"). A match is accepted only when unambiguous (one distinct id).
    """

    def __init__(self, panel: pd.DataFrame) -> None:
        self.ex_ny: dict[tuple[str, int | None], list[str]] = {}
        self.in_ny: dict[tuple[str, int | None], list[str]] = {}
        self.ex_sn: dict[tuple[str, str], list[str]] = {}
        self.in_sn: dict[tuple[str, str], list[str]] = {}
        for pid, nm, by, ssn in zip(
            panel["player_id"], panel["player_name_fbref"], panel["born"], panel["season"]
        ):
            b = None if pd.isna(by) else int(by)
            nn = normalize_name(nm)
            ini = initials_form(nn)
            self.ex_ny.setdefault((nn, b), []).append(pid)
            self.in_ny.setdefault((ini, b), []).append(pid)
            self.ex_sn.setdefault((ssn, nn), []).append(pid)
            self.in_sn.setdefault((ssn, ini), []).append(pid)

    @staticmethod
    def _uniq(ids: list[str] | None) -> str | None:
        return ids[0] if ids and len(set(ids)) == 1 else None

    def by_name_year(self, name, birth_year) -> str | None:
        b = None if pd.isna(birth_year) else int(birth_year)
        nn = normalize_name(name)
        return self._uniq(self.ex_ny.get((nn, b))) or self._uniq(self.in_ny.get((initials_form(nn), b)))

    def by_season_name(self, season, name) -> str | None:
        nn = normalize_name(name)
        return self._uniq(self.ex_sn.get((season, nn))) or self._uniq(self.in_sn.get((season, initials_form(nn))))


def _match_understat_to_panel(und: pd.DataFrame, panel: pd.DataFrame,
                              matcher: "_PanelMatcher") -> list[str | None]:
    """Match Understat rows to panel player_id: (season,club)+name exact→fuzzy, then a
    club-independent (season, name) fallback that recovers club-name divergence (e.g.
    "Wolverhampton Wanderers" vs "Wolves") while staying season-scoped and unambiguous."""
    by_club: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for pid, nm, club, ssn in zip(
        panel["player_id"], panel["player_name_fbref"], panel["club"], panel["season"]
    ):
        by_club.setdefault((ssn, normalize_club(club)), []).append((normalize_name(nm), pid))

    out: list[str | None] = []
    for nm, team, ssn in zip(und["player"], und["team"], und["_season_c"]):
        cand = by_club.get((ssn, normalize_club(team)), [])
        nn = normalize_name(nm)
        exact = [pid for cnm, pid in cand if cnm == nn]
        if len(set(exact)) == 1:
            out.append(exact[0])
            continue
        if cand:
            m = process.extractOne(nn, [c[0] for c in cand],
                                   scorer=fuzz.token_set_ratio, score_cutoff=85)
            if m is not None:
                out.append(cand[m[2]][1])
                continue
        out.append(matcher.by_season_name(ssn, nm))  # club-independent fallback (or None)
    return out


def _load_understat(seasons: list[str] | None = None,
                    understat_root: Path | None = None) -> pd.DataFrame:
    """Concatenate the Understat top-5 player-season tables (xG source, all seasons)."""
    seasons = seasons or list(FBREF_SEASONS)
    root = understat_root or (project_root() / "data" / "raw" / "understat")
    frames = []
    for lg in TOP5_LEAGUES:
        for sc in seasons:
            path = root / league_slug(lg) / f"{sc}.parquet"
            if path.exists():
                frames.append(load_parquet(path))
    if not frames:
        return pd.DataFrame(columns=["player", "team", "season", "xg", "xa", "np_xg"])
    return pd.concat(frames, ignore_index=True)


# ── attach stages ────────────────────────────────────────────────────────────────
def attach_xg(panel: pd.DataFrame, kaggle: pd.DataFrame,
              understat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach xG with source priority Kaggle(24-25 top5) > Understat(top5) > NaN (P2-D3).

    Returns ``(panel, understat_orphans)``. ``xag`` is filled from Kaggle only;
    ``understat_xa`` is kept as a separate column (coalesce deferred to Phase 4).
    """
    matcher = _PanelMatcher(panel)

    # Kaggle 2024-25 top-5 (sum across mid-season stints)
    kag = kaggle.copy()
    kag["player_id"] = [
        matcher.by_name_year(nm, by) for nm, by in zip(kag["player_name"], kag["birth_year"])
    ]
    kag = kag.dropna(subset=["player_id"]).groupby("player_id")[["xg", "npxg", "xag"]].sum(min_count=1)
    kag = kag.rename(columns={"xg": "kaggle_xg", "npxg": "kaggle_npxg", "xag": "kaggle_xag"})
    kag = kag.reset_index()
    kag["season"] = "2024-25"

    # Understat (all top-5 seasons), matched against the panel
    und = understat.copy()
    und["_season_c"] = und["season"].map(FBREF_SEASON_TO_CANONICAL)
    und["player_id"] = _match_understat_to_panel(und, panel, matcher)
    orphans = und[und["player_id"].isna()][["player", "team", "_season_c"]].rename(
        columns={"_season_c": "season"}
    )
    n_orphan = len(orphans)
    n_total = len(und)
    logger.info("Understat: %d/%d rows matched, %d orphans (%.1f%%)",
                n_total - n_orphan, n_total, n_orphan, 100 * n_orphan / max(n_total, 1))
    if n_total and n_orphan / n_total > 0.10:
        logger.warning("Understat orphan rate %.1f%% exceeds 10%% — possible FBref backbone "
                       "coverage gap; see reports/name_resolution_audit.md", 100 * n_orphan / n_total)
    und_agg = (
        und.dropna(subset=["player_id"]).groupby(["player_id", "_season_c"])[["xg", "np_xg", "xa"]]
        .sum(min_count=1).reset_index()
        .rename(columns={"_season_c": "season", "xg": "understat_xg",
                         "np_xg": "understat_npxg", "xa": "understat_xa"})
    )

    n = len(panel)
    panel = panel.merge(kag, on=["player_id", "season"], how="left")
    panel = panel.merge(und_agg, on=["player_id", "season"], how="left")
    assert len(panel) == n, "row count changed attaching xG"

    is_top5 = panel["league"].isin(TOP5_LEAGUES)
    is2425 = panel["season"].eq("2024-25")
    kag_ok = is2425 & is_top5 & panel["kaggle_xg"].notna()
    panel["xg"] = np.where(kag_ok, panel["kaggle_xg"],
                           np.where(is_top5 & panel["understat_xg"].notna(), panel["understat_xg"], np.nan))
    panel["npxg"] = np.where(is2425 & is_top5 & panel["kaggle_npxg"].notna(), panel["kaggle_npxg"],
                             np.where(is_top5 & panel["understat_npxg"].notna(), panel["understat_npxg"], np.nan))
    panel["xag"] = np.where(is2425 & is_top5, panel["kaggle_xag"], np.nan)
    panel["xg_source"] = np.where(panel["xg"].isna(), "none",
                                  np.where(kag_ok, "kaggle", "understat"))
    panel = panel.drop(columns=["kaggle_xg", "kaggle_npxg", "kaggle_xag", "understat_npxg"])
    return panel, orphans


def attach_market_value(panel: pd.DataFrame, tm_seasons: pd.DataFrame) -> pd.DataFrame:
    """Left-join the Stage-2 target (market_value_eur) on (player_id, season) (D-05/D-16)."""
    s = tm_seasons.copy()
    s["player_id"] = s["tm_player_id"].astype(str)
    s = s[["player_id", "season", "market_value_eur", "snapshot_date"]].rename(
        columns={"snapshot_date": "market_value_date"}
    )
    n = len(panel)
    panel = panel.merge(s, on=["player_id", "season"], how="left")
    assert len(panel) == n, "row explosion attaching market value"
    panel["log_market_value"] = np.log1p(panel["market_value_eur"])
    cov = panel.groupby("season")["market_value_eur"].apply(lambda x: x.notna().sum())
    logger.info("MV coverage by season: %s", cov.to_dict())
    return panel


def attach_contract(panel: pd.DataFrame, tm_players: pd.DataFrame) -> pd.DataFrame:
    """Attach contract + DOB + WC proxy from Transfermarkt (per player_id)."""
    tp = tm_players.copy()
    tp["player_id"] = tp["tm_player_id"].astype(str)
    sub = tp[["player_id", "contract_expiration_date", "date_of_birth",
              "has_international_caps", "current_club_name"]].drop_duplicates("player_id")
    panel = panel.merge(sub, on="player_id", how="left")
    panel["contract_end_date"] = pd.to_datetime(panel["contract_expiration_date"], errors="coerce")
    panel["season_end_date"] = pd.to_datetime(panel["season"].map(SEASON_END_DATES))
    panel["contract_remaining_months"] = (
        (panel["contract_end_date"] - panel["season_end_date"]).dt.days / 30.44
    ).clip(lower=0)
    panel["has_contract_date"] = panel["contract_end_date"].notna()
    panel["worldcup_25_squad"] = panel["has_international_caps"].fillna(False).astype(bool)
    return panel


def attach_fifa(panel: pd.DataFrame, fifa: pd.DataFrame) -> pd.DataFrame:
    """Attach FIFA overall/potential per (player_id, season) (D-17). FC25 potential null."""
    f = fifa.copy()
    f["birth_year"] = pd.to_datetime(f["date_of_birth"], errors="coerce").dt.year
    matcher = _PanelMatcher(panel)
    f["player_id"] = [
        matcher.by_name_year(nm, by) if pd.notna(by) else matcher.by_season_name(ssn, nm)
        for nm, by, ssn in zip(f["player_name"], f["birth_year"], f["season"])
    ]
    agg = (
        f.dropna(subset=["player_id"]).groupby(["player_id", "season"])
        .agg(fifa_rating=("overall", "max"), fifa_potential=("potential", "max")).reset_index()
    )
    n = len(panel)
    panel = panel.merge(agg, on=["player_id", "season"], how="left")
    assert len(panel) == n, "row explosion attaching FIFA"
    return panel


def attach_league_meta(panel: pd.DataFrame, competitions: pd.DataFrame) -> pd.DataFrame:
    """Attach league tier / UEFA coefficient / value multiplier (joined via competition_id)."""
    comp = competitions.set_index("competition_id")
    cid = panel["league"].map(TM_LEAGUE_COMPETITION_IDS)
    panel["league_tier"] = cid.map(comp["league_tier"]).astype("Int8")
    panel["uefa_coefficient"] = cid.map(comp["uefa_coefficient"])
    panel["league_value_multiplier"] = cid.map(comp["league_value_multiplier"])
    return panel


# ── final schema ─────────────────────────────────────────────────────────────────
_FINAL_ORDER = [
    "player_id", "player_name", "birth_year", "birth_date", "nationality", "continent_group",
    "season", "season_end_year", "season_end_date", "league", "league_tier",
    "uefa_coefficient", "league_value_multiplier", "club", "is_loan", "is_split_season",
    "age_at_season_end", "age_precision", "primary_position", "detailed_position",
    "minutes_played", "meets_min_minutes", "min_minutes_threshold",
    "xg", "xag", "npxg", "understat_xa", "xg_source",
    "market_value_eur", "market_value_date", "log_market_value",
    "contract_end_date", "contract_remaining_months", "has_contract_date",
    "fifa_rating", "fifa_potential", "worldcup_25_squad",
    "resolve_method", "resolve_score", "data_richness",
]


def finalize_panel(panel: pd.DataFrame, resolver: PlayerIDResolver) -> pd.DataFrame:
    """Compute identity/age columns, order the schema, enforce dtypes, assert uniqueness."""
    continent = load_lookup_csv("continent_map")
    panel["nationality"] = panel["nation"].map(resolver.normalize_nationality)
    panel["continent_group"] = panel["nationality"].map(
        lambda n: continent.get(n, "Other") if isinstance(n, str) else "Other"
    )
    panel["birth_date"] = pd.to_datetime(panel["date_of_birth"], errors="coerce")
    panel["season_end_year"] = panel["season_end_date"].dt.year.astype("Int16")

    exact = panel["birth_date"].notna()
    age_exact = (panel["season_end_date"] - panel["birth_date"]).dt.days / 365.25
    age_year = panel["season_end_year"].astype("float") - panel["born"].astype("float")
    panel["age_at_season_end"] = np.where(exact, age_exact, age_year)
    panel["age_precision"] = np.where(exact, "exact", "year_only")

    panel = panel.rename(columns={"player_name_fbref": "player_name"})
    panel["is_loan"] = False  # D-15 loan detection deferred (current TM snapshot lacks it)

    # order: §5.3 schema first, then namespaced long-tail stat columns
    leading = [c for c in _FINAL_ORDER if c in panel.columns]
    stat_tail = [c for c in panel.columns if c not in set(leading)
                 and c not in {"nation", "born", "age", "date_of_birth",
                               "contract_expiration_date", "has_international_caps",
                               "current_club_name", "player_name"}]
    # Drop FBref stat columns soccerdata never delivered (100% null across the panel):
    # the extended-table gap — tackles/passing/possession/creation/PSxG. Keeping them
    # only breeds NaN confusion (see decisions_log P2-D6 / memory BLOCKER note).
    empty = [c for c in stat_tail if panel[c].isna().all()]
    if empty:
        logger.info("Dropped %d all-null FBref stat columns (soccerdata extended-stat gap)", len(empty))
        stat_tail = [c for c in stat_tail if c not in empty]
    panel = panel[leading + stat_tail]

    assert panel.duplicated(["player_id", "season"]).sum() == 0, "duplicate (player_id, season)"
    logger.info("Final panel: %d rows × %d cols", len(panel), panel.shape[1])
    return panel


def build_unified_panel(leagues: list[str] | None = None,
                        seasons: list[str] | None = None,
                        manual_overrides: pd.DataFrame | None = None,
                        ) -> tuple[pd.DataFrame, PlayerIDResolver, pd.DataFrame]:
    """Run the full Phase-2 build → (unified_panel, resolver, understat_orphans)."""
    interim = project_root() / "data" / "interim"
    tm_players = load_parquet(interim / "tm_players.parquet")
    resolver = PlayerIDResolver(tm_players, manual_overrides=manual_overrides)

    stats = load_fbref_stats(leagues, seasons)
    resolved = resolve_backbone(stats, resolver)
    resolved = split_id_collisions(resolved)
    resolved = split_minutes_overflow(resolved)
    panel = collapse_split_season(resolved)
    panel = panel.rename(columns={"player": "player_name_fbref"})

    panel, orphans = attach_xg(
        panel, load_parquet(interim / "kaggle_2024_25_clean.parquet"), _load_understat(seasons)
    )
    panel = attach_market_value(panel, load_parquet(interim / "tm_player_seasons.parquet"))
    panel = attach_contract(panel, tm_players)
    panel = attach_fifa(panel, load_parquet(interim / "fifa_ratings.parquet"))
    panel = attach_league_meta(panel, load_parquet(interim / "tm_competitions.parquet"))
    panel = finalize_panel(panel, resolver)
    return panel, resolver, orphans
