"""Pure-pandas EDA computations (no plotting).

Every number the notebook displays and ``eda_findings.md`` cites is produced here, so
the narrative is reproducible and the notebook stays logic-free. ``summarize(panel)``
rolls everything into one dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.eda.style import LEAGUE_ORDER, POSITION_ORDER, SEASON_ORDER, TOP5_LEAGUES

# ── column taxonomy ──────────────────────────────────────────────────────────
_IDENTITY = ["player_id", "player_name", "birth_year", "birth_date", "nationality", "continent_group"]
_SEASON_CTX = ["season", "season_end_year", "season_end_date", "league", "league_tier",
               "uefa_coefficient", "league_value_multiplier", "club", "is_loan",
               "is_split_season", "age_at_season_end", "age_precision"]
_POSITION = ["primary_position", "detailed_position"]
_TARGET = ["market_value_eur", "market_value_date", "log_market_value"]
_CONTRACT = ["contract_end_date", "contract_remaining_months", "has_contract_date"]
_EXTERNAL = ["fifa_rating", "fifa_potential", "worldcup_25_squad"]
_XG = ["xg", "xag", "npxg", "understat_xa", "understat_xg", "xg_source"]
_PROVENANCE = ["resolve_method", "resolve_score", "data_richness", "minutes_played",
               "meets_min_minutes", "min_minutes_threshold"]

_GK_STATS = {
    "goals_against", "goals_against_per_90", "shots_on_target_against", "saves", "save_pct",
    "gk_wins", "gk_draws", "gk_losses", "clean_sheets", "clean_sheet_pct",
    "gk_penalty_attempts", "gk_penalty_allowed", "gk_penalty_saved", "gk_penalty_missed",
    "gk_penalty_save_pct",
}
_SOURCE_GAP = {"xg", "xag", "npxg", "understat_xa", "understat_xg", "fifa_rating", "fifa_potential"}


def categorize_columns(panel: pd.DataFrame) -> dict[str, list[str]]:
    """Group the 207 columns into logical blocks for the overview section."""
    known = set(_IDENTITY + _SEASON_CTX + _POSITION + _TARGET + _CONTRACT
                + _EXTERNAL + _XG + _PROVENANCE)
    perf = [c for c in panel.columns if c not in known]
    return {
        "identity": [c for c in _IDENTITY if c in panel.columns],
        "season_context": [c for c in _SEASON_CTX if c in panel.columns],
        "position": [c for c in _POSITION if c in panel.columns],
        "target": [c for c in _TARGET if c in panel.columns],
        "contract": [c for c in _CONTRACT if c in panel.columns],
        "external": [c for c in _EXTERNAL if c in panel.columns],
        "xg": [c for c in _XG if c in panel.columns],
        "provenance": [c for c in _PROVENANCE if c in panel.columns],
        "performance_clean": [c for c in perf if "__" not in c],
        "performance_namespaced": [c for c in perf if "__" in c],
    }


# ── Section 2: target ────────────────────────────────────────────────────────
def target_distribution(panel: pd.DataFrame) -> dict:
    """Skew/kurtosis of raw vs log MV + a Shapiro-Wilk normality stat on the log."""
    raw = panel["market_value_eur"].dropna()
    logv = np.log1p(raw)
    sample = logv.sample(min(5000, len(logv)), random_state=42)
    sw_stat, sw_p = stats.shapiro(sample)
    return {
        "n": int(len(raw)),
        "skew_raw": float(stats.skew(raw)),
        "kurtosis_raw": float(stats.kurtosis(raw)),
        "skew_log": float(stats.skew(logv)),
        "kurtosis_log": float(stats.kurtosis(logv)),
        "shapiro_log_stat": float(sw_stat),
        "shapiro_log_p": float(sw_p),
        "median_eur": float(raw.median()),
    }


def mv_medians(panel: pd.DataFrame) -> dict:
    """Median MV by season / league / position, plus premium ratios and YoY inflation."""
    mv = "market_value_eur"
    by_season = panel.groupby("season")[mv].median().reindex(SEASON_ORDER)
    by_league = panel.groupby("league")[mv].median().reindex(LEAGUE_ORDER)
    by_pos_med = panel.groupby("primary_position")[mv].median().reindex(POSITION_ORDER)
    by_pos_mean = panel.groupby("primary_position")[mv].mean().reindex(POSITION_ORDER)
    yoy = by_season.pct_change().dropna() * 100
    return {
        "by_season": by_season,
        "by_league": by_league,
        "by_position_median": by_pos_med,
        "by_position_mean": by_pos_mean,
        "yoy_pct": yoy,
        "premium_pl_vs_superlig": float(by_league["Premier League"] / by_league["Süper Lig"]),
        "premium_laliga_vs_superlig": float(by_league["La Liga"] / by_league["Süper Lig"]),
    }


# ── Section 3: missingness ───────────────────────────────────────────────────
def classify_missing(col: str, panel: pd.DataFrame) -> str:
    """Label a column's missingness as source_gap / structural / sparse / full."""
    if "psxg" in col or col in _SOURCE_GAP:
        return "source_gap"
    if col in _GK_STATS:
        return "structural_gk"          # null for outfield rows
    if col in _TARGET:
        return "target_sparse"
    if col in _CONTRACT:
        return "contract_sparse"
    cats = categorize_columns(panel)
    if col in cats["performance_clean"] or col in cats["performance_namespaced"]:
        return "structural_outfield"     # null for GK rows
    return "other"


def missingness(panel: pd.DataFrame) -> dict:
    """Per-column null rate + kind label, and structural-vs-source-gap proof points."""
    null_rate = panel.isna().mean().sort_values(ascending=False)
    kinds = {c: classify_missing(c, panel) for c in null_rate.index}
    gk = panel[panel["primary_position"] == "GK"]
    # proof: the surviving structural gap is GK stats (null for outfield) collapsing to ~0
    # within GK; source-gap nulls (xG lower-4, FC25 potential) are by-design.
    proof = {
        "saves_null_overall": float(panel["saves"].isna().mean()),
        "saves_null_within_gk": float(gk["saves"].isna().mean()),
        "xg_null_lower4": float(panel[~panel["league"].isin(TOP5_LEAGUES)]["xg"].isna().mean()),
        "xg_null_top5": float(panel[panel["league"].isin(TOP5_LEAGUES)]["xg"].isna().mean()),
        "xag_null_overall": float(panel["xag"].isna().mean()),
        "fifa_potential_null_2425": float(panel[panel.season == "2024-25"]["fifa_potential"].isna().mean()),
        "contract_null_overall": float(panel["contract_remaining_months"].isna().mean()),
        "mv_null_overall": float(panel["market_value_eur"].isna().mean()),
    }
    return {"null_rate": null_rate, "kinds": kinds, "proof": proof}


# ── Section 4: position ──────────────────────────────────────────────────────
def position_stats(panel: pd.DataFrame) -> dict:
    counts = panel["primary_position"].value_counts().reindex(POSITION_ORDER)
    changers = panel.groupby("player_id")["primary_position"].nunique()
    n_changers = int((changers > 1).sum())
    by_league = (panel.groupby(["league", "primary_position"]).size()
                 .unstack(fill_value=0).reindex(LEAGUE_ORDER)[POSITION_ORDER])
    return {
        "counts": counts,
        "gk_pct": float(100 * counts["GK"] / counts.sum()),
        "n_position_changers": n_changers,
        "pct_position_changers": float(100 * n_changers / panel["player_id"].nunique()),
        "by_league": by_league,
    }


# ── Section 5: age ───────────────────────────────────────────────────────────
def _peak_age(sub: pd.DataFrame) -> float | None:
    s = sub.dropna(subset=["age_at_season_end", "market_value_eur"])
    if len(s) < 50:
        return None
    sm = lowess(s["market_value_eur"], s["age_at_season_end"], frac=0.5, return_sorted=True)
    return float(sm[sm[:, 1].argmax(), 0])


def age_stats(panel: pd.DataFrame) -> dict:
    age = panel["age_at_season_end"].dropna()
    peak = {pos: _peak_age(panel[panel.primary_position == pos]) for pos in POSITION_ORDER}
    return {
        "min": float(age.min()),
        "median": float(age.median()),
        "max": float(age.max()),
        "n_implausible": int(((age < 15) | (age > 43)).sum()),
        "peak_age_by_position": peak,
    }


# ── Section 8: xG calibration ────────────────────────────────────────────────
def xg_calibration(panel: pd.DataFrame) -> dict:
    df = panel.dropna(subset=["xg", "goals"])
    out = {"n": int(len(df)), "pearson_overall": float(df["xg"].corr(df["goals"])), "by_source": {}}
    for src in ["kaggle", "understat"]:
        s = df[df["xg_source"] == src]
        if len(s) > 10:
            out["by_source"][src] = {
                "n": int(len(s)),
                "pearson": float(s["xg"].corr(s["goals"])),
                "mean_resid_goals_minus_xg": float((s["goals"] - s["xg"]).mean()),
            }
    diff = (df["goals"] - df["xg"])
    over = df.assign(diff=diff).nlargest(5, "diff")[["player_name", "season", "goals", "xg"]]
    under = df.assign(diff=diff).nsmallest(5, "diff")[["player_name", "season", "goals", "xg"]]
    out["top_overperformers"] = over.to_dict("records")
    out["top_underperformers"] = under.to_dict("records")
    return out


# ── Section 9: provenance ────────────────────────────────────────────────────
def provenance_stats(panel: pd.DataFrame) -> dict:
    matched = {"exact", "exact_name_year", "fuzzy_nat", "fuzzy_team", "manual"}
    richness = (panel.assign(syn=panel.data_richness.eq("synthetic_basic"))
                .groupby(["league", "season"])["syn"].mean().mul(100)
                .unstack().reindex(LEAGUE_ORDER)[SEASON_ORDER])
    return {
        "resolve_method": panel["resolve_method"].value_counts(),
        "data_richness": panel["data_richness"].value_counts(),
        "age_precision": panel["age_precision"].value_counts(),
        "tm_matched_pct": float(100 * panel["resolve_method"].isin(matched).mean()),
        "synthetic_pct_by_league_season": richness,
    }


# ── Section 10: outliers ─────────────────────────────────────────────────────
def outliers(panel: pd.DataFrame) -> dict:
    top10 = panel.nlargest(10, "market_value_eur")[
        ["player_name", "season", "club", "market_value_eur", "age_at_season_end"]]
    low_min = panel[(panel.minutes_played < 100) & (panel.market_value_eur > 10e6)]
    return {
        "top10_mv": top10.to_dict("records"),
        "n_low_min_high_mv": int(len(low_min)),
        "low_min_examples": low_min.nlargest(5, "market_value_eur")[
            ["player_name", "season", "minutes_played", "market_value_eur"]].to_dict("records"),
        "max_minutes": float(panel["minutes_played"].max()),
        "n_over_3500_min": int((panel.minutes_played > 3500).sum()),
    }


# ── Section 6: case studies ──────────────────────────────────────────────────
def _find_player(panel: pd.DataFrame, name_contains: str) -> str | None:
    hits = panel[panel["player_name"].str.contains(name_contains, case=False, na=False)]
    if hits.empty:
        return None
    # prefer the player_id with the most seasons (the canonical career)
    return hits["player_id"].value_counts().idxmax()


def select_case_studies(panel: pd.DataFrame) -> dict:
    """Pick 4 archetype players (young riser / peak / late-career / cross-tier mover)."""
    archetypes = {
        "young_riser (Bellingham)": "Bellingham",
        "peak / cross-league (Kane)": "Kane",
        "late-career outlier (Modrić)": "Modri",
        "lower→top-5 mover (Minteh)": "Minteh",
    }
    out: dict[str, dict] = {}
    for label, needle in archetypes.items():
        pid = _find_player(panel, needle)
        if pid is None:
            continue
        rows = panel[panel.player_id == pid].sort_values("season")
        out[label] = {
            "player_id": pid,
            "name": rows["player_name"].iloc[0],
            "seasons": rows["season"].tolist(),
            "leagues": rows["league"].tolist(),
        }
    return out


# ── roll-up ──────────────────────────────────────────────────────────────────
def summarize(panel: pd.DataFrame) -> dict:
    """One dict with every headline number, for eda_findings.md."""
    return {
        "shape": tuple(panel.shape),
        "n_players": int(panel["player_id"].nunique()),
        "n_leagues": int(panel["league"].nunique()),
        "n_seasons": int(panel["season"].nunique()),
        "columns": {k: len(v) for k, v in categorize_columns(panel).items()},
        "target": target_distribution(panel),
        "mv": mv_medians(panel),
        "missing": missingness(panel),
        "position": position_stats(panel),
        "age": age_stats(panel),
        "xg": xg_calibration(panel),
        "provenance": provenance_stats(panel),
        "outliers": outliers(panel),
        "case_studies": select_case_studies(panel),
    }
