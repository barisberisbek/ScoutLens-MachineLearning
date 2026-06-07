"""Project-wide constants and lookup tables.

Single source of truth for magic numbers, league/position maps, stat lists, and
season splits. No business logic lives here beyond a pure position-normalization
helper. See PROJECT_ROADMAP.md §3, §6.3, §10 for provenance.
"""

from __future__ import annotations

# ── League name normalization ──────────────────────────────────────────────
# Kaggle / soccerdata sometimes prefix leagues with a country code
# (e.g. "eng Premier League"). Map raw names to canonical names.
LEAGUE_NAME_PREFIX_MAP: dict[str, str] = {
    "eng Premier League": "Premier League",
    "es La Liga": "La Liga",
    "de Bundesliga": "Bundesliga",
    "it Serie A": "Serie A",
    "fr Ligue 1": "Ligue 1",
    # Lower 4 — add alternate source formats here if they differ.
    "Eredivisie": "Eredivisie",
    "Liga Portugal": "Liga Portugal",
    "Belgian Pro League": "Belgian Pro League",
    "Süper Lig": "Süper Lig",
}

# ── Position normalization ──────────────────────────────────────────────────
# FBref emits short codes, possibly multi-position ("DF,MF"). Take the primary
# (first) code and map to the canonical GK/DEF/MID/FWD taxonomy.
POSITION_PRIMARY_MAP: dict[str, str] = {
    "GK": "GK",
    "DF": "DEF",
    "MF": "MID",
    "FW": "FWD",
}


def normalize_position(raw: str) -> str:
    """Normalize an FBref position string to a primary GK/DEF/MID/FWD label.

    Handles multi-position strings by taking the first component, e.g.
    "DF,MF" -> "DEF", "FW" -> "FWD". Raises ValueError on an unknown code so
    taxonomy drift is caught loudly rather than silently mislabeled.
    """
    if raw is None or str(raw).strip() == "":
        raise ValueError("normalize_position received an empty position string")
    first = str(raw).split(",")[0].strip()
    try:
        return POSITION_PRIMARY_MAP[first]
    except KeyError as exc:
        raise ValueError(f"Unknown position code: {first!r} (from {raw!r})") from exc


# ── Per-90 stats (§10.1) ────────────────────────────────────────────────────
PER_90_STATS: list[str] = [
    "goals", "assists", "shots", "shots_on_target",
    "xg", "xag", "npxg", "sca", "gca",
    "progressive_passes", "progressive_carries", "key_passes",
    "tackles", "interceptions", "blocks", "clearances",
    "aerial_won", "aerial_lost",
    "touches", "progressive_passes_received",
    "fouls_committed", "fouls_drawn",
    "saves", "goals_against", "psxg",
]

# ── Trajectory lag stats (§10.9) ────────────────────────────────────────────
LAG_STATS: list[str] = [
    "xg_per_90", "minutes_played", "progressive_passes_per_90",
    "xag_per_90", "tackles_per_90", "saves_per_90",
]

# ── Minimum-minutes filters (§6.3) ──────────────────────────────────────────
MIN_MINUTES_TOP5: int = 450
MIN_MINUTES_LOWER: int = 300

# ── Season splits (D-01) ────────────────────────────────────────────────────
TRAIN_SEASONS: list[str] = ["2021-22", "2022-23", "2023-24"]
TEST_SEASON: str = "2024-25"
ALL_SEASONS: list[str] = TRAIN_SEASONS + [TEST_SEASON]

# ── League groups (§6.3) ────────────────────────────────────────────────────
TOP5_LEAGUES: list[str] = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
LOWER4_LEAGUES: list[str] = ["Eredivisie", "Liga Portugal", "Belgian Pro League", "Süper Lig"]
ALL_LEAGUES: list[str] = TOP5_LEAGUES + LOWER4_LEAGUES


# ── Kaggle (Hubert) FBref column rename map ─────────────────────────────────
# The Kaggle dataset merges FBref's 10 stat tables; bare column names are the
# canonical ones kept after de-duplication. This maps raw FBref short names to
# readable snake_case. Columns absent here fall back to _auto_normalize().
# NOTE: bare 'Lost' is the defense table's "challenges lost" → tackles_lost.
# The misc table's aerials-lost arrives suffixed and is preserved separately via
# STATS_PRESERVE_SUFFIXED below.
FBREF_COLUMN_RENAME: dict[str, str] = {
    # Identity
    "Player": "player_name",
    "Nation": "nationality_raw",
    "Pos": "position_raw",
    "Squad": "club",
    "Comp": "league_raw",
    "Age": "age",
    "Born": "birth_year",
    # Playing time
    "MP": "matches_played",
    "Starts": "matches_started",
    "Min": "minutes_played",
    "90s": "matches_90s",
    # Scoring
    "Gls": "goals",
    "Ast": "assists",
    "G+A": "goals_plus_assists",
    "G-PK": "non_penalty_goals",
    "PK": "penalty_goals",
    "PKatt": "penalty_attempts",
    "CrdY": "yellow_cards",
    "CrdR": "red_cards",
    # Advanced
    "xG": "xg",
    "npxG": "npxg",
    "xAG": "xag",
    "npxG+xAG": "npxg_plus_xag",
    "PrgC": "progressive_carries",
    "PrgP": "progressive_passes",
    "PrgR": "progressive_passes_received",
    "G+A-PK": "goals_assists_minus_penalty",
    "xG+xAG": "xg_plus_xag",
    # Shooting
    "Sh": "shots",
    "SoT": "shots_on_target",
    "SoT%": "shots_on_target_pct",
    "Sh/90": "shots_per_90",
    "SoT/90": "shots_on_target_per_90",
    "G/Sh": "goals_per_shot",
    "G/SoT": "goals_per_shot_on_target",
    "Dist": "avg_shot_distance",
    "FK": "free_kicks",
    "npxG/Sh": "npxg_per_shot",
    "G-xG": "goals_minus_xg",
    "np:G-xG": "npg_minus_npxg",
    # Passing
    "Cmp": "passes_completed",
    "Att": "passes_attempted",
    "Cmp%": "pass_completion_pct",
    "TotDist": "pass_total_distance",
    "PrgDist": "pass_progressive_distance",
    "KP": "key_passes",
    "1/3": "passes_into_final_third",
    "PPA": "passes_into_penalty_area",
    "CrsPA": "crosses_into_penalty_area",
    "xA": "xa",
    "A-xAG": "assists_minus_xag",
    # SCA/GCA
    "SCA": "shot_creating_actions",
    "SCA90": "sca_per_90",
    "GCA": "goal_creating_actions",
    "GCA90": "gca_per_90",
    "PassLive": "passes_live",
    "PassDead": "passes_dead",
    "TO": "take_ons_leading_to_shot",
    "Fld": "fouls_drawn",
    # Defense
    "Tkl": "tackles",
    "TklW": "tackles_won",
    "Def 3rd": "tackles_def_third",
    "Mid 3rd": "tackles_mid_third",
    "Att 3rd": "tackles_att_third",
    "Tkl%": "tackle_success_pct",
    "Lost": "tackles_lost",  # defense "challenges lost"; aerials-lost preserved separately
    "Blocks": "blocks",
    "Pass": "passes_blocked",
    "Int": "interceptions",
    "Tkl+Int": "tackles_plus_interceptions",
    "Clr": "clearances",
    "Err": "errors_leading_to_shot",
    # Possession
    "Touches": "touches",
    "Def Pen": "touches_def_pen",
    "Att Pen": "touches_att_pen",
    "Live": "touches_live",
    "Carries": "carries",
    "CPA": "carries_into_penalty_area",
    "Mis": "miscontrols",
    "Dis": "dispossessed",
    "Rec": "passes_received",
    # Misc
    "Fls": "fouls_committed",
    "Recov": "ball_recoveries",
    "Won": "aerials_won",
    "Won%": "aerials_won_pct",
    "PKwon": "penalty_kicks_won",
    "PKcon": "penalty_kicks_conceded",
    "OG": "own_goals",
    # Keeper
    "GA": "goals_against",
    "GA90": "goals_against_per_90",
    "SoTA": "shots_on_target_against",
    "Saves": "saves",
    "Save%": "save_pct",
    "CS": "clean_sheets",
    "CS%": "clean_sheet_pct",
    "PKA": "penalty_kicks_allowed",
    "PKsv": "penalty_kicks_saved",
    "PSxG": "psxg",
    "PSxG/SoT": "psxg_per_sot",
    "PSxG+/-": "psxg_plus_minus",
}

# Suffixed columns that are semantically distinct and must be PRESERVED (renamed
# instead of dropped) during de-duplication, because their bare base name belongs
# to a different stat. Without this, aerials-lost would be discarded in favor of
# the defense "challenges lost" (bare `Lost`).
STATS_PRESERVE_SUFFIXED: dict[str, str] = {
    "Lost_stats_misc": "aerials_lost",
}
