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


# ── FBref (soccerdata) scrape configuration ─────────────────────────────────
# Maps our canonical league name → the soccerdata league key passed to
# sd.FBref(leagues=[...]). The Big-5 keys are soccerdata built-ins; the lower-4
# keys are defined in a custom league_dict.json (see data/external/) because
# soccerdata 1.9.0 does not ship them.
FBREF_LEAGUE_IDS: dict[str, str] = {
    "Premier League": "ENG-Premier League",
    "La Liga": "ESP-La Liga",
    "Bundesliga": "GER-Bundesliga",
    "Serie A": "ITA-Serie A",
    "Ligue 1": "FRA-Ligue 1",
    "Eredivisie": "NED-Eredivisie",
    "Liga Portugal": "POR-Primeira Liga",
    "Belgian Pro League": "BEL-Pro League",
    "Süper Lig": "TUR-Süper Lig",
}

# soccerdata 4-digit season codes (e.g. '2324' == 2023-2024).
FBREF_SEASONS: list[str] = ["2122", "2223", "2324", "2425"]

# Player-season stat tables. soccerdata 1.9.0 natively allows only the NATIVE set
# (fbref.py:546); the EXTENDED set is fetched via the monkeypatched method (same engine,
# wider whitelist). All 11 FBref player-season tables are scraped. NOTE: none of these
# carry xG/npxG/xAG/progression-counts (a soccerdata limitation) — xG comes from Kaggle
# (2024-25 top-5) and Understat (top-5, all seasons); lower-league xG stays unavailable.
FBREF_STAT_TYPES_NATIVE: list[str] = ["standard", "shooting", "playing_time", "keeper", "misc"]
FBREF_STAT_TYPES_EXTENDED: list[str] = [
    "passing", "passing_types", "gca", "defense", "possession", "keeper_adv",
]
FBREF_STAT_TYPES: list[str] = FBREF_STAT_TYPES_NATIVE + FBREF_STAT_TYPES_EXTENDED


# ── Transfermarkt (davidcariboo player-scores) configuration ────────────────
# Transfermarkt competition_id per canonical league (verified against competitions.csv).
TM_LEAGUE_COMPETITION_IDS: dict[str, str] = {
    "Premier League": "GB1",
    "La Liga": "ES1",
    "Bundesliga": "L1",
    "Serie A": "IT1",
    "Ligue 1": "FR1",
    "Eredivisie": "NL1",
    "Liga Portugal": "PO1",
    "Belgian Pro League": "BE1",
    "Süper Lig": "TR1",
}

# Season-end reference dates for MV snapshot alignment (D-16).
SEASON_END_DATES: dict[str, str] = {
    "2021-22": "2022-06-30",
    "2022-23": "2023-06-30",
    "2023-24": "2024-06-30",
    "2024-25": "2025-06-30",
}
MV_SNAPSHOT_WINDOW_DAYS: int = 45

# Transfermarkt position taxonomy → our primary positions. TM uses full words
# (NOT FBref's GK/DF/MF/FW), so normalize_position() does NOT apply here.
TM_POSITION_MAP: dict[str, str] = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfield": "MID",
    "Attack": "FWD",
}


# ── FIFA / EA FC ratings configuration ──────────────────────────────────────
# stefanoleone992 schema (FIFA 22/23/24 all live in the EA FC 24 male_players.csv,
# one snapshot per fifa_version). player_id IS the sofifa id. age/overall/potential
# already have the target names.
FIFA_STEFANO_COLUMN_RENAME: dict[str, str] = {
    "player_id": "sofifa_id",
    "short_name": "player_name",
    "long_name": "player_name_full",
    "dob": "date_of_birth",
    "nationality_name": "nationality",
    "club_name": "club",
    "league_name": "league",
    "player_positions": "position_detail",
}

# nyagami EA FC 25 schema (PascalCase; NO potential, NO dob, NO sofifa_id).
FIFA_NYAGAMI_COLUMN_RENAME: dict[str, str] = {
    "Name": "player_name",
    "OVR": "overall",
    "Age": "age",
    "Nation": "nationality",
    "League": "league",
    "Team": "club",
    "Position": "position_detail",
    "url": "player_url",
}

# FIFA position codes → primary GK/DEF/MID/FWD. (normalize_position does NOT apply —
# FIFA uses ST/CB/CM/… not FBref's GK/DF/MF/FW.) Multi-position strings ("ST, CF")
# are resolved on the first token.
FIFA_POSITION_MAP: dict[str, str] = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "LCB": "DEF", "RCB": "DEF", "SW": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
    "LDM": "MID", "RDM": "MID", "LCM": "MID", "RCM": "MID", "DM": "MID", "AM": "MID",
    "ST": "FWD", "CF": "FWD", "LW": "FWD", "RW": "FWD", "LF": "FWD", "RF": "FWD",
    "LS": "FWD", "RS": "FWD", "LWF": "FWD", "RWF": "FWD",
}

FIFA_YEAR_TO_SEASON: dict[int, str] = {
    22: "2021-22", 23: "2022-23", 24: "2023-24", 25: "2024-25",
}


# ── Phase 2 integration crosswalks ──────────────────────────────────────────
# soccerdata 4-digit season code → canonical "YYYY-YY". FBref/Understat parquets
# store the short code (e.g. "2324"); the panel uses the canonical form.
FBREF_SEASON_TO_CANONICAL: dict[str, str] = {
    "2122": "2021-22", "2223": "2022-23", "2324": "2023-24", "2425": "2024-25",
}

# Reverse of FBREF_LEAGUE_IDS: the soccerdata league key stored in the FBref/
# Understat `league` column (e.g. "ENG-Premier League") → our canonical name.
FBREF_LEAGUE_TO_CANONICAL: dict[str, str] = {v: k for k, v in FBREF_LEAGUE_IDS.items()}

# Nationality full-name variants (Transfermarkt / FIFA spellings) that do NOT
# match a continent_map.csv key after accent/punctuation normalization, mapped to
# the canonical continent_map spelling. Accent/punctuation-only variants
# (e.g. "Cote d'Ivoire" → "Côte d'Ivoire", "Curacao" → "Curaçao") are handled by
# the resolver's normalized index and need NOT be listed here — only reorderings
# and extra-word variants do.
NATIONALITY_NAME_ALIAS: dict[str, str] = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",
    "The Gambia": "Gambia",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Chinese Taipei": "China",  # rare; no separate Taiwan bucket in continent_map
    "Neukaledonien": "New Caledonia",
}

# Club-name normalization for the Understat `fuzzy_team` disambiguation path.
# Token-level aliases applied AFTER lowercase/accent-strip; keeps "Manchester Utd"
# (FBref) and "Manchester United" (TM/Understat) from silently mismatching.
CLUB_ALIAS_MAP: dict[str, str] = {
    "utd": "united",
    "inter": "internazionale",
    "wolves": "wolverhampton",
    "spurs": "tottenham",
    "psg": "paris",
    "atletico": "atletico",
    "betis": "betis",
}

# Club-name stopwords dropped during normalization (legal/structural suffixes that
# vary across sources). Applied token-wise.
CLUB_STOPWORDS: frozenset[str] = frozenset({
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "us", "ud", "cd", "rc",
    "club", "calcio", "de", "of",
})


# ── FBref 11-table stat merge (Phase 2 §5.3) ────────────────────────────────
# Raw soccerdata tables store columns as "<Section>_<Short>" (e.g. "Performance_Gls",
# "Per 90 Minutes_Gls"). A short-only rename collapses semantically-distinct columns
# (Short/Medium/Long_Cmp all → passes_completed; Per-90 vs counting), so the builder
# uses an explicit per-table CURATED map for the columns that deserve clean canonical
# names, and namespaces everything else as "<table>__<snaked_raw>" (collision-free).
# NOTE: soccerdata FBref tables carry NO xG/npxG/xAG, NO progression COUNTS (PrgP/PrgC),
# and NO aerial-duel columns — those gaps are filled (where possible) from Kaggle/Understat
# downstream, NOT here.
FBREF_TABLE_STAT_RENAME: dict[str, dict[str, str]] = {
    "standard": {
        "Playing Time_MP": "matches_played",
        "Playing Time_Starts": "matches_started",
        "Playing Time_Min": "minutes_played",
        "Playing Time_90s": "matches_90s",
        "Performance_Gls": "goals",
        "Performance_Ast": "assists",
        "Performance_G+A": "goals_plus_assists",
        "Performance_G-PK": "non_penalty_goals",
        "Performance_PK": "penalty_goals",
        "Performance_PKatt": "penalty_attempts",
        "Performance_CrdY": "yellow_cards",
        "Performance_CrdR": "red_cards",
        "Per 90 Minutes_Gls": "goals_per_90",
        "Per 90 Minutes_Ast": "assists_per_90",
        "Per 90 Minutes_G+A": "goals_plus_assists_per_90",
        "Per 90 Minutes_G-PK": "non_penalty_goals_per_90",
        "Per 90 Minutes_G+A-PK": "goals_assists_minus_penalty_per_90",
    },
    "shooting": {
        "Standard_Sh": "shots",
        "Standard_SoT": "shots_on_target",
        "Standard_SoT%": "shots_on_target_pct",
        "Standard_Sh/90": "shots_per_90",
        "Standard_SoT/90": "shots_on_target_per_90",
        "Standard_G/Sh": "goals_per_shot",
        "Standard_G/SoT": "goals_per_shot_on_target",
    },
    "passing": {
        "Total_Cmp": "passes_completed",
        "Total_Att": "passes_attempted",
        "Total_Cmp%": "pass_completion_pct",
        "Total_TotDist": "pass_total_distance",
        "Total_PrgDist": "pass_progressive_distance",
        "Short_Cmp": "passes_short_completed",
        "Short_Att": "passes_short_attempted",
        "Short_Cmp%": "pass_short_completion_pct",
        "Medium_Cmp": "passes_medium_completed",
        "Medium_Att": "passes_medium_attempted",
        "Medium_Cmp%": "pass_medium_completion_pct",
        "Long_Cmp": "passes_long_completed",
        "Long_Att": "passes_long_attempted",
        "Long_Cmp%": "pass_long_completion_pct",
        "KP": "key_passes",
        "1/3": "passes_into_final_third",
        "PPA": "passes_into_penalty_area",
        "CrsPA": "crosses_into_penalty_area",
        "A-xAG": "assists_minus_xag",
    },
    "passing_types": {
        "Pass Types_Live": "passes_live",
        "Pass Types_Dead": "passes_dead",
        "Pass Types_FK": "passes_free_kick",
        "Pass Types_TB": "through_balls",
        "Pass Types_Sw": "switches",
        "Pass Types_Crs": "crosses",
        "Pass Types_TI": "throw_ins",
        "Pass Types_CK": "corner_kicks",
        "Outcomes_Off": "passes_offside",
        "Outcomes_Blocks": "passes_blocked_by_opp",
    },
    "gca": {
        "SCA_SCA": "shot_creating_actions",
        "SCA_SCA90": "sca_per_90",
        "GCA_GCA": "goal_creating_actions",
        "GCA_GCA90": "gca_per_90",
    },
    "defense": {
        "Tackles_Tkl": "tackles",
        "Tackles_TklW": "tackles_won",
        "Tackles_Def 3rd": "tackles_def_third",
        "Tackles_Mid 3rd": "tackles_mid_third",
        "Tackles_Att 3rd": "tackles_att_third",
        "Challenges_Tkl": "dribblers_tackled",
        "Challenges_Att": "dribbles_challenged",
        "Challenges_Tkl%": "tackle_success_pct",
        "Challenges_Lost": "challenges_lost",
        "Blocks_Blocks": "blocks",
        "Blocks_Sh": "blocked_shots",
        "Blocks_Pass": "blocked_passes",
        "Int": "interceptions",
        "Tkl+Int": "tackles_plus_interceptions",
        "Clr": "clearances",
        "Err": "errors_leading_to_shot",
    },
    "possession": {
        "Touches_Touches": "touches",
        "Touches_Def Pen": "touches_def_pen",
        "Touches_Def 3rd": "touches_def_third",
        "Touches_Mid 3rd": "touches_mid_third",
        "Touches_Att 3rd": "touches_att_third",
        "Touches_Att Pen": "touches_att_pen",
        "Touches_Live": "touches_live",
        "Take-Ons_Att": "take_ons_attempted",
        "Take-Ons_Succ": "take_ons_succeeded",
        "Take-Ons_Succ%": "take_on_success_pct",
        "Take-Ons_Tkld": "take_ons_tackled",
        "Take-Ons_Tkld%": "take_on_tackled_pct",
        "Carries_Carries": "carries",
        "Carries_TotDist": "carry_total_distance",
        "Carries_PrgDist": "carry_progressive_distance",
        "Carries_1/3": "carries_into_final_third",
        "Carries_CPA": "carries_into_penalty_area",
        "Carries_Mis": "miscontrols",
        "Carries_Dis": "dispossessed",
        "Rec": "passes_received",
    },
    "misc": {
        "Performance_2CrdY": "second_yellow_cards",
        "Performance_Fls": "fouls_committed",
        "Performance_Fld": "fouls_drawn",
        "Performance_Off": "offsides",
        "Performance_Crs": "crosses_made",
        "Performance_PKwon": "penalty_kicks_won",
        "Performance_PKcon": "penalty_kicks_conceded",
        "Performance_OG": "own_goals",
    },
    "playing_time": {
        "Playing Time_Mn/MP": "minutes_per_match",
        "Playing Time_Min%": "team_minutes_pct",
        "Starts_Compl": "complete_matches",
        "Subs_Subs": "sub_appearances",
        "Subs_unSub": "unused_sub",
        "Team Success_PPM": "team_points_per_match",
        "Team Success_onG": "team_goals_on_pitch",
        "Team Success_onGA": "team_goals_against_on_pitch",
        "Team Success_+/-": "plus_minus",
        "Team Success_+/-90": "plus_minus_per_90",
        "Team Success_On-Off": "on_off",
    },
    "keeper": {
        "Performance_GA": "goals_against",
        "Performance_GA90": "goals_against_per_90",
        "Performance_SoTA": "shots_on_target_against",
        "Performance_Saves": "saves",
        "Performance_Save%": "save_pct",
        "Performance_W": "gk_wins",
        "Performance_D": "gk_draws",
        "Performance_L": "gk_losses",
        "Performance_CS": "clean_sheets",
        "Performance_CS%": "clean_sheet_pct",
        "Penalty Kicks_PKatt": "gk_penalty_attempts",
        "Penalty Kicks_PKA": "gk_penalty_allowed",
        "Penalty Kicks_PKsv": "gk_penalty_saved",
        "Penalty Kicks_PKm": "gk_penalty_missed",
        "Penalty Kicks_Save%": "gk_penalty_save_pct",
    },
    "keeper_adv": {
        "Expected_PSxG": "psxg",
        "Expected_PSxG/SoT": "psxg_per_sot",
        "Expected_PSxG+/-": "psxg_plus_minus",
        "Expected_/90": "psxg_per_90",
        "Launched_Cmp": "gk_launched_completed",
        "Launched_Att": "gk_launched_attempted",
        "Launched_Cmp%": "gk_launch_completion_pct",
        "Passes_Att (GK)": "gk_passes_attempted",
        "Passes_Thr": "gk_throws",
        "Passes_Launch%": "gk_pass_launch_pct",
        "Passes_AvgLen": "gk_pass_avg_len",
        "Goal Kicks_Att": "gk_goal_kicks",
        "Goal Kicks_Launch%": "gk_goal_kick_launch_pct",
        "Goal Kicks_AvgLen": "gk_goal_kick_avg_len",
        "Crosses_Opp": "gk_crosses_faced",
        "Crosses_Stp": "gk_crosses_stopped",
        "Crosses_Stp%": "gk_crosses_stopped_pct",
        "Sweeper_#OPA": "gk_sweeper_actions",
        "Sweeper_#OPA/90": "gk_sweeper_per_90",
        "Sweeper_AvgDist": "gk_sweeper_avg_dist",
    },
}

# Raw columns dropped per table because they duplicate a canonical claimed by a
# higher-priority table (e.g. shooting's "Standard_Gls" duplicates standard's goals;
# the bare "90s" repeats matches_90s in every table). Dropping is logged, never silent.
FBREF_DROP_DUP_COLS: dict[str, frozenset[str]] = {
    "shooting": frozenset({"90s", "Standard_Gls", "Standard_PK", "Standard_PKatt"}),
    "passing": frozenset({"90s", "Ast"}),
    "passing_types": frozenset({"90s", "Att", "Outcomes_Cmp"}),
    "gca": frozenset({"90s"}),
    "defense": frozenset({"90s"}),
    "possession": frozenset({"90s"}),
    "misc": frozenset({"90s", "Performance_CrdY", "Performance_CrdR",
                       "Performance_Int", "Performance_TklW"}),
    "playing_time": frozenset({"Playing Time_MP", "Playing Time_Min",
                              "Playing Time_90s", "Starts_Starts"}),
    "keeper": frozenset({"Playing Time_MP", "Playing Time_Starts",
                        "Playing Time_Min", "Playing Time_90s"}),
    "keeper_adv": frozenset({"90s", "Goals_GA"}),
}

# Stat-table merge priority: the first table to claim a canonical name wins. `standard`
# is the authoritative source of identity + headline counting stats.
FBREF_TABLE_PRIORITY: list[str] = [
    "standard", "shooting", "passing", "passing_types", "gca",
    "defense", "possession", "misc", "playing_time", "keeper", "keeper_adv",
]
