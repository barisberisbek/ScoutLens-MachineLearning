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
