"""Per-position Stage-1 projection targets (revised §6.2 / P2-D6).

Only the survivor stat universe + xG/xAG is projectable. MID's `xag_per_90` was swapped for
`understat_xa_per_90` — xag exists only for 2024-25 (Kaggle) so it has no training history;
understat_xa is its empirical equivalent (r≈0.93, Phase-3 EDA) with historical coverage.
"""

from __future__ import annotations

STAGE1_TARGETS: dict[str, list[str]] = {
    "GK": ["saves_per_90", "save_pct", "clean_sheets_per_match", "goals_against_per_90"],
    "DEF": ["tackles_won_per_90", "interceptions_per_90", "goals_per_90"],
    "MID": ["goals_per_90", "assists_per_90", "xg_per_90", "understat_xa_per_90",
            "tackles_won_per_90", "interceptions_per_90"],
    "FWD": ["goals_per_90", "shots_per_90", "xg_per_90", "assists_per_90",
            "npxg_per_90", "understat_xa_per_90"],
}

POSITIONS: list[str] = ["GK", "DEF", "MID", "FWD"]

# Union of all distinct targets (for building next-season columns once).
ALL_TARGETS: list[str] = sorted({t for ts in STAGE1_TARGETS.values() for t in ts})
