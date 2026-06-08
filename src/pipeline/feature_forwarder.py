"""⭐ Feature Forwarder — deterministic season-forward transform (NOT ML).

Ages a player's current-season feature row to its next-season equivalent so the Stage-2
model values a *projected* future, not the present (this is what makes the pipeline immune
to circular reasoning). A bug here silently corrupts every prediction → mandatory tests.

Rules: age +1, contract −12mo (clip 0), lag-shift (current → lag1), swap the position's
Stage-1-target stats for their projections (NaN → keep current, flagged), recompute the
deterministic composites from the forwarded per-90s, advance the season + year-inflation.
Static (league, position, nationality, FIFA, one-hots) and z-scores are kept unchanged —
z-scores can't be recomputed without next-season group stats (a flagged approximation).
"""

from __future__ import annotations

import pandas as pd

from src.features.composites import add_composite_features
from src.models.stage1.target_specs import STAGE1_TARGETS
from src.utils.constants import LAG_STATS, PEAK_AGE_BY_POSITION, YEAR_INFLATION


def next_season(season: str) -> str:
    """'2023-24' → '2024-25'."""
    start = int(season.split("-")[0])
    return f"{start + 1}-{str(start + 2)[-2:]}"


def year_inflation(season: str) -> float:
    """Look up the cumulative inflation multiplier; extrapolate ×1.15/yr beyond the table."""
    if season in YEAR_INFLATION:
        return YEAR_INFLATION[season]
    last = max(YEAR_INFLATION)
    gap = int(season.split("-")[0]) - int(last.split("-")[0])
    return YEAR_INFLATION[last] * (1.15 ** max(gap, 0))


def forward_features(rows: pd.DataFrame, position: str, projections) -> pd.DataFrame:
    """Forward a position's current-season rows by one season.

    `projections` is a DataFrame (index-aligned to `rows`, columns = Stage-1 targets) or a
    dict {target: scalar} applied to every row. Returns a new DataFrame with the same columns.
    """
    fwd = rows.copy()
    targets = STAGE1_TARGETS.get(position, [])
    if isinstance(projections, dict):
        proj = pd.DataFrame({t: [projections.get(t)] * len(rows) for t in targets}, index=rows.index)
    else:
        proj = projections

    # 1. age
    age = rows["age_at_season_end"] + 1
    fwd["age_at_season_end"] = age
    if "age_squared" in fwd:
        fwd["age_squared"] = age ** 2
    if "age_cubed" in fwd:
        fwd["age_cubed"] = age ** 3
    if "distance_from_peak" in fwd:
        fwd["distance_from_peak"] = (age - PEAK_AGE_BY_POSITION[position]).abs()
    if "is_young" in fwd:
        fwd["is_young"] = (age < 24).fillna(False)
        fwd["is_peak"] = age.between(24, 29).fillna(False)
        fwd["is_declining"] = (age > 29).fillna(False)

    # 2. contract
    if "contract_remaining_months" in fwd:
        nm = (rows["contract_remaining_months"] - 12).clip(lower=0)
        fwd["contract_remaining_months"] = nm
        if "contract_years_until_expiry" in fwd:
            fwd["contract_years_until_expiry"] = nm / 12
        if "is_expiring" in fwd:
            fwd["is_expiring"] = (nm <= 6).fillna(False)
            fwd["is_long_contract"] = (nm > 36).fillna(False)

    # 3 + 4. lag shift + stat forward (capture current BEFORE overwriting)
    for stat in LAG_STATS:
        if stat not in fwd.columns:
            continue
        lag_col, delta_col = f"{stat}_lag1", f"delta_{stat}"
        if lag_col in fwd.columns:
            fwd[lag_col] = rows[stat]
        if stat in targets and stat in proj.columns:
            fwd[stat] = proj[stat].where(proj[stat].notna(), rows[stat])   # NaN proj → keep current
        if delta_col in fwd.columns:
            fwd[delta_col] = fwd[stat] - fwd.get(lag_col, rows[stat])
    # Stage-1 targets that aren't lagged (npxg, understat_xa, save_pct, clean_sheets, …)
    for t in targets:
        if t in LAG_STATS or t not in fwd.columns or t not in proj.columns:
            continue
        fwd[t] = proj[t].where(proj[t].notna(), rows[t])

    if "has_lag1" in fwd:
        fwd["has_lag1"] = True
    if "consecutive_seasons" in fwd:
        fwd["consecutive_seasons"] = rows["consecutive_seasons"].fillna(0) + 1

    # 5. recompute the deterministic composites from the forwarded per-90s (z-scores left stale)
    fwd = add_composite_features(fwd)

    # 7. season + inflation
    ns = rows["season"].map(next_season)
    fwd["season"] = ns
    if "year_inflation_multiplier" in fwd:
        fwd["year_inflation_multiplier"] = ns.map(year_inflation)

    return fwd


def fallback_notes(position: str, projections: dict) -> list[str]:
    """Which Stage-1 targets fell back to current values (NaN projection)."""
    return [f"fallback_to_current_for_{t}" for t in STAGE1_TARGETS.get(position, [])
            if projections.get(t) is None or pd.isna(projections.get(t))]
