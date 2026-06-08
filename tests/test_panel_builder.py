"""Tests for split-season collapse and the id-collision guards (Phase 2 builder).

The collapse is where mid-season-transfer stints are merged; a bug here silently
distorts every aggregate stat, so the sum / minutes-weighted-average / per-90-recompute
rules and the namesake guards are pinned with deterministic fixtures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.integration.unified_panel_builder import (
    collapse_split_season,
    split_id_collisions,
    split_minutes_overflow,
)

_BASE = dict(
    nation="ENG", age=23, primary_position="MID", detailed_position="MF",
    resolve_method="exact", resolve_score=100.0, data_richness="full",
)


def _stint(pid, season, team, minutes, goals, pct, g90, born=2000, name="Sample Player"):
    return {
        "player_id": pid, "season": season, "team": team, "league": "Premier League",
        "player": name, "born": born, "minutes_played": minutes, "matches_90s": minutes / 90,
        "goals": goals, "pass_completion_pct": pct, "goals_per_90": g90, **_BASE,
    }


@pytest.fixture
def split_frame() -> pd.DataFrame:
    # P1: one player, two clubs in 2023-24 (real split). P2: single stint.
    return pd.DataFrame([
        _stint("P1", "2023-24", "Arsenal", 1000, 5, 80.0, 0.45),
        _stint("P1", "2023-24", "Chelsea", 500, 3, 90.0, 0.54),
        _stint("P2", "2023-24", "Liverpool", 2000, 10, 85.0, 0.45, name="Other Player"),
    ])


def test_collapse_unique_player_season(split_frame):
    out = collapse_split_season(split_frame)
    assert out.duplicated(["player_id", "season"]).sum() == 0
    assert len(out) == 2  # P1 (collapsed) + P2


def test_collapse_sums_counting_stats(split_frame):
    out = collapse_split_season(split_frame)
    p1 = out[out.player_id == "P1"].iloc[0]
    assert p1["minutes_played"] == 1500          # 1000 + 500
    assert p1["goals"] == 8                       # 5 + 3
    assert p1["is_split_season"]


def test_collapse_minutes_weighted_percentage(split_frame):
    out = collapse_split_season(split_frame)
    p1 = out[out.player_id == "P1"].iloc[0]
    # (80*1000 + 90*500) / 1500 = 83.33…
    assert p1["pass_completion_pct"] == pytest.approx((80 * 1000 + 90 * 500) / 1500)


def test_collapse_per90_recomputed_from_totals(split_frame):
    out = collapse_split_season(split_frame)
    p1 = out[out.player_id == "P1"].iloc[0]
    # recomputed from summed goals / summed 90s, NOT the average of the per-90 columns
    assert p1["goals_per_90"] == pytest.approx(8 / (1500 / 90))
    assert p1["goals_per_90"] != pytest.approx((0.45 + 0.54) / 2)


def test_collapse_club_is_max_minutes_stint(split_frame):
    out = collapse_split_season(split_frame)
    p1 = out[out.player_id == "P1"].iloc[0]
    assert p1["club"] == "Arsenal"               # 1000 > 500 minutes (D-15)


def test_single_stint_not_flagged_split(split_frame):
    out = collapse_split_season(split_frame)
    assert not out[out.player_id == "P2"].iloc[0]["is_split_season"]


def test_split_id_collisions_reassigns_distinct_names():
    # one TM id wrongly shared by two different players (mononym magnet)
    df = pd.DataFrame([
        _stint("435338", "2021-22", "Arsenal", 3000, 5, 80.0, 0.15, name="Gabriel Magalhaes"),
        _stint("435338", "2021-22", "Wolves", 1000, 1, 70.0, 0.09, name="Gabriel Other"),
    ])
    out = split_id_collisions(df)
    dom = out[out.player == "Gabriel Magalhaes"].iloc[0]
    other = out[out.player == "Gabriel Other"].iloc[0]
    assert dom["player_id"] == "435338"                      # dominant minutes keeps id
    assert other["player_id"].startswith("synthetic_")       # other re-assigned
    assert other["resolve_method"] == "synthetic_split"


def test_minutes_overflow_splits_same_name_namesakes():
    # two distinct same-name players (identical mononym) summing past the physical cap
    df = pd.DataFrame([
        _stint("X", "2022-23", "Paris Saint-Germain", 3000, 7, 85.0, 0.21, name="Vitinha"),
        _stint("X", "2022-23", "Marseille", 2000, 4, 80.0, 0.18, name="Vitinha"),
    ])
    out = split_minutes_overflow(df, cap=3800)
    psg = out[out.team == "Paris Saint-Germain"].iloc[0]
    mar = out[out.team == "Marseille"].iloc[0]
    assert psg["player_id"] == "X"                           # max-minutes keeps id
    assert mar["player_id"].startswith("synthetic_")         # namesake split off
    # and after the split, neither row collapses past the cap
    collapsed = collapse_split_season(out)
    assert (collapsed["minutes_played"] <= 3800).all()
