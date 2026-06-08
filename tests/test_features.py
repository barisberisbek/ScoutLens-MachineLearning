"""Tests for the Phase-4 feature modules.

One+ test per module, plus a row-count-invariance guard on the full pipeline. Fixtures are
tiny hand-built frames so every expected value is exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.age_curve import add_age_curve_features
from src.features.categorical import add_categorical_features
from src.features.composites import add_composite_features
from src.features.contract import add_contract_features
from src.features.fifa import add_fifa_features
from src.features.lag import add_lag_features
from src.features.multipliers import add_multiplier_features
from src.features.per_90 import add_per_90_features


# ── M1 per-90 ─────────────────────────────────────────────────────────────────
def test_per_90_correctness():
    df = pd.DataFrame({
        "minutes_played": [1800, 0], "matches_played": [20, 1],
        "xg": [9.0, 5.0], "clean_sheets": [10, 0],
        # base stats present so PER_90_STATS lookups don't KeyError:
        "goals": [9, 1], "assists": [0, 0], "shots": [0, 0], "shots_on_target": [0, 0],
        "xag": [0.0, 0.0], "npxg": [0.0, 0.0], "understat_xa": [0.0, 0.0],
        "understat_xg": [0.0, 0.0], "saves": [0, 0], "tackles_won": [0, 0],
        "interceptions": [0, 0], "goals_against": [0, 0], "shots_on_target_against": [0, 0],
    })
    out = add_per_90_features(df)
    assert out["xg_per_90"].iloc[0] == pytest.approx(9.0 / (1800 / 90))  # 0.45
    assert np.isnan(out["xg_per_90"].iloc[1])                            # minutes 0 → NaN
    assert out["clean_sheets_per_match"].iloc[0] == pytest.approx(10 / 20)  # per-match, not per-90


# ── M2 composites ─────────────────────────────────────────────────────────────
def _composite_frame(xg, shots, **extra):
    base = dict(xg_per_90=xg, shots_per_90=shots, xag_per_90=[np.nan, np.nan],
                understat_xa_per_90=[np.nan, np.nan], tackles_won_per_90=[1.0, 1.0],
                interceptions_per_90=[1.0, 1.0], goals_per_90=[0.5, 0.5], assists_per_90=[0.2, 0.2])
    base.update(extra)
    return pd.DataFrame(base)


def test_composite_nan_propagation():
    out = add_composite_features(_composite_frame(xg=[np.nan, 0.4], shots=[2.0, 2.0]))
    assert np.isnan(out["goal_threat"].iloc[0])          # xg NaN → goal_threat NaN
    assert out["goal_threat"].iloc[1] == pytest.approx(0.4 + 0.5 * 2.0)


def test_shooting_efficiency_threshold():
    out = add_composite_features(_composite_frame(xg=[0.3, 0.3], shots=[0.2, 2.0],
                                                  goals_per_90=[0.1, 1.0]))
    assert np.isnan(out["shooting_efficiency"].iloc[0])  # shots < 0.5 → NaN
    assert out["shooting_efficiency"].iloc[1] == pytest.approx(1.0 / 2.0)


# ── M3 z-scores ───────────────────────────────────────────────────────────────
def test_z_score_grouping():
    from src.features.z_scores import add_z_score_features
    rng = np.random.default_rng(0)
    rows = []
    for pos, mu in [("GK", 0.1), ("FWD", 0.8)]:
        for v in rng.normal(mu, 0.2, 25):
            rows.append({"season": "2023-24", "primary_position": pos, "league": "Premier League",
                         "goals_per_90": v})
    df = pd.DataFrame(rows)
    out = add_z_score_features(df)
    gk = out[out.primary_position == "GK"]["goals_per_90_z_pos"]
    fwd = out[out.primary_position == "FWD"]["goals_per_90_z_pos"]
    assert gk.mean() == pytest.approx(0, abs=1e-6) and gk.std() == pytest.approx(1, abs=0.05)
    assert fwd.mean() == pytest.approx(0, abs=1e-6)
    # GK standardized within GK, not against FWD's higher mean → GK z not strongly negative
    assert abs(gk.mean()) < 0.1


# ── M4 age curve ──────────────────────────────────────────────────────────────
def test_age_curve():
    df = pd.DataFrame({"age_at_season_end": [25.0, 31.0], "primary_position": ["DEF", "GK"]})
    out = add_age_curve_features(df)
    assert out["distance_from_peak"].iloc[0] == pytest.approx(1.0)   # |25 - 26|
    assert out["is_peak"].iloc[0] and not out["is_young"].iloc[0] and not out["is_declining"].iloc[0]
    assert out["age_squared"].iloc[0] == pytest.approx(625.0)


# ── M5 contract ───────────────────────────────────────────────────────────────
def test_contract_features():
    df = pd.DataFrame({"contract_remaining_months": [6.0, 40.0, np.nan]})
    out = add_contract_features(df)
    assert out["is_expiring"].iloc[0] and not out["is_long_contract"].iloc[0]
    assert out["is_long_contract"].iloc[1] and not out["is_expiring"].iloc[1]
    assert not out["is_expiring"].iloc[2]  # NaN → False, never imputed


# ── M6 lag ────────────────────────────────────────────────────────────────────
def _lag_frame(rows):
    cols = {c: [r.get(c) for r in rows] for c in
            ["player_id", "season", "goals_per_90", "minutes_played", "matches_played",
             "assists_per_90", "shots_per_90", "xg_per_90", "xag_per_90",
             "tackles_won_per_90", "interceptions_per_90", "log_market_value",
             "fifa_rating", "fifa_potential"]}
    return pd.DataFrame(cols)


def test_lag_correctness():
    df = _lag_frame([
        {"player_id": "p1", "season": "2021-22", "goals_per_90": 0.2},
        {"player_id": "p1", "season": "2022-23", "goals_per_90": 0.5},
        {"player_id": "p1", "season": "2023-24", "goals_per_90": 0.9},
    ])
    out = add_lag_features(df).sort_values("season").reset_index(drop=True)
    assert np.isnan(out["goals_per_90_lag1"].iloc[0]) and not out["has_lag1"].iloc[0]
    assert out["goals_per_90_lag1"].iloc[1] == pytest.approx(0.2)
    assert out["delta_goals_per_90"].iloc[1] == pytest.approx(0.5 - 0.2)
    assert out["consecutive_seasons"].iloc[1] == 2 and out["consecutive_seasons"].iloc[2] == 3


def test_lag_gap_resets_consecutive():
    df = _lag_frame([
        {"player_id": "p1", "season": "2021-22", "goals_per_90": 0.2},
        {"player_id": "p1", "season": "2023-24", "goals_per_90": 0.9},  # 2022-23 skipped
    ])
    out = add_lag_features(df).sort_values("season").reset_index(drop=True)
    assert out["consecutive_seasons"].iloc[1] == 1   # gap → reset
    assert out["has_lag1"].iloc[1]                    # a prior row still exists


def test_lag_player_id_change():
    df = _lag_frame([
        {"player_id": "synthetic_abc", "season": "2021-22", "goals_per_90": 0.2},
        {"player_id": "12345", "season": "2022-23", "goals_per_90": 0.5},  # synthetic→TM transition
    ])
    out = add_lag_features(df)
    tm_row = out[out.player_id == "12345"].iloc[0]
    assert not tm_row["has_lag1"] and np.isnan(tm_row["goals_per_90_lag1"])


# ── M8 categorical ────────────────────────────────────────────────────────────
def test_categorical_one_hot():
    df = pd.DataFrame({"primary_position": ["DEF"], "continent_group": ["Europe"]})
    out = add_categorical_features(df)
    assert out["is_DEF"].iloc[0]
    assert sum(out[f"is_{p}"].iloc[0] for p in ["GK", "DEF", "MID", "FWD"]) == 1
    assert out["is_continent_Europe"].iloc[0] and not out["is_continent_Africa"].iloc[0]


# ── M7 multipliers ────────────────────────────────────────────────────────────
def test_multiplier_attach():
    df = pd.DataFrame({"season": ["2024-25", "2021-22"], "league": ["Premier League", "Süper Lig"]})
    out = add_multiplier_features(df)
    assert out["year_inflation_multiplier"].iloc[0] == pytest.approx(1.505)
    assert out["year_inflation_multiplier"].iloc[1] == pytest.approx(1.000)
    assert out["league_value_multiplier"].iloc[0] == pytest.approx(1.30)  # PL from CSV


# ── M9 fifa ───────────────────────────────────────────────────────────────────
def test_fifa_potential_gap():
    df = pd.DataFrame({"fifa_rating": [80.0, 85.0], "fifa_potential": [90.0, np.nan],
                       "fifa_rating_lag1": [78.0, np.nan]})
    out = add_fifa_features(df)
    assert out["fifa_potential_gap"].iloc[0] == pytest.approx(10.0)
    assert np.isnan(out["fifa_potential_gap"].iloc[1])      # FC25 no potential
    assert out["delta_fifa_rating"].iloc[0] == pytest.approx(2.0)


# ── M10 pipeline invariance ───────────────────────────────────────────────────
def test_row_count_invariance():
    from src.features.build_features import build_features
    from src.utils.io import load_parquet, project_root
    panel = load_parquet(project_root() / "data" / "processed" / "unified_panel.parquet")
    feats = build_features(panel)
    assert len(feats) == len(panel)
    assert set(panel.columns).issubset(feats.columns)
    assert feats.duplicated(["player_id", "season"]).sum() == 0
