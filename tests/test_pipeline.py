"""Tests for the inference pipeline — the Feature Forwarder rules are mandatory (a bug there
corrupts every prediction). Integration tests that need trained models skip if absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.feature_forwarder import fallback_notes, forward_features, next_season, year_inflation
from src.utils.constants import YEAR_INFLATION
from src.utils.io import project_root

_MODELS_PRESENT = (project_root() / "models" / "stage2").exists() and \
    any((project_root() / "models" / "stage2").rglob("*.pkl"))
_needs_models = pytest.mark.skipif(not _MODELS_PRESENT, reason="trained models not present")


def _row(position="DEF", age=25.0, contract=18.0) -> pd.DataFrame:
    base = {
        "player_id": "p1", "player_name": "Test Player", "season": "2023-24",
        "primary_position": position, "age_at_season_end": age, "age_squared": age ** 2,
        "age_cubed": age ** 3, "distance_from_peak": 0.0, "is_young": False, "is_peak": True,
        "is_declining": False, "contract_remaining_months": contract,
        "contract_years_until_expiry": contract / 12, "is_expiring": False, "is_long_contract": False,
        "has_lag1": True, "consecutive_seasons": 2, "year_inflation_multiplier": 1.254,
        "league_value_multiplier": 1.30, "fifa_rating": 80.0, "market_value_eur": 2e7,
        # per-90s used by composites + as targets/lags
        "goals_per_90": 0.20, "assists_per_90": 0.10, "shots_per_90": 1.0, "xg_per_90": 0.18,
        "xag_per_90": np.nan, "understat_xa_per_90": 0.12, "npxg_per_90": 0.15,
        "tackles_won_per_90": 1.5, "interceptions_per_90": 1.2,
        "goals_per_90_lag1": 0.10, "delta_goals_per_90": 0.10,
        "tackles_won_per_90_lag1": 1.3, "delta_tackles_won_per_90": 0.2,
        "minutes_played": 2500.0, "matches_played": 30,
    }
    return pd.DataFrame([base])


# ── forwarder rules (mandatory) ───────────────────────────────────────────────
def test_age_forward():
    out = forward_features(_row("DEF", age=25.0), "DEF", {"tackles_won_per_90": 1.6, "interceptions_per_90": 1.3, "goals_per_90": 0.25}).iloc[0]
    assert out["age_at_season_end"] == 26.0
    assert out["age_squared"] == pytest.approx(676.0)
    assert out["distance_from_peak"] == pytest.approx(0.0)   # DEF peak = 26
    assert out["is_peak"] and not out["is_declining"]


def test_contract_forward():
    out = forward_features(_row(contract=18.0), "DEF", {}).iloc[0]
    assert out["contract_remaining_months"] == 6.0
    assert out["is_expiring"]                                 # 6 ≤ 6
    out2 = forward_features(_row(contract=6.0), "DEF", {}).iloc[0]
    assert out2["contract_remaining_months"] == 0.0


def test_stage1_stats_replaced():
    proj = {"tackles_won_per_90": 2.0, "interceptions_per_90": 1.5, "goals_per_90": 0.4}
    out = forward_features(_row("DEF"), "DEF", proj).iloc[0]
    assert out["goals_per_90"] == pytest.approx(0.4)
    assert out["tackles_won_per_90"] == pytest.approx(2.0)


def test_lag_shift():
    out = forward_features(_row("DEF"), "DEF", {"goals_per_90": 0.4, "tackles_won_per_90": 2.0, "interceptions_per_90": 1.5}).iloc[0]
    assert out["goals_per_90_lag1"] == pytest.approx(0.20)    # current → lag1
    assert out["delta_goals_per_90"] == pytest.approx(0.4 - 0.20)
    assert out["consecutive_seasons"] == 3


def test_static_fields_unchanged():
    out = forward_features(_row("DEF"), "DEF", {}).iloc[0]
    assert out["league_value_multiplier"] == 1.30
    assert out["fifa_rating"] == 80.0
    assert out["primary_position"] == "DEF"


def test_season_update():
    out = forward_features(_row("DEF"), "DEF", {}).iloc[0]
    assert out["season"] == "2024-25"
    assert out["year_inflation_multiplier"] == pytest.approx(YEAR_INFLATION["2024-25"])


def test_nan_fallback():
    proj = {"tackles_won_per_90": np.nan, "interceptions_per_90": 1.5, "goals_per_90": 0.4}
    out = forward_features(_row("DEF"), "DEF", proj).iloc[0]
    assert out["tackles_won_per_90"] == pytest.approx(1.5)    # NaN proj → kept current
    assert "fallback_to_current_for_tackles_won_per_90" in fallback_notes("DEF", proj)


def test_next_season_and_inflation():
    assert next_season("2023-24") == "2024-25"
    assert next_season("2024-25") == "2025-26"
    assert year_inflation("2024-25") == pytest.approx(YEAR_INFLATION["2024-25"])
    assert year_inflation("2025-26") > YEAR_INFLATION["2024-25"]   # extrapolated up


def test_composites_recomputed():
    # goal_threat = xg_per_90 + 0.5*shots_per_90 with FORWARDED values (DEF goals proj only;
    # FWD would forward xg/shots) — here check goal_threat reflects forwarded shots if present
    out = forward_features(_row("FWD"), "FWD",
                           {"xg_per_90": 0.5, "shots_per_90": 3.0, "goals_per_90": 0.6,
                            "assists_per_90": 0.2, "npxg_per_90": 0.45, "understat_xa_per_90": 0.2}).iloc[0]
    assert out["goal_threat"] == pytest.approx(0.5 + 0.5 * 3.0)


# ── integration (need trained models) ─────────────────────────────────────────
@_needs_models
def test_loader_best_models():
    from src.pipeline.loaders import best_stage2, best_stage2_names
    names = best_stage2_names()
    assert set(names) == {"GK", "DEF", "MID", "FWD"}
    assert set(best_stage2()) == {"GK", "DEF", "MID", "FWD"}


@_needs_models
def test_predict_player_value_structure():
    from src.pipeline.inference import predict_player_value
    from src.utils.io import load_parquet
    f = load_parquet(project_root() / "data" / "processed" / "features.parquet")
    row = f[(f.season == "2023-24") & f.market_value_eur.notna()].iloc[0]
    res = predict_player_value(f, row["player_id"], "2023-24")
    for k in ["projected_mv_next_season", "current_mv_estimate", "confidence_interval",
              "stage1_projections", "mv_delta"]:
        assert k in res
    assert res["next_season"] == "2024-25"


@_needs_models
def test_ci_contains_prediction_and_batch_rowcount():
    from src.pipeline.inference import predict_batch
    from src.utils.io import load_parquet
    f = load_parquet(project_root() / "data" / "processed" / "features.parquet")
    sub = f[(f.season == "2023-24") & f.market_value_eur.notna()].head(50)
    out = predict_batch(sub, current_season="2023-24")
    assert len(out) == len(sub)                               # row count preserved
    assert out["projected_mv_next"].notna().all()             # no NaN predictions
    inside = (out["projected_mv_next"] >= out["ci_low"]) & (out["projected_mv_next"] <= out["ci_high"])
    assert inside.mean() > 0.9                                 # point estimate inside its own 95% CI
