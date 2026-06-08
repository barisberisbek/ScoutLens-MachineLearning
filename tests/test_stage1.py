"""Tests for Stage-1 pair construction, leakage safety, persistence, and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.stage1 import data_loader
from src.models.stage1.evaluate import naive_baseline_mae, regression_metrics
from src.models.stage1.models import RidgeModel, XGBoostModel
from src.models.stage1.target_specs import ALL_TARGETS

_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]


def _fake_features(n_players=3, position="MID", meets=None) -> pd.DataFrame:
    rows = []
    for p in range(n_players):
        for i, s in enumerate(_SEASONS):
            row = {"player_id": f"p{p}", "player_name": f"P{p}", "season": s,
                   "primary_position": position,
                   "meets_min_minutes": True if meets is None else meets[p][i],
                   "market_value_eur": 1e6, "log_market_value": 13.8,
                   "age_at_season_end": 25.0, "is_GK": position == "GK",
                   "fifa_rating": 80.0}
            for t in ALL_TARGETS:
                row[t] = 0.1 * p + 0.05 * i
            rows.append(row)
    return pd.DataFrame(rows)


# ── pair construction ─────────────────────────────────────────────────────────
def test_pair_construction():
    pairs = data_loader.build_all_pairs(_fake_features(3, "MID"))
    assert len(pairs) == 9                      # 3 players × (2 train + 1 val)
    assert (pairs.split == "train").sum() == 6
    assert (pairs.split == "val").sum() == 3


def test_pair_player_id_change_skipped():
    df = _fake_features(1, "MID")
    df.loc[df.season == "2023-24", "player_id"] = "different"  # breaks continuity
    pairs = data_loader.build_all_pairs(df)
    # p0 keeps 21-22→22-23 only (22-23→23-24 broken by id change); 'different' has 1 season
    assert (pairs.player_id == "p0").sum() == 1


# ── leakage / feature separation ──────────────────────────────────────────────
def test_no_target_or_marketvalue_in_features():
    pairs = data_loader.build_all_pairs(_fake_features(3, "MID"))
    feats = data_loader.feature_columns(pairs)
    assert not any(c.startswith("next_") for c in feats)
    assert not any("market_value" in c for c in feats)
    assert "goals_per_90" in feats          # current-season value IS a feature (past info)
    assert "is_GK" in feats
    data_loader.assert_no_leakage(feats)    # should not raise


def test_assert_no_leakage_raises():
    with pytest.raises(AssertionError):
        data_loader.assert_no_leakage(["age_at_season_end", "next_goals_per_90"])


# ── temporal split ────────────────────────────────────────────────────────────
def test_temporal_split_no_leakage():
    pairs = data_loader.build_all_pairs(_fake_features(3, "MID"))
    train, val = pairs[pairs.split == "train"], pairs[pairs.split == "val"]
    assert set(train.season) <= {"2021-22", "2022-23"}
    assert set(val.season) == {"2023-24"}
    assert "2024-25" not in set(train.season)   # train current-seasons never the test year


# ── min-minutes filter ────────────────────────────────────────────────────────
def test_min_minutes_filter():
    # player p0 fails min-minutes in 2022-23 → both pairs touching it drop (9 → 7)
    meets = [[True, False, True, True], [True, True, True, True], [True, True, True, True]]
    pairs = data_loader.build_all_pairs(_fake_features(3, "MID", meets=meets))
    assert (pairs.player_id == "p0").sum() == 1   # only 23-24→24-25 survives for p0
    assert len(pairs) == 7


# ── per-position filter ───────────────────────────────────────────────────────
def test_per_position_filter():
    df = pd.concat([_fake_features(2, "GK"), _fake_features(2, "FWD")], ignore_index=True)
    # disambiguate ids across the two position blocks
    df.loc[df.is_GK, "player_id"] = "gk_" + df.loc[df.is_GK, "player_id"]
    df.loc[~df.is_GK, "player_id"] = "fw_" + df.loc[~df.is_GK, "player_id"]
    pairs = data_loader.build_all_pairs(df)
    gk = pairs[pairs.primary_position == "GK"]
    assert (gk.primary_position == "GK").all() and len(gk) == 6


# ── persistence roundtrip ─────────────────────────────────────────────────────
def test_model_save_load(tmp_path):
    import joblib
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(X["f0"] * 2 + rng.normal(scale=0.1, size=80))
    model = RidgeModel().fit(X, y)
    p = tmp_path / "m.pkl"
    joblib.dump(model.best_estimator_, p)
    loaded = joblib.load(p)
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))


# ── metrics ───────────────────────────────────────────────────────────────────
def test_metric_calculations():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.25)
    assert m["rmse"] == pytest.approx(0.5)


def test_baseline_computation():
    y_true = np.array([1.0, 2.0, np.nan, 4.0])
    current = np.array([1.5, 2.0, 3.0, 4.0])
    # masked to rows 0,1,3: |1-1.5|+|2-2|+|4-4| = 0.5 / 3
    assert naive_baseline_mae(y_true, current) == pytest.approx(0.5 / 3)


# ── CPU XGBoost smoke (replaces the dropped GPU test) ─────────────────────────
def test_cpu_xgboost_fit():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(X["f0"] + X["f1"] * 0.5 + rng.normal(scale=0.1, size=120))
    model = XGBoostModel().fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (120,) and np.isfinite(preds).all()
    assert model.best_estimator_.named_steps["model"].get_params()["device"] == "cpu"
