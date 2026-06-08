"""Tests for Stage-2 dataset construction, leakage safety, metrics, and transfer matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.stage2 import data_loader
from src.models.stage2.evaluate import euro_space_metrics, regression_metrics
from src.models.stage2.models import MLPModel, RidgeModel

_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]


def _fake(n=3, position="MID", meets=None) -> pd.DataFrame:
    rows = []
    for p in range(n):
        for i, s in enumerate(_SEASONS):
            mv = 1e6 * (p + 1) + 1e5 * i
            rows.append({"player_id": f"p{p}", "player_name": f"P{p}", "season": s,
                         "primary_position": position,
                         "meets_min_minutes": True if meets is None else meets[p][i],
                         "market_value_eur": mv, "log_market_value": float(np.log1p(mv)),
                         "log_market_value_lag1": 13.0, "delta_log_market_value": 0.1,
                         "market_value_date": pd.Timestamp("2024-06-30"),
                         "goals_per_90": 0.3 + 0.05 * p, "age_at_season_end": 25.0,
                         "is_GK": position == "GK", "fifa_rating": 80.0})
    return pd.DataFrame(rows)


# ── dataset construction ──────────────────────────────────────────────────────
def test_dataset_construction():
    train, val, feats = data_loader.build_dataset(_fake(3, "MID"), "MID")
    assert len(train) == 9 and len(val) == 3            # 3 players × 3 train seasons / 1 val
    assert train.duplicated(["player_id", "season"]).sum() == 0


def test_mv_null_rows_excluded():
    df = _fake(2, "FWD")
    df.loc[df.season == "2021-22", "market_value_eur"] = np.nan
    train, val, feats = data_loader.build_dataset(df, "FWD")
    assert (train.season == "2021-22").sum() == 0       # null-MV rows dropped


# ── leakage / features ────────────────────────────────────────────────────────
def test_no_marketvalue_in_features():
    train, val, feats = data_loader.build_dataset(_fake(3, "MID"), "MID")
    for forbidden in ["log_market_value", "log_market_value_lag1", "delta_log_market_value",
                      "market_value_eur"]:
        assert forbidden not in feats
    assert "goals_per_90" in feats and "fifa_rating" in feats
    data_loader.assert_no_leakage(feats)


def test_assert_no_leakage_raises():
    with pytest.raises(AssertionError):
        data_loader.assert_no_leakage(["age_at_season_end", "log_market_value_lag1"])


# ── temporal split ────────────────────────────────────────────────────────────
def test_temporal_split_no_leakage():
    train, val, feats = data_loader.build_dataset(_fake(3, "MID"), "MID")
    assert set(train.season) <= {"2021-22", "2022-23", "2023-24"}
    assert set(val.season) == {"2024-25"}


# ── min-minutes + position filters ────────────────────────────────────────────
def test_min_minutes_filter():
    meets = [[True, False, True, True], [True, True, True, True], [True, True, True, True]]
    train, val, feats = data_loader.build_dataset(_fake(3, "MID", meets=meets), "MID")
    assert len(train) == 8                              # p0's 2022-23 row removed


def test_per_position_filter():
    df = pd.concat([_fake(2, "GK"), _fake(2, "DEF")], ignore_index=True)
    df.loc[df.is_GK, "player_id"] = "gk_" + df.loc[df.is_GK, "player_id"]
    train, val, feats = data_loader.build_dataset(df, "GK")
    assert (train.primary_position == "GK").all()


# ── persistence ───────────────────────────────────────────────────────────────
def test_model_save_load(tmp_path):
    import joblib
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(X["f0"] + rng.normal(scale=0.1, size=80))
    model = RidgeModel().fit(X, y)
    joblib.dump(model.best_estimator_, tmp_path / "m.pkl")
    loaded = joblib.load(tmp_path / "m.pkl")
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))


# ── metrics (log + euro space) ────────────────────────────────────────────────
def test_log_space_metrics():
    y_log = np.log1p(np.array([1e6, 4e6, 9e6]))
    p_log = np.log1p(np.array([1e6, 4e6, 1.0e7]))           # last one over by 1M
    e = euro_space_metrics(y_log, p_log)
    assert e["mae_eur"] == pytest.approx(1e6 / 3, rel=1e-6)
    m = regression_metrics(y_log, p_log)
    assert m["mae"] >= 0


def test_baseline_predictor_logic():
    # the median predictor used as Stage-2 baseline
    y_tr = pd.Series([10.0, 12.0, 14.0, 16.0])
    from sklearn.metrics import mean_absolute_error
    y_va = np.array([11.0, 13.0])
    base = mean_absolute_error(y_va, np.full(len(y_va), y_tr.median()))   # median=13 → |11-13|+|13-13|=2/2
    assert base == pytest.approx(1.0)


# ── tightened MLP grid (Phase-6 decision) ─────────────────────────────────────
def test_mlp_grid_tightened():
    grid = MLPModel()._param_grid()
    assert grid["model__alpha"] == [0.1, 1.0, 10.0]        # not Stage-1's 0.001/0.01


# ── transfer matching ─────────────────────────────────────────────────────────
def test_transfer_match():
    from src.models.stage2.transfer_validation import _match_transfers
    f_pre = pd.DataFrame({
        "player_name": ["Bruno Fernandes", "Other Guy"],
        "age_at_season_end": [29.0, 22.0], "primary_position": ["MID", "DEF"],
    })
    transfers = pd.DataFrame({"player_name": ["Bruno Fernandes"], "player_age": [29],
                              "transfer_fee_eur": [50e6], "to_league": ["Premier League"]})
    matched = _match_transfers(transfers, f_pre)
    assert len(matched) == 1 and matched.iloc[0]["position"] == "MID"
