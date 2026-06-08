"""End-to-end inference: Stage 1 → Feature Forwarder → Stage 2 (single player + batch)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.stage1.target_specs import POSITIONS, STAGE1_TARGETS
from src.pipeline.confidence import ci_bounds_array, confidence_interval
from src.pipeline.feature_forwarder import fallback_notes, forward_features
from src.pipeline.loaders import best_stage1, best_stage2


def predict_batch(features_df: pd.DataFrame, current_season: str | None = None,
                  with_ci: bool = True) -> pd.DataFrame:
    """Forward-value every row (optionally restricted to `current_season`). Row count preserved."""
    s1, s2 = best_stage1(), best_stage2()
    rows = features_df if current_season is None else features_df[features_df["season"] == current_season]
    frames = []
    for pos in POSITIONS:
        sub = rows[rows["primary_position"] == pos]
        if sub.empty:
            continue
        proj = pd.DataFrame(index=sub.index)
        for target in STAGE1_TARGETS[pos]:
            est = s1[(pos, target)]
            proj[target] = est.predict(sub[list(est.feature_names_in_)])
        s2est = s2[pos]
        cur_log = s2est.predict(sub[list(s2est.feature_names_in_)])
        fwd = forward_features(sub, pos, proj)
        proj_log = s2est.predict(fwd[list(s2est.feature_names_in_)])
        res = pd.DataFrame({
            "player_id": sub["player_id"].values, "player_name": sub["player_name"].values,
            "position": pos, "current_season": sub["season"].values, "next_season": fwd["season"].values,
            "observed_current_mv": sub["market_value_eur"].values,
            "current_mv_estimate": np.expm1(cur_log), "projected_mv_next": np.expm1(proj_log),
            "n_fallback": proj.isna().sum(axis=1).values, "_proj_log": proj_log,
        })
        if with_ci:
            lo, hi = ci_bounds_array(proj_log, pos)
            res["ci_low"], res["ci_high"] = lo, hi
        frames.append(res)

    out = pd.concat(frames, ignore_index=True)
    base = out["observed_current_mv"].fillna(out["current_mv_estimate"])
    out["mv_delta"] = out["projected_mv_next"] - base
    out["mv_delta_pct"] = 100 * out["mv_delta"] / base
    return out.drop(columns="_proj_log")


def predict_player_value(features_df: pd.DataFrame, player_id: str, current_season: str) -> dict:
    """Full forward-looking valuation for one (player, season) → structured dict."""
    sub = features_df[(features_df["player_id"] == player_id) & (features_df["season"] == current_season)]
    if sub.empty:
        raise ValueError(f"no row for player_id={player_id} season={current_season}")
    sub = sub.iloc[[0]]
    pos = sub.iloc[0]["primary_position"]
    s1, s2 = best_stage1(), best_stage2()

    proj = {t: float(s1[(pos, t)].predict(sub[list(s1[(pos, t)].feature_names_in_)])[0])
            for t in STAGE1_TARGETS[pos]}
    s2est = s2[pos]
    cur_log = float(s2est.predict(sub[list(s2est.feature_names_in_)])[0])
    fwd = forward_features(sub, pos, proj)
    proj_log = float(s2est.predict(fwd[list(s2est.feature_names_in_)])[0])

    proj_mv, cur_est = float(np.expm1(proj_log)), float(np.expm1(cur_log))
    obs = sub.iloc[0]["market_value_eur"]
    base = float(obs) if pd.notna(obs) else cur_est
    return {
        "player_id": player_id, "player_name": sub.iloc[0]["player_name"], "position": pos,
        "current_season": current_season, "next_season": fwd.iloc[0]["season"],
        "observed_current_mv": float(obs) if pd.notna(obs) else None,
        "current_mv_estimate": cur_est, "projected_mv_next_season": proj_mv,
        "mv_delta": proj_mv - base, "mv_delta_pct": (100 * (proj_mv - base) / base) if base else None,
        "confidence_interval": confidence_interval(proj_log, pos),
        "stage1_projections": proj, "notes": fallback_notes(pos, proj),
    }
