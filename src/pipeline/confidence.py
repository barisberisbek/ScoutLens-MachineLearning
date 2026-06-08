"""Confidence intervals via Stage-2 residual quantiles (per position).

At build time we predict the best Stage-2 model on its own training rows and store the
residual quantiles (log space). At inference, CI = prediction + {q05, q95} → expm1 to euros.
Simple, defensible, and honest about the model's typical spread.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np

from src.models.stage1.target_specs import POSITIONS
from src.models.stage2 import data_loader as s2_loader
from src.pipeline.loaders import best_stage2
from src.utils.io import load_parquet, project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

_QUANTS = [5, 25, 50, 75, 95]


def _path():
    return project_root() / "models" / "pipeline_residual_quantiles.pkl"


def compute_residual_quantiles(features_path: str | None = None) -> dict:
    """Per-position residual quantiles of the best Stage-2 model on its train set → saved pkl."""
    models = best_stage2()
    out: dict[str, dict] = {}
    for pos in POSITIONS:
        train, _, feats = s2_loader.load(pos, features_path)
        est = models[pos]
        resid = train[s2_loader.TARGET].to_numpy() - est.predict(train[feats])
        out[pos] = {f"q{q:02d}": float(np.percentile(resid, q)) for q in _QUANTS}
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, path)
    logger.info("Saved residual quantiles → %s", path)
    return out


def compute_pipeline_residual_quantiles(features_path: str | None = None) -> dict:
    """Per-position residual quantiles of the FULL pipeline (Stage1→Forwarder→Stage2).

    Uses the 2023-24→2024-25 end-to-end residuals (actual 24-25 log-MV − projected), so the CI
    reflects the whole pipeline's spread, not just Stage-2's. (Single-split: calibrated on the
    same season it's reported against, so coverage is ~nominal by construction — documented.)
    """
    from src.pipeline.inference import predict_batch  # lazy: avoid circular import

    f = load_parquet(features_path or (project_root() / "data" / "processed" / "features.parquet"))
    pre = f[(f["season"] == "2023-24") & f["meets_min_minutes"].fillna(False)]
    pred = predict_batch(pre, with_ci=False)
    actual = (f[(f["season"] == "2024-25") & f["log_market_value"].notna()]
              [["player_id", "log_market_value"]].rename(columns={"log_market_value": "actual_log"}))
    merged = pred.merge(actual, on="player_id", how="inner")
    merged["pred_log"] = np.log1p(merged["projected_mv_next"])
    out: dict[str, dict] = {}
    for pos in POSITIONS:
        resid = (merged.loc[merged["position"] == pos, "actual_log"]
                 - merged.loc[merged["position"] == pos, "pred_log"]).to_numpy()
        if len(resid) >= 10:
            out[pos] = {f"q{q:02d}": float(np.percentile(resid, q)) for q in _QUANTS}
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, path)
    load_quantiles.cache_clear()
    logger.info("Saved pipeline residual quantiles → %s (n positions=%d)", path, len(out))
    return out


@lru_cache(maxsize=1)
def load_quantiles() -> dict:
    """Load saved residual quantiles (compute + save if missing)."""
    if _path().exists():
        return joblib.load(_path())
    return compute_residual_quantiles()


def confidence_interval(pred_log: float, position: str, level: int = 95) -> tuple[float, float]:
    """Return (low_eur, high_eur) for a log-space prediction at the given symmetric level."""
    q = load_quantiles()[position]
    lo_key, hi_key = (("q05", "q95") if level == 95 else ("q25", "q75"))
    return float(np.expm1(pred_log + q[lo_key])), float(np.expm1(pred_log + q[hi_key]))


def ci_bounds_array(pred_log, position: str, level: int = 95):
    """Vectorized CI for an array of log predictions."""
    q = load_quantiles()[position]
    lo_key, hi_key = (("q05", "q95") if level == 95 else ("q25", "q75"))
    pred_log = np.asarray(pred_log, dtype="float64")
    return np.expm1(pred_log + q[lo_key]), np.expm1(pred_log + q[hi_key])
