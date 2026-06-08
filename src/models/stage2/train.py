"""Stage-2 training: per position, fit Ridge/XGBoost/MLP/Stacked → log_market_value."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from src.models.stage1.target_specs import POSITIONS
from src.models.stage2 import data_loader, persist
from src.models.stage2.ensemble import StackedModel
from src.models.stage2.evaluate import euro_space_metrics, regression_metrics
from src.models.stage2.models import BASE_MODELS
from src.utils.logging import get_logger

logger = get_logger(__name__)

ALL_MODEL_NAMES = ["Ridge", "XGBoost", "MLP", "Stacked"]
TARGET = data_loader.TARGET


def train_stage2(
    positions: list[str] | None = None,
    model_names: list[str] | None = None,
    features_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train all (position, model) combos; return (metrics_df, val_predictions_df)."""
    positions = positions or POSITIONS
    model_names = model_names or ALL_MODEL_NAMES
    base_names = [m for m in model_names if m in BASE_MODELS]

    metrics_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    for position in tqdm(positions, desc="Stage2"):
        train, val, feats = data_loader.load(position, features_path)
        data_loader.assert_no_leakage(feats)
        X_tr, y_tr = train[feats], train[TARGET]
        X_va, y_va = val[feats], val[TARGET]
        base_mae = mean_absolute_error(y_va, np.full(len(y_va), y_tr.median())) if len(y_va) else float("nan")

        fitted, preds = {}, {}
        for mname in base_names:
            model = BASE_MODELS[mname]().fit(X_tr, y_tr)
            persist.save(model.best_estimator_, position, mname)
            fitted[mname] = model.best_estimator_
            preds[mname] = model.best_estimator_.predict(X_va) if len(X_va) else []
            metrics_rows.append(_row(position, mname, len(X_tr), len(X_va), y_va, preds[mname], model.cv_mae_, base_mae))

        if "Stacked" in model_names and len(fitted) >= 2:
            st = StackedModel().fit(X_tr, y_tr, fitted)
            persist.save(st.model, position, "Stacked")
            preds["Stacked"] = st.predict(X_va) if len(X_va) else []
            metrics_rows.append(_row(position, "Stacked", len(X_tr), len(X_va), y_va, preds["Stacked"], float("nan"), base_mae))

        if len(X_va) and preds:
            best = min(preds, key=lambda m: regression_metrics(y_va, preds[m])["mae"])
            pred_frames.append(pd.DataFrame({
                "player_id": val["player_id"].values, "player_name": val["player_name"].values,
                "season": val["season"].values, "position": position, "best_model": best,
                "predicted_log_mv": preds[best], "actual_log_mv": y_va.values,
                "predicted_mv_eur": np.expm1(preds[best]), "actual_mv_eur": np.expm1(y_va.values),
            }))

    return pd.DataFrame(metrics_rows), (pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame())


def _row(position, model, n_train, n_val, y_va, pred, cv_mae, base_mae) -> dict:
    m = regression_metrics(y_va, pred)
    e = euro_space_metrics(y_va, pred)
    improvement = ((base_mae - m["mae"]) / base_mae) if base_mae and base_mae == base_mae and base_mae > 0 else float("nan")
    return {"position": position, "model": model, "n_train": n_train, "n_val": n_val,
            "mae_log": m["mae"], "rmse_log": m["rmse"], "r2": m["r2"],
            "mae_eur": e["mae_eur"], "median_ae_eur": e["median_ae_eur"],
            "cv_mae": cv_mae, "baseline_mae_log": base_mae, "improvement": improvement,
            # alias for shared select_best (which keys on 'mae')
            "mae": m["mae"]}
