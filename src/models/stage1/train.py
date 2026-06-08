"""Stage-1 training loop: per (position, target), fit Ridge/XGBoost/MLP/Stacked, persist,
and collect validation metrics + best-model predictions.
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.models.stage1 import data_loader, persist
from src.models.stage1.ensemble import StackedModel
from src.models.stage1.evaluate import naive_baseline_mae, regression_metrics
from src.models.stage1.models import BASE_MODELS
from src.models.stage1.target_specs import POSITIONS, STAGE1_TARGETS
from src.utils.logging import get_logger

logger = get_logger(__name__)

ALL_MODEL_NAMES = ["Ridge", "XGBoost", "MLP", "Stacked"]
_MIN_TRAIN = 30


def train_stage1(
    positions: list[str] | None = None,
    model_names: list[str] | None = None,
    targets_filter: list[str] | None = None,
    features_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train all (position, target, model) combos; return (metrics_df, val_predictions_df)."""
    positions = positions or POSITIONS
    model_names = model_names or ALL_MODEL_NAMES
    base_names = [m for m in model_names if m in BASE_MODELS]

    metrics_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    for position in positions:
        train, val, feats = data_loader.load(position, features_path)
        data_loader.assert_no_leakage(feats)
        targets = [t for t in STAGE1_TARGETS[position] if targets_filter is None or t in targets_filter]

        for target in tqdm(targets, desc=f"Stage1 {position}", leave=False):
            ycol = f"next_{target}"
            if ycol not in train.columns:
                continue
            tr = train[train[ycol].notna()]
            if len(tr) < _MIN_TRAIN:
                logger.warning("%s/%s: only %d train rows (<%d) — skipped", position, target, len(tr), _MIN_TRAIN)
                continue
            X_tr, y_tr = tr[feats], tr[ycol]
            ev = val[val[ycol].notna() & val[target].notna()]
            X_ev, y_ev = ev[feats], ev[ycol]
            base_mae = naive_baseline_mae(y_ev, ev[target]) if len(ev) else float("nan")

            fitted: dict = {}
            preds: dict = {}
            for mname in base_names:
                model = BASE_MODELS[mname]().fit(X_tr, y_tr)
                persist.save(model.best_estimator_, position, target, mname)
                fitted[mname] = model.best_estimator_
                pred = model.best_estimator_.predict(X_ev) if len(ev) else []
                preds[mname] = pred
                metrics_rows.append(_row(position, target, mname, len(tr), len(ev),
                                         regression_metrics(y_ev, pred), model.cv_mae_, base_mae))

            if "Stacked" in model_names and len(fitted) >= 2:
                st = StackedModel().fit(X_tr, y_tr, fitted)
                persist.save(st.model, position, target, "Stacked")
                pred = st.predict(X_ev) if len(ev) else []
                preds["Stacked"] = pred
                metrics_rows.append(_row(position, target, "Stacked", len(tr), len(ev),
                                         regression_metrics(y_ev, pred), float("nan"), base_mae))

            # store the best model's val predictions (for the projection output parquet)
            if len(ev) and preds:
                best_name = min(preds, key=lambda m: regression_metrics(y_ev, preds[m])["mae"])
                pred_frames.append(pd.DataFrame({
                    "player_id": ev["player_id"].values,
                    "current_season": ev["season"].values,
                    "position": position, "target": target, "best_model": best_name,
                    "predicted_next": preds[best_name], "actual_next": y_ev.values,
                    "naive_next": ev[target].values,
                }))

    metrics_df = pd.DataFrame(metrics_rows)
    val_pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return metrics_df, val_pred_df


def _row(position, target, model, n_train, n_val, met, cv_mae, base_mae) -> dict:
    improvement = ((base_mae - met["mae"]) / base_mae) if base_mae and base_mae == base_mae and base_mae > 0 else float("nan")
    return {"position": position, "target": target, "model": model, "n_train": n_train,
            "n_val": n_val, "mae": met["mae"], "rmse": met["rmse"], "r2": met["r2"],
            "mape": met["mape"], "cv_mae": cv_mae, "baseline_mae": base_mae,
            "improvement": improvement}
