"""Stage-1 validation metrics, best-model selection, and feature importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Simpler-model preference for tie-breaking the best model per target.
_MODEL_ORDER = {"Ridge": 0, "XGBoost": 1, "MLP": 2, "Stacked": 3}


def regression_metrics(y_true, y_pred) -> dict:
    """MAE / RMSE / R² / MAPE (MAPE only on |y_true| > 0.1 to avoid blow-up near zero)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "mape": np.nan}
    mask = np.abs(y_true) > 0.1
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "mape": mape,
    }


def naive_baseline_mae(y_true, current) -> float:
    """MAE of the 'next == current' projection on rows where both are present."""
    y_true = np.asarray(y_true, dtype="float64")
    current = np.asarray(current, dtype="float64")
    mask = ~np.isnan(y_true) & ~np.isnan(current)
    return float(mean_absolute_error(y_true[mask], current[mask])) if mask.any() else np.nan


def select_best(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per (position, target): the lowest-MAE model; ties broken toward the simpler model."""
    df = metrics_df.copy()
    df["_order"] = df["model"].map(_MODEL_ORDER)
    df = df.sort_values(["mae", "_order"])
    best = df.groupby(["position", "target"], as_index=False).first().drop(columns="_order")
    return best


def feature_importance(estimator, model_name: str, X_val: pd.DataFrame, y_val,
                       feature_names: list[str], top_k: int = 20) -> list[tuple[str, float]]:
    """Top-K features: XGB gain / Ridge |coef| / permutation importance for MLP & Stacked."""
    try:
        if model_name == "XGBoost":
            imp = estimator.named_steps["model"].feature_importances_
        elif model_name == "Ridge":
            imp = np.abs(estimator.named_steps["model"].coef_)
        else:  # MLP, Stacked — model-agnostic permutation importance
            r = permutation_importance(estimator, X_val, y_val, n_repeats=3,
                                       random_state=42, scoring="neg_mean_absolute_error")
            imp = r.importances_mean
    except Exception:
        return []
    order = np.argsort(imp)[::-1][:top_k]
    return [(feature_names[i], float(imp[i])) for i in order]
