"""Stage-2 metrics: log-space (reused) + euro-space (expm1) for real-world interpretation."""

from __future__ import annotations

import numpy as np
import pandas as pd

# log-space metrics + importance are identical to Stage 1; select_best differs (no target dim).
from src.models.stage1.evaluate import _MODEL_ORDER, feature_importance, regression_metrics  # noqa: F401


def select_best(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per position: lowest-MAE model; ties broken toward the simpler model (no target dim)."""
    df = metrics_df.copy()
    df["_order"] = df["model"].map(_MODEL_ORDER)
    df = df.sort_values(["mae", "_order"])
    return df.groupby("position", as_index=False).first().drop(columns="_order")


def euro_space_metrics(y_true_log, y_pred_log) -> dict:
    """MAE / median-AE in euros, from log-space predictions (expm1 inverse of log1p)."""
    yt = np.expm1(np.asarray(y_true_log, dtype="float64"))
    yp = np.expm1(np.asarray(y_pred_log, dtype="float64"))
    if len(yt) == 0:
        return {"mae_eur": np.nan, "median_ae_eur": np.nan}
    ae = np.abs(yp - yt)
    return {"mae_eur": float(ae.mean()), "median_ae_eur": float(np.median(ae))}
