"""Render Stage-2 reports: metrics, feature importance, and §11.4 transfer validation.

    python scripts/evaluate_stage2.py
"""

from __future__ import annotations

import pandas as pd

from src.models.stage2 import data_loader, persist
from src.models.stage2.evaluate import feature_importance, select_best
from src.models.stage2.transfer_validation import run_transfer_validation, write_report
from src.utils.io import load_parquet, project_root
from src.utils.logging import get_logger

logger = get_logger("evaluate_stage2")


def _fmt(v, nd=3):
    return "—" if v != v else f"{v:.{nd}f}"


def write_metrics_md(metrics_df: pd.DataFrame) -> None:
    best = set(zip(*select_best(metrics_df)[["position", "model"]].values.T))
    lines = ["# Stage 2 — Validation Metrics", "",
             "Target `log_market_value`; train 2021-22/22-23/23-24, validate 2024-25 (D-01). "
             "Trained on ACTUAL observed stats (D-02). Baseline = train-median predictor. "
             "`log_market_value_lag1` EXCLUDED from features (avoids 'copy last year's value'; "
             "ablation later). `MAE_eur` = mean |€| error via expm1.", "",
             "| pos | model | n_train | n_val | MAE_log | MAE_€M | medAE_€M | R² | Δ_base% |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in metrics_df.sort_values(["position", "mae_log"]).itertuples(index=False):
        star = " ★" if (r.position, r.model) in best else ""
        dpct = _fmt(100 * r.improvement, 1) if r.improvement == r.improvement else "—"
        lines.append(f"| {r.position} | {r.model}{star} | {r.n_train} | {r.n_val} | {_fmt(r.mae_log)} | "
                     f"{_fmt(r.mae_eur/1e6, 2)} | {_fmt(r.median_ae_eur/1e6, 2)} | {_fmt(r.r2)} | {dpct} |")
    lines += ["", "★ = best (lowest log-MAE) per position. R²>0.90 would warrant a leakage recheck.", ""]
    path = project_root() / "reports" / "stage2_metrics.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def write_importance_md(metrics_df: pd.DataFrame) -> None:
    lines = ["# Stage 2 — Feature Importance (best model per position)", "",
             "XGBoost = gain · Ridge = |coef| · MLP/Stacked = permutation importance.", ""]
    for r in select_best(metrics_df).sort_values("position").itertuples(index=False):
        train, val, feats = data_loader.load(r.position)
        try:
            est = persist.load(r.position, r.model)
            top = feature_importance(est, r.model, val[feats], val[data_loader.TARGET], feats, top_k=20)
        except Exception as e:  # noqa: BLE001
            logger.warning("importance failed %s/%s: %s", r.position, r.model, e)
            top = []
        lines += [f"## {r.position} · {r.model}", ""]
        lines += [f"{i+1}. `{n}` ({v:.4f})" for i, (n, v) in enumerate(top)] or ["(unavailable)"]
        lines.append("")
    path = project_root() / "reports" / "stage2_feature_importance.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def main() -> None:
    metrics_df = load_parquet(project_root() / "data" / "processed" / "stage2_metrics.parquet")
    write_metrics_md(metrics_df)
    write_importance_md(metrics_df)
    res, agg = run_transfer_validation(metrics_df)
    write_report(res, agg)
    logger.info("Transfer validation: n=%s median_ratio=%.2f pearson_log=%.3f",
                agg.get("n"), agg.get("median_ratio", float("nan")), agg.get("pearson_log", float("nan")))


if __name__ == "__main__":
    main()
