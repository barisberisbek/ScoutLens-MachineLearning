"""Render Stage-1 validation reports from the trained models + saved metrics (Phase 5).

    python scripts/evaluate_stage1.py

Writes reports/stage1_metrics.md (full per-model table + best-per-target, GK flagged) and
reports/stage1_feature_importance.md (top-20 features for the best model per target).
"""

from __future__ import annotations

import pandas as pd

from src.models.stage1 import data_loader, persist
from src.models.stage1.evaluate import feature_importance, select_best
from src.utils.io import load_parquet, project_root
from src.utils.logging import get_logger

logger = get_logger("evaluate_stage1")


def _fmt(v, nd=4):
    return "—" if v != v else f"{v:.{nd}f}"


def write_metrics_md(metrics_df: pd.DataFrame) -> None:
    best = select_best(metrics_df)
    best_key = set(zip(best.position, best.target, best.model))
    lines = ["# Stage 1 — Validation Metrics", "",
             "Temporal split (D-01): train = (2021-22→22-23)+(2022-23→23-24), "
             "validation = (2023-24→24-25). Baseline = naive 'next == current'. "
             "`Δ%` = improvement over baseline MAE (positive = beats naive).", "",
             "## Best model per target", "",
             "| pos | target | model | n_train | n_val | MAE | baseline | Δ% | R² |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in best.sort_values(["position", "target"]).itertuples(index=False):
        flag = " ⚠️" if r.position == "GK" else ""
        dpct = _fmt(100 * r.improvement, 1) if r.improvement == r.improvement else "—"
        lines.append(f"| {r.position}{flag} | {r.target} | {r.model} | {r.n_train} | {r.n_val} | "
                     f"{_fmt(r.mae)} | {_fmt(r.baseline_mae)} | {dpct} | {_fmt(r.r2,2)} |")
    lines += ["", "⚠️ GK targets train on only n=325 pairs — metrics are higher-variance.", "",
              "## All models", "",
              "| pos | target | model | n_train | n_val | MAE | RMSE | R² | MAPE% | Δ% |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in metrics_df.sort_values(["position", "target", "mae"]).itertuples(index=False):
        star = " ★" if (r.position, r.target, r.model) in best_key else ""
        dpct = _fmt(100 * r.improvement, 1) if r.improvement == r.improvement else "—"
        lines.append(f"| {r.position} | {r.target} | {r.model}{star} | {r.n_train} | {r.n_val} | "
                     f"{_fmt(r.mae)} | {_fmt(r.rmse)} | {_fmt(r.r2,2)} | {_fmt(r.mape,1)} | {dpct} |")
    path = project_root() / "reports" / "stage1_metrics.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def write_importance_md(metrics_df: pd.DataFrame) -> None:
    best = select_best(metrics_df)
    lines = ["# Stage 1 — Feature Importance (best model per target)", "",
             "XGBoost = gain · Ridge = |coef| · MLP/Stacked = permutation importance (3 repeats).", ""]
    cache: dict[str, tuple] = {}
    for r in best.sort_values(["position", "target"]).itertuples(index=False):
        if r.position not in cache:
            cache[r.position] = data_loader.load(r.position)
        train, val, feats = cache[r.position]
        ycol = f"next_{r.target}"
        ev = val[val[ycol].notna() & val[r.target].notna()]
        try:
            est = persist.load(r.position, r.target, r.model)
            top = feature_importance(est, r.model, ev[feats], ev[ycol], feats, top_k=20)
        except Exception as e:  # noqa: BLE001
            logger.warning("importance failed for %s/%s/%s: %s", r.position, r.target, r.model, e)
            top = []
        lines.append(f"## {r.position} · {r.target} · {r.model}")
        lines.append("")
        lines += [f"{i+1}. `{name}` ({val:.4f})" for i, (name, val) in enumerate(top)] or ["(unavailable)"]
        lines.append("")
    path = project_root() / "reports" / "stage1_feature_importance.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def main() -> None:
    metrics_path = project_root() / "data" / "processed" / "stage1_metrics.parquet"
    metrics_df = load_parquet(metrics_path)
    write_metrics_md(metrics_df)
    write_importance_md(metrics_df)


if __name__ == "__main__":
    main()
