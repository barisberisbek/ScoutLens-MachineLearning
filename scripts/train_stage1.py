"""Train Stage-1 projection models (Phase 5).

    python scripts/train_stage1.py --position all
    python scripts/train_stage1.py --position GK
    python scripts/train_stage1.py --position MID --model Ridge --target goals_per_90

Trains 4 model types per (position, target), saves them under models/stage1/, and writes the
raw validation metrics + best-model val predictions to data/processed/.
"""

from __future__ import annotations

import argparse

from src.models.stage1 import data_loader
from src.models.stage1.target_specs import POSITIONS
from src.models.stage1.train import ALL_MODEL_NAMES, train_stage1
from src.utils.io import project_root, save_parquet
from src.utils.logging import get_logger

logger = get_logger("train_stage1")


def _report_pairs(positions: list[str]) -> None:
    logger.info("--- Pair counts ---")
    for pos in positions:
        train, val, feats = data_loader.load(pos)
        logger.info("  %s: %d train / %d val pairs (%d features)", pos, len(train), len(val), len(feats))
        if pos == positions[0] and len(train):
            sample = train.iloc[0]
            logger.info("  sample pair: player=%s %s→%s pos=%s",
                        sample["player_name"], sample["season"], sample["next_season"], pos)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Stage-1 projection models")
    ap.add_argument("--position", default="all", help="all | GK | DEF | MID | FWD")
    ap.add_argument("--model", default=None, help="Ridge | XGBoost | MLP | Stacked (default all)")
    ap.add_argument("--target", default=None, help="restrict to a single target stat")
    args = ap.parse_args()

    positions = POSITIONS if args.position == "all" else [args.position]
    models = ALL_MODEL_NAMES if args.model is None else [args.model]
    targets = None if args.target is None else [args.target]

    _report_pairs(positions)
    logger.info("Training: positions=%s models=%s targets=%s", positions, models, targets or "all")

    metrics_df, val_pred_df = train_stage1(positions=positions, model_names=models, targets_filter=targets)

    interim = project_root() / "data" / "processed"
    save_parquet(metrics_df, interim / "stage1_metrics.parquet")
    if len(val_pred_df):
        save_parquet(val_pred_df, interim / "stage1_val_predictions.parquet")
    logger.info("Done: %d metric rows, %d val predictions. Saved to data/processed/.",
                len(metrics_df), len(val_pred_df))

    # quick summary: best model per target + improvement over naive baseline
    if len(metrics_df):
        from src.models.stage1.evaluate import select_best
        best = select_best(metrics_df)
        logger.info("--- Best model per target (val MAE | Δ vs naive) ---")
        for r in best.itertuples(index=False):
            logger.info("  %-4s %-22s %-8s MAE=%.4f  Δ=%+.1f%%  R²=%.2f",
                        r.position, r.target, r.model, r.mae,
                        100 * r.improvement if r.improvement == r.improvement else float("nan"), r.r2)


if __name__ == "__main__":
    main()
