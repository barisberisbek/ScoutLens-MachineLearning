"""Train Stage-2 valuation models (Phase 6).

    python scripts/train_stage2.py --position all
    python scripts/train_stage2.py --position GK

Per position: Ridge / XGBoost / MLP / Stacked → log_market_value (D-02: actual observed
stats only). Saves models under models/stage2/ + metrics + val predictions.
"""

from __future__ import annotations

import argparse

from src.models.stage1.target_specs import POSITIONS
from src.models.stage2 import data_loader
from src.models.stage2.evaluate import select_best
from src.models.stage2.train import ALL_MODEL_NAMES, train_stage2
from src.utils.io import project_root, save_parquet
from src.utils.logging import get_logger

logger = get_logger("train_stage2")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Stage-2 valuation models")
    ap.add_argument("--position", default="all", help="all | GK | DEF | MID | FWD")
    ap.add_argument("--model", default=None, help="Ridge | XGBoost | MLP | Stacked")
    args = ap.parse_args()

    positions = POSITIONS if args.position == "all" else [args.position]
    models = ALL_MODEL_NAMES if args.model is None else [args.model]

    logger.info("--- Row counts (MV-observed + min-minutes) ---")
    for pos in positions:
        train, val, feats = data_loader.load(pos)
        logger.info("  %s: %d train / %d val rows (%d features)", pos, len(train), len(val), len(feats))

    metrics_df, val_pred_df = train_stage2(positions=positions, model_names=models)
    interim = project_root() / "data" / "processed"
    save_parquet(metrics_df, interim / "stage2_metrics.parquet")
    if len(val_pred_df):
        save_parquet(val_pred_df, interim / "stage2_val_predictions.parquet")

    if len(metrics_df):
        logger.info("--- Best model per position (val) ---")
        for r in select_best(metrics_df).itertuples(index=False):
            dpct = 100 * r.improvement if r.improvement == r.improvement else float("nan")
            logger.info("  %-4s %-8s MAE_log=%.3f  MAE_eur=%.2fM  R²=%.3f  Δ=%+.1f%%",
                        r.position, r.model, r.mae_log, r.mae_eur / 1e6, r.r2, dpct)
    logger.info("Done: %d metric rows, %d val predictions.", len(metrics_df), len(val_pred_df))


if __name__ == "__main__":
    main()
