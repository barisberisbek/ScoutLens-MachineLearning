"""Run the full valuation pipeline (Phase 7).

    python scripts/run_pipeline.py --player "Bellingham" --season 2023-24   # single
    python scripts/run_pipeline.py --batch --season 2024-25                  # batch → parquet
"""

from __future__ import annotations

import argparse
import json

from src.pipeline.inference import predict_batch, predict_player_value
from src.utils.io import load_parquet, project_root, save_parquet
from src.utils.logging import get_logger

logger = get_logger("run_pipeline")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Stage1→Forwarder→Stage2 pipeline")
    ap.add_argument("--player", default=None, help="player name substring (single prediction)")
    ap.add_argument("--season", default="2024-25", help="current season to forward FROM")
    ap.add_argument("--batch", action="store_true", help="predict all players for --season → parquet")
    args = ap.parse_args()

    features = load_parquet(project_root() / "data" / "processed" / "features.parquet")

    if args.batch:
        out = predict_batch(features, current_season=args.season)
        path = project_root() / "data" / "processed" / "pipeline_batch_predictions.parquet"
        save_parquet(out, path)
        logger.info("Batch: %d players (%s) → %s", len(out), args.season, path)
        return

    if not args.player:
        ap.error("provide --player NAME or --batch")
    hit = features[(features["season"] == args.season)
                   & features["player_name"].str.contains(args.player, case=False, na=False)]
    if hit.empty:
        logger.error("no %s row matching '%s'", args.season, args.player)
        return
    pid = hit.iloc[0]["player_id"]
    result = predict_player_value(features, pid, args.season)
    result["confidence_interval"] = [round(x / 1e6, 1) for x in result["confidence_interval"]]
    result["stage1_projections"] = {k: round(v, 3) for k, v in result["stage1_projections"].items()}
    print(json.dumps({k: (round(v / 1e6, 2) if isinstance(v, float) and abs(v) > 1e4 else v)
                      for k, v in result.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
