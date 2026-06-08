"""Run the 3-layer pipeline validation + the end-to-end examples report (Phase 7).

    python scripts/validate_pipeline.py
"""

from __future__ import annotations

import pandas as pd

from src.models.stage2.transfer_validation import run_transfer_validation
from src.models.stage2.transfer_validation import write_report as write_transfer_report
from src.pipeline.confidence import compute_pipeline_residual_quantiles
from src.pipeline.inference import predict_player_value
from src.pipeline.validation import run_layer3, write_report
from src.utils.io import load_parquet, project_root, save_parquet
from src.utils.logging import get_logger

logger = get_logger("validate_pipeline")

# Archetype examples (resolved by name substring in the 2023-24 cohort).
_STARS = ["Bellingham", "Haaland", "Vinicius", "Vinícius"]
_MID = ["Gündogan", "Gundogan", "Koundé", "Kounde"]


def _example(features: pd.DataFrame, name: str, season: str = "2023-24") -> dict | None:
    hit = features[(features["season"] == season)
                   & features["player_name"].str.contains(name, case=False, na=False)]
    if hit.empty:
        return None
    pid = hit.iloc[0]["player_id"]
    res = predict_player_value(features, pid, season)
    nxt = features[(features["player_id"] == pid) & (features["season"] == res["next_season"])]
    res["actual_next_mv"] = float(nxt.iloc[0]["market_value_eur"]) if len(nxt) and pd.notna(nxt.iloc[0]["market_value_eur"]) else None
    return res


def write_examples(features: pd.DataFrame, e2e: pd.DataFrame) -> None:
    chosen, seen = [], set()
    for name in _STARS + _MID:
        r = _example(features, name)
        if r and r["player_id"] not in seen:
            chosen.append(r); seen.add(r["player_id"])
    # 2-3 discovery candidates: biggest positive projected delta with an observed current MV
    disc = (e2e[e2e["observed_current_mv"].notna()].sort_values("mv_delta_pct", ascending=False).head(3))
    for r in disc.itertuples(index=False):
        if r.player_id in seen:
            continue
        res = predict_player_value(features, r.player_id, "2023-24")
        res["actual_next_mv"] = float(r.actual_mv) if pd.notna(r.actual_mv) else None
        chosen.append(res); seen.add(r.player_id)

    lines = ["# Pipeline — End-to-End Examples", "",
             "Each player forwarded from 2023-24 → 2024-25: current valuation, Stage-1 stat "
             "projections, projected next-season value (+95% CI), and the actual 2024-25 value.", ""]
    def m(x):
        return "—" if x is None else f"€{x/1e6:.1f}M"
    for r in chosen:
        lines += [f"## {r['player_name']} ({r['position']})", "",
                  f"- observed current MV (23-24): {m(r['observed_current_mv'])} · model current estimate: {m(r['current_mv_estimate'])}",
                  f"- **projected 24-25 value: {m(r['projected_mv_next_season'])}** "
                  f"(95% CI {m(r['confidence_interval'][0])}–{m(r['confidence_interval'][1])}) · "
                  f"actual 24-25: {m(r.get('actual_next_mv'))}",
                  f"- Δ vs current: {r['mv_delta_pct']:+.0f}%" if r["mv_delta_pct"] is not None else "- Δ: —",
                  f"- Stage-1 projections: " + ", ".join(f"{k}={v:.2f}" for k, v in r["stage1_projections"].items()),
                  ""]
    path = project_root() / "reports" / "pipeline_e2e_examples.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def main() -> None:
    compute_pipeline_residual_quantiles()              # CI from full-pipeline residuals
    # regenerate §11.4 transfer report with the naive (TM MV) comparison
    s2_metrics = load_parquet(project_root() / "data" / "processed" / "stage2_metrics.parquet")
    res_t, transfer_agg = run_transfer_validation(s2_metrics)
    write_transfer_report(res_t, transfer_agg)
    # Layer-3 + 3-layer report (with the honest transfer comparison woven in)
    pipe, metrics = run_layer3()
    write_report(pipe, metrics, transfer_agg)
    save_parquet(pipe, project_root() / "data" / "processed" / "pipeline_e2e_predictions.parquet")
    features = load_parquet(project_root() / "data" / "processed" / "features.parquet")
    write_examples(features, pipe)
    logger.info("Done.")


if __name__ == "__main__":
    main()
