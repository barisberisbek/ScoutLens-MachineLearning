"""Build the model-ready feature matrix (Phase 4).

Loads the unified panel, runs the feature pipeline, asserts invariants, saves
`data/processed/features.parquet`, and writes `reports/features_summary.md`.

    python scripts/build_features.py
"""

from __future__ import annotations

import pandas as pd

from src.features.build_features import build_features
from src.utils.io import load_parquet, project_root, save_parquet
from src.utils.logging import get_logger

logger = get_logger("build_features")


def _group_of(col: str) -> str:
    """Bucket a new feature column for the summary report."""
    if col.endswith("_z_pos") or col.endswith("_z_league"):
        return "z-scores (M3)"
    if col.endswith("_lag1") or col.startswith("delta_") or col in ("has_lag1", "consecutive_seasons"):
        return "lag / trajectory (M6)"
    if col.endswith("_per_90") or col.endswith("_per_match"):
        return "per-90 / per-match (M1)"
    if col in ("goal_threat", "creative_threat", "defensive_actions", "shooting_efficiency",
               "attacking_output", "composite_completeness"):
        return "composites (M2)"
    if col in ("age_squared", "age_cubed", "distance_from_peak", "is_young", "is_peak", "is_declining"):
        return "age curve (M4)"
    if col in ("contract_years_until_expiry", "is_expiring", "is_long_contract"):
        return "contract (M5)"
    if col in ("year_inflation_multiplier",):
        return "multipliers (M7)"
    if col.startswith("is_") and ("continent_" in col or col[3:] in ("GK", "DEF", "MID", "FWD")):
        return "categorical (M8)"
    if col in ("fifa_potential_gap", "delta_fifa_rating"):
        return "fifa (M9)"
    return "other"


def _write_summary(panel: pd.DataFrame, features: pd.DataFrame) -> None:
    new_cols = [c for c in features.columns if c not in panel.columns]
    lines = ["# Features Summary (Phase 4)", "",
             f"`unified_panel.parquet` ({panel.shape[1]} cols) → `features.parquet` "
             f"({features.shape[1]} cols): **+{len(new_cols)} engineered features**, "
             f"{len(features):,} rows.", ""]

    # grouped column list with null %
    groups: dict[str, list[str]] = {}
    for c in new_cols:
        groups.setdefault(_group_of(c), []).append(c)
    for grp in ["per-90 / per-match (M1)", "composites (M2)", "z-scores (M3)", "age curve (M4)",
                "contract (M5)", "lag / trajectory (M6)", "multipliers (M7)", "categorical (M8)",
                "fifa (M9)", "other"]:
        cols = groups.get(grp)
        if not cols:
            continue
        lines += [f"## {grp} ({len(cols)})", "", "| column | % null |", "|---|---|"]
        for c in cols:
            lines.append(f"| {c} | {100 * features[c].isna().mean():.1f} |")
        lines.append("")

    # per-position null breakdown for the new cols (GK vs outfield mostly differ on GK stats)
    lines += ["## % null by position (new features, mean over group)", "",
              "| position | mean % null (new cols) |", "|---|---|"]
    for pos, g in features.groupby("primary_position"):
        lines.append(f"| {pos} | {100 * g[new_cols].isna().mean().mean():.1f} |")
    lines.append("")

    # top-10 highest-null new columns (debug aid)
    top = features[new_cols].isna().mean().sort_values(ascending=False).head(10)
    lines += ["## Top-10 highest-null new columns (debug)", "", "| column | % null |", "|---|---|"]
    for c, v in top.items():
        lines.append(f"| {c} | {100 * v:.1f} |")
    lines.append("")
    lines += ["Note: high null is expected for xG-based features (top-5 only), GK stats (null "
              "for outfield), lag features in 2021-22 (no prior season), and 2024-25 "
              "`fifa_potential_gap` (FC25 has no potential). No imputation is done here — Phase 5 "
              "handles NaN (tree models natively, linear via a pipeline imputer).", ""]

    path = project_root() / "reports" / "features_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def main() -> None:
    panel = load_parquet(project_root() / "data" / "processed" / "unified_panel.parquet")
    features = build_features(panel)

    # hard invariants
    assert len(features) == len(panel), "row count changed"
    assert set(panel.columns).issubset(features.columns), "an original column was dropped/renamed"
    assert features.duplicated(["player_id", "season"]).sum() == 0, "duplicate (player_id, season)"
    n_new = features.shape[1] - panel.shape[1]
    assert 50 <= n_new <= 120, f"unexpected new-column count: {n_new}"
    logger.info("Invariants OK: %d rows, +%d features", len(features), n_new)

    out = project_root() / "data" / "processed" / "features.parquet"
    save_parquet(features, out)
    logger.info("Saved %s (%d × %d)", out, *features.shape)
    _write_summary(panel, features)


if __name__ == "__main__":
    main()
