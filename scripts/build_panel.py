"""Build the unified panel (Phase 2, roadmap §5.3 / §16 Phase 2).

Runs the full source merge (FBref backbone → name resolution → split-season collapse →
xG / MV / contract / FIFA / league-meta attach → schema finalize) and writes the panel,
the resolution audit log, and the two reports. Manual overrides are optional — the panel
is deliverable without them (they are populated in a later pass from the first audit).

Usage:
    python scripts/build_panel.py --validate     # Premier League 2324 slice
    python scripts/build_panel.py                 # full 9×4 build
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.integration.unified_panel_builder import build_unified_panel
from src.integration.panel_reports import write_audit_report, write_coverage_matrix
from src.utils.io import project_root, save_parquet
from src.utils.logging import get_logger

logger = get_logger("build_panel")


def _load_manual_overrides() -> pd.DataFrame | None:
    path = project_root() / "data" / "manual" / "manual_id_overrides.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def _scaffold_overrides() -> None:
    """Create an empty manual_id_overrides.csv with the documented schema if absent."""
    path = project_root() / "data" / "manual" / "manual_id_overrides.csv"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["variant_name", "normalized_name", "birth_year", "nationality",
            "target_player_id", "source", "notes"]
    pd.DataFrame(columns=cols).to_csv(path, index=False, encoding="utf-8")
    logger.info("Scaffolded empty manual override table → %s", path)


def _summary(panel: pd.DataFrame) -> None:
    n = len(panel)
    logger.info("--- Unified panel: %d rows × %d cols ---", n, panel.shape[1])
    logger.info("  leagues=%d seasons=%d distinct players=%d",
                panel["league"].nunique(), panel["season"].nunique(), panel["player_id"].nunique())
    logger.info("  MV non-null: %d (%.1f%%) | xG non-null: %d | FIFA non-null: %d",
                panel["market_value_eur"].notna().sum(), 100 * panel["market_value_eur"].notna().mean(),
                panel["xg"].notna().sum(), panel["fifa_rating"].notna().sum())
    logger.info("  xg_source: %s", panel["xg_source"].value_counts().to_dict())
    logger.info("  age_precision: %s", panel["age_precision"].value_counts().to_dict())

    # hard asserts (fail loudly)
    assert panel.duplicated(["player_id", "season"]).sum() == 0
    if n > 5000:  # full build
        assert panel["league"].nunique() == 9, "expected 9 leagues"
        assert panel["season"].nunique() == 4, "expected 4 seasons"
        mv2425_top5 = panel[(panel.season == "2024-25") & panel.league.isin(
            ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"])]["market_value_eur"].notna().sum()
        logger.info("  MV non-null 2024-25 top-5: %d (TM had ~1880)", mv2425_top5)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the unified panel (Phase 2)")
    ap.add_argument("--validate", action="store_true", help="Premier League 2324 slice")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    _scaffold_overrides()
    overrides = _load_manual_overrides()

    if args.validate:
        panel, resolver, orphans = build_unified_panel(
            leagues=["Premier League"], seasons=["2324"], manual_overrides=overrides)
    else:
        panel, resolver, orphans = build_unified_panel(manual_overrides=overrides)

    _summary(panel)

    # outputs
    out = project_root() / "data" / "processed" / (
        "unified_panel_sample.parquet" if args.validate else "unified_panel.parquet")
    save_parquet(panel, out)
    logger.info("Saved unified panel → %s", out)

    audit = resolver.audit_frame()
    audit_csv = project_root() / "data" / "manual" / "match_log.csv"
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_csv, index=False, encoding="utf-8")
    logger.info("Wrote %d audit rows → %s", len(audit), audit_csv)

    if not args.validate:
        write_audit_report(audit, panel, orphans)
        write_coverage_matrix(panel)


if __name__ == "__main__":
    main()
