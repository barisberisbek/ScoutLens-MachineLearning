"""Phase-2 data-quality reports: name-resolution audit + coverage matrix.

Pure reporting (no panel mutation). Reads the resolver audit buffer and the finished
panel, writes two committed markdown reports under ``reports/``.
"""

from __future__ import annotations

import pandas as pd

from src.utils.constants import ALL_SEASONS, TOP5_LEAGUES
from src.utils.io import project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MATCHED = ["exact", "exact_name_year", "fuzzy_nat", "fuzzy_team", "manual"]


def write_audit_report(audit: pd.DataFrame, panel: pd.DataFrame,
                       understat_orphans: pd.DataFrame) -> None:
    """Write reports/name_resolution_audit.md (methods, Understat orphans, spot-checks)."""
    lines: list[str] = ["# Name Resolution Audit (Phase 2)", ""]
    n = len(audit)
    lines += [f"FBref backbone resolution decisions: **{n}** rows.", ""]

    lines += ["## Resolution method distribution", "", "| method | count | % |", "|---|---|---|"]
    for method, cnt in audit["method"].value_counts().items():
        lines.append(f"| {method} | {cnt} | {100 * cnt / n:.1f}% |")
    matched = audit["method"].isin(_MATCHED).sum()
    lines += ["", f"**TM-matched: {matched} ({100 * matched / n:.1f}%)** · "
              f"synthetic/unmatched: {n - matched} ({100 * (n - matched) / n:.1f}%)", ""]

    # by league (from the panel, which carries the final method)
    lines += ["## TM-match rate by league", "", "| league | rows | matched % |", "|---|---|---|"]
    for lg, g in panel.groupby("league"):
        lines.append(f"| {lg} | {len(g)} | {100 * g['resolve_method'].isin(_MATCHED).mean():.1f}% |")
    lines.append("")

    # Understat orphans
    n_orph = len(understat_orphans)
    lines += ["## Understat orphans (matched to FBref backbone)", ""]
    lines += [f"Unmatched Understat player-rows: **{n_orph}**. These are logged, never "
              "silently dropped; an orphan rate above 10% would signal an FBref coverage gap.", ""]
    if n_orph:
        sample = understat_orphans.head(25)
        lines += ["| player | team | season |", "|---|---|---|"]
        for r in sample.itertuples(index=False):
            lines.append(f"| {r.player} | {r.team} | {r.season} |")
        if n_orph > 25:
            lines.append(f"| … | … | (+{n_orph - 25} more) |")
        lines.append("")

    # lowest-score accepted fuzzy matches for manual spot-check
    fuzzy = audit[audit["method"].isin(["fuzzy_nat", "fuzzy_team"])].sort_values("score").head(30)
    lines += ["## 30 lowest-score accepted fuzzy matches (spot-check these)", "",
              "| source | input_name | birth_year | nationality | score | player_id |",
              "|---|---|---|---|---|---|"]
    for r in fuzzy.itertuples(index=False):
        lines.append(f"| {r.source} | {r.input_name} | {r.input_birth_year} | "
                     f"{r.input_nationality} | {r.score} | {r.resolved_player_id} |")
    lines.append("")

    path = project_root() / "reports" / "name_resolution_audit.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def write_coverage_matrix(panel: pd.DataFrame) -> None:
    """Write reports/coverage_matrix.md (per-source non-null coverage by season × league)."""
    sources = {
        "minutes (FBref)": "minutes_played",
        "xG": "xg",
        "market_value": "market_value_eur",
        "FIFA": "fifa_rating",
        "contract": "contract_end_date",
    }
    lines: list[str] = ["# Coverage Matrix (Phase 2)", "",
                        f"Unified panel: **{len(panel)}** (player, season) rows.", ""]

    lines += ["## Non-null coverage by source × season (%)", "",
              "| source | " + " | ".join(ALL_SEASONS) + " |",
              "|---|" + "---|" * len(ALL_SEASONS)]
    for label, col in sources.items():
        row = [label]
        for ssn in ALL_SEASONS:
            g = panel[panel["season"] == ssn]
            row.append(f"{100 * g[col].notna().mean():.0f}" if len(g) else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # MV coverage by league × season (the modeling-relevant view)
    lines += ["## market_value non-null COUNT by league × season", "",
              "| league | " + " | ".join(ALL_SEASONS) + " |",
              "|---|" + "---|" * len(ALL_SEASONS)]
    for lg, g in panel.groupby("league"):
        row = [lg]
        for ssn in ALL_SEASONS:
            row.append(str(int(g[g["season"] == ssn]["market_value_eur"].notna().sum())))
        lines.append("| " + " | ".join(row) + " |")
    lines += ["",
              "Note: xG is available for top-5 only (Kaggle 2024-25, Understat historical); "
              "lower-4 xG is NaN by design (P2-D3 / Phase-4 decision). FIFA potential is null "
              "for 2024-25 (FC25). Contract dates ~64% present (TM snapshot).", ""]

    path = project_root() / "reports" / "coverage_matrix.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
