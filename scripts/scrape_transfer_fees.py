"""CLI: scrape 2024-25 top-5 transfer fees → CSV + coverage + davidcariboo cross-validation.

    python scripts/scrape_transfer_fees.py            # uses HTML cache if present
    python scripts/scrape_transfer_fees.py --force    # re-fetch pages
"""

from __future__ import annotations

import argparse
import re
import unicodedata

import pandas as pd

from src.data.transfer_fee_scraper import scrape_all_top5_transfers
from src.utils.constants import TOP5_LEAGUES
from src.utils.io import load_parquet, project_root


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", s.lower()).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-fetch pages (ignore HTML cache).")
    args = parser.parse_args()

    root = project_root()
    df = scrape_all_top5_transfers(season_id=2024, force=args.force)
    out_csv = root / "data" / "raw" / "transfer_fees_2025" / "transfer_fees_2024_25.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    fee = pd.to_numeric(df["transfer_fee_eur"], errors="coerce")
    paid = df[fee > 0]
    n_total, n_paid = len(df), len(paid)
    n_20m, n_60m = int((fee >= 20e6).sum()), int((fee >= 60e6).sum())
    max_fee = fee.max()

    print(f"\n{'=' * 64}\nSaved: {out_csv}  (shape={df.shape})")
    print(f"Total arrivals: {n_total} | fee>0: {n_paid}")
    print("\nPer-league (all arrivals):")
    print(df["to_league"].value_counts().to_string())
    if n_paid:
        print(f"\nFee EUR — median {paid['transfer_fee_eur'].median():,.0f} | "
              f"mean {paid['transfer_fee_eur'].mean():,.0f} | max {max_fee:,.0f}")
        print(f"  fees >= €20M: {n_20m} | >= €60M: {n_60m}")

    # ── davidcariboo comparison (join tm_transfers -> tm_players for names) ──
    lines = ["# Transfer Fees Cross-Validation (2024-25 top-5)", ""]
    lines.append("- Source A: custom Transfermarkt scrape (this phase)")
    lines.append("- Source B: davidcariboo `tm_transfers.parquet` (Phase 1C, fee>0)\n")
    try:
        tm_tr = load_parquet(root / "data" / "interim" / "tm_transfers.parquet")
        tm_pl = load_parquet(root / "data" / "interim" / "tm_players.parquet")
        b = tm_tr.merge(tm_pl[["tm_player_id", "player_name"]], on="tm_player_id", how="left")
        b_names = {_norm(n) for n in b["player_name"].dropna()}
        a_names = {_norm(n) for n in paid["player_name"].dropna()}
        inter = a_names & b_names
        lines += [
            "## Counts",
            f"- Source A total arrivals: {n_total}",
            f"- Source A fee>0: {n_paid}",
            f"- Source B fee>0: {len(b)}",
            f"- Name overlap (A∩B): {len(inter)}",
            f"- Source A max fee: €{max_fee:,.0f} | >=€20M: {n_20m} | >=€60M: {n_60m}",
            "",
            "## Big transfers (>=€20M) found in scrape, ABSENT from davidcariboo",
        ]
        paid_fee = pd.to_numeric(paid["transfer_fee_eur"], errors="coerce")
        big_new = paid[(paid_fee >= 20e6) & (~paid["player_name"].map(_norm).isin(b_names))]
        for _, r in big_new.sort_values("transfer_fee_eur", ascending=False).head(10).iterrows():
            lines.append(f"- {r['player_name']}: €{r['transfer_fee_eur']:,.0f} "
                         f"({r['from_club']} → {r['to_club']}, {r['to_league']})")
        print(f"\nvs davidcariboo: A fee>0={n_paid}, B fee>0={len(b)}, name-overlap={len(inter)}")
    except FileNotFoundError:
        lines.append("_(tm_transfers/tm_players not found — run Phase 1C first.)_")
        print("\n[WARN] tm_transfers/tm_players missing — skipped davidcariboo comparison.")

    report = root / "reports" / "transfer_fees_cross_validation.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Cross-validation report: {report}")

    # ── Validation ──
    assert n_total > 500, f"Expected >500 arrivals, got {n_total}"
    assert n_paid > 200, f"Expected >200 fee>0, got {n_paid}"
    assert df["to_league"].isin(TOP5_LEAGUES).all(), "Non-top-5 to_league present"
    if pd.isna(max_fee) or max_fee < 30e6:
        raise SystemExit(
            f"[INVESTIGATE] max fee €{max_fee:,.0f} < €30M — likely a parser/block failure, "
            "not a real market. Inspect the HTML cache before trusting this run."
        )
    print("\nValidations passed.")


if __name__ == "__main__":
    main()
