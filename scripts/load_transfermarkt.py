"""CLI: ingest Transfermarkt CSVs into data/interim/tm_*.parquet + coverage report.

    python scripts/load_transfermarkt.py
"""

from __future__ import annotations

from src.data.transfermarkt_loader import (
    load_competitions,
    load_national_teams,
    load_player_seasons_mv,
    load_players,
    load_transfers,
)
from src.utils.constants import SEASON_END_DATES, TOP5_LEAGUES
from src.utils.io import project_root, save_parquet


def main() -> None:
    interim = project_root() / "data" / "interim"

    competitions = load_competitions()
    players = load_players()
    seasons_mv = load_player_seasons_mv()
    transfers = load_transfers()
    national_teams = load_national_teams()

    outputs = {
        "tm_competitions": competitions,
        "tm_players": players,
        "tm_player_seasons": seasons_mv,
        "tm_transfers": transfers,
        "tm_national_teams": national_teams,
    }
    for name, df in outputs.items():
        save_parquet(df, interim / f"{name}.parquet")

    # ── Coverage summary ──
    print(f"\n{'=' * 64}\nPARQUETS")
    for name, df in outputs.items():
        print(f"  {name}: {df.shape}")

    print("\nPLAYERS per league (current club):")
    print(players["current_league_name"].value_counts().to_string())

    print("\nMV snapshot coverage per season (matched players / 9-league players):")
    n_players = players["tm_player_id"].nunique()
    for season in SEASON_END_DATES:
        n = seasons_mv.loc[seasons_mv["season"] == season, "tm_player_id"].nunique()
        print(f"  {season}: {n}  ({100 * n / max(n_players, 1):.1f}% of {n_players})")

    # Orphan MV: snapshots whose player has no current metadata in our set (Phase 2 will drop)
    orphan = (~seasons_mv["tm_player_id"].isin(set(players["tm_player_id"]))).mean()
    print(f"\nMV snapshots WITHOUT matching player metadata (orphans): {100 * orphan:.1f}%")

    print("\nTop-5 MV snapshot counts per season (Stage-2 train/val set size):")
    top5_mv = seasons_mv[seasons_mv["domestic_competition_id"].isin({"GB1", "ES1", "L1", "IT1", "FR1"})]
    print(top5_mv.groupby("season")["tm_player_id"].nunique().to_string())

    print("\nLower-league (tier-2) MV snapshot counts per season:")
    lower_ids = seasons_mv["domestic_competition_id"].isin(
        {"NL1", "PO1", "BE1", "TR1"}
    )
    print(seasons_mv[lower_ids].groupby("season")["tm_player_id"].nunique().to_string())

    fee = transfers["transfer_fee_eur"]
    print(f"\nTOP-5 inbound transfers (fee>0, 2024-08..2025-08): {len(transfers)}")
    if len(transfers):
        print(f"  fee EUR — median {fee.median():,.0f} | mean {fee.mean():,.0f} | max {fee.max():,.0f}")
        print("  Top-5 by fee:")
        top = transfers.nlargest(5, "transfer_fee_eur")
        for _, r in top.iterrows():
            print(f"    {r.get('to_club_name','?')} <- {r.get('from_club_name','?')}: "
                  f"{r['transfer_fee_eur']:,.0f} EUR ({r.get('window','?')})")

    # ── Validation (hard-assert robust checks; report-only for the soft target) ──
    assert len(competitions) == 9, f"Expected 9 competitions, got {len(competitions)}"
    assert len(players) > 5000, f"Expected >5000 players, got {len(players)}"
    for season in SEASON_END_DATES:
        top5_n = seasons_mv[
            (seasons_mv["season"] == season)
            & seasons_mv["domestic_competition_id"].isin({"GB1", "ES1", "L1", "IT1", "FR1"})
        ].shape[0]
        assert top5_n > 0, f"No top-5 MV snapshots for {season}"
    assert seasons_mv["days_from_season_end"].abs().max() <= 45, "MV snapshot outside ±45d window"
    assert 0 < seasons_mv["market_value_eur"].max() < 300_000_000, "MV out of sane range"
    if len(transfers) <= 100:
        print(f"\n[WARN] only {len(transfers)} top-5 inbound transfers (spec target >200) — review.")
    print("\nValidations passed.")


if __name__ == "__main__":
    main()
