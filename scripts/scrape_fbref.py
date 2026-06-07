"""CLI: scrape FBref player-season stats via soccerdata into data/raw/fbref/.

Examples:
    python scripts/scrape_fbref.py --validate        # small slice (validate first)
    python scripts/scrape_fbref.py                   # full plan (resumable)
    python scripts/scrape_fbref.py --league "Eredivisie" --season 2324
    python scripts/scrape_fbref.py --force           # re-fetch everything
"""

from __future__ import annotations

import argparse

from src.data.fbref_scraper import scrape_all_fbref

# Small validation slice: 1 top-5 + 1 lower league, 1 season, native + extended types.
_VALIDATE_LEAGUES = ["Premier League", "Eredivisie"]
_VALIDATE_SEASONS = ["2324"]
_VALIDATE_STAT_TYPES = ["standard", "passing", "defense", "gca"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape FBref player-season stats.")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached.")
    parser.add_argument("--league", type=str, default=None, help="Only this league.")
    parser.add_argument("--season", type=str, default=None, help="Only this season.")
    parser.add_argument("--stat-type", type=str, default=None, help="Only this stat type.")
    parser.add_argument(
        "--validate", action="store_true", help="Run the small validation slice only."
    )
    args = parser.parse_args()

    if args.validate:
        status = scrape_all_fbref(
            force=args.force,
            leagues=_VALIDATE_LEAGUES,
            seasons=_VALIDATE_SEASONS,
            stat_types=_VALIDATE_STAT_TYPES,
        )
    else:
        status = scrape_all_fbref(
            force=args.force,
            leagues=[args.league] if args.league else None,
            seasons=[args.season] if args.season else None,
            stat_types=[args.stat_type] if args.stat_type else None,
        )

    print(f"\n{'=' * 60}")
    print(f"Completed: {len(status['completed'])}")
    print(f"Skipped (cached): {len(status['skipped'])}")
    print(f"Failed: {len(status['failed'])}")
    if status["failed"]:
        print("\nFAILURES:")
        for item in status["failed"]:
            print(f"  {item}")


if __name__ == "__main__":
    main()
