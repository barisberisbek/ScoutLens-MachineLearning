"""CLI: scrape Understat player-season stats (xG) for the top-5 leagues.

    python scripts/scrape_understat.py            # all top-5 x all seasons (resumable)
    python scripts/scrape_understat.py --force    # re-fetch everything
"""

from __future__ import annotations

import argparse

from src.data.understat_loader import scrape_all_understat


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Understat player-season stats.")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached.")
    parser.add_argument("--league", type=str, default=None, help="Only this league.")
    parser.add_argument("--season", type=str, default=None, help="Only this season.")
    args = parser.parse_args()

    status = scrape_all_understat(
        force=args.force,
        leagues=[args.league] if args.league else None,
        seasons=[args.season] if args.season else None,
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
