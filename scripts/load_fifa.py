"""CLI: build the unified FIFA ratings parquet + coverage report.

    python scripts/load_fifa.py
"""

from __future__ import annotations

import pandas as pd

from src.data.fifa_loader import load_all_fifa
from src.utils.io import project_root, save_parquet


def main() -> None:
    df = load_all_fifa()
    out = project_root() / "data" / "interim" / "fifa_ratings.parquet"
    save_parquet(df, out)

    print(f"\n{'=' * 64}\nSaved: {out}  (shape={df.shape})")

    print("\nPlayers per season:")
    print(df.groupby("season").size().to_string())

    print("\noverall / potential (median, max) per season:")
    for season, g in df.groupby("season"):
        o, p = g["overall"], g["potential"]
        pot = f"{p.median():.0f}/{p.max():.0f}" if p.notna().any() else "NULL"
        print(f"  {season}: overall {o.median():.0f}/{o.max():.0f} | potential {pot}")

    print("\nposition distribution:")
    print(df["primary_position"].value_counts(dropna=False).to_string())

    top5 = ["Premier League", "La Liga", "LaLiga", "Bundesliga", "Serie A", "Ligue 1"]
    in_top5 = int(
        df["league"].astype("string").str.contains("|".join(top5), case=False, na=False).sum()
    )
    print(f"\nrows whose league looks top-5: {in_top5} / {len(df)}")

    n_other = int((df["primary_position"] == "Other").sum())
    print(f"primary_position == 'Other': {n_other} ({100 * n_other / len(df):.2f}%)")

    print("\nMbappé spot-check (overall/potential per season):")
    mb = df[df["player_name"].astype(str).str.contains("Mbapp", case=False, na=False)]
    for _, r in mb.sort_values("fifa_year").iterrows():
        print(f"  {r['season']}: {r['player_name']} OVR={r['overall']} POT={r['potential']} ({r['source']})")

    # ── Validation ──
    counts = df.groupby("season").size()
    assert (counts >= 15000).all(), f"A season has <15K players: {counts.to_dict()}"
    ov = df["overall"].dropna()
    assert ov.between(40, 99).all(), "overall outside [40,99]"
    m = df[df["potential"].notna()]
    assert (m["potential"] >= m["overall"]).all(), "potential < overall somewhere"
    pos_ok = df["primary_position"].isin(["GK", "DEF", "MID", "FWD"]).mean()
    assert pos_ok > 0.99, f"primary_position coverage only {pos_ok:.3f}"
    assert df["player_name"].notna().all(), "null player_name present"
    print("\nValidations passed.")


if __name__ == "__main__":
    main()
