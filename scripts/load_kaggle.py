"""CLI: load & clean the Kaggle 2024-25 dataset, save to interim parquet, print a summary.

Run from the project root:  python scripts/load_kaggle.py
"""

from __future__ import annotations

from src.data.kaggle_loader import load_kaggle_2024_25
from src.utils.io import project_root, save_parquet


def main() -> None:
    df = load_kaggle_2024_25()
    out_path = project_root() / "data" / "interim" / "kaggle_2024_25_clean.parquet"
    save_parquet(df, out_path)

    print(f"\nSaved: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"\nLeague distribution:\n{df['league'].value_counts()}")
    print(f"\nPosition distribution:\n{df['position_primary'].value_counts()}")

    n_split_players = (
        df.groupby(["player_name", "birth_year", "nationality_iso3"], dropna=False)[
            "is_split_season"
        ]
        .any()
        .sum()
    )
    print(f"\nSplit-season players: {int(n_split_players)}")


if __name__ == "__main__":
    main()
