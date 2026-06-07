"""I/O helpers: project paths, lookup-CSV loading, and parquet read/write.

Infrastructure only — no domain logic. Used across phases to keep file handling
consistent (parent dirs created on save, clear errors on missing inputs).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the project root: the nearest ancestor directory containing CLAUDE.md.

    Walks up from this file's location so it works regardless of the current
    working directory (scripts, notebooks, tests).
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "CLAUDE.md").exists():
            return parent
    raise FileNotFoundError(
        "Could not locate project root (no CLAUDE.md found in any parent of "
        f"{Path(__file__).resolve()})"
    )


def load_lookup_csv(filename: str) -> dict:
    """Load a two-column lookup CSV from data/external/ as a {key: value} dict.

    The first column is the key, the second is the value. The ``.csv`` suffix is
    optional, e.g. both ``"continent_map"`` and ``"continent_map.csv"`` work.
    """
    name = filename if filename.endswith(".csv") else f"{filename}.csv"
    path = project_root() / "data" / "external" / name
    if not path.exists():
        raise FileNotFoundError(f"Lookup CSV not found: {path}")
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"{path} has {df.shape[1]} column(s); need at least 2 (key, value)")
    key_col, value_col = df.columns[0], df.columns[1]
    return dict(zip(df[key_col], df[value_col]))


def save_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Write ``df`` to a parquet file, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False, **kwargs)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file into a DataFrame, with a clear error if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path, engine="pyarrow")
