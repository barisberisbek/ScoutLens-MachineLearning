"""Shared plot styling and figure-saving for the EDA notebook.

A single source of visual truth so every figure is consistent (colorblind-safe,
FT/Economist-clean) and lands in ``reports/figures/`` programmatically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.io import project_root

# Canonical orderings reused across figures.
POSITION_ORDER: list[str] = ["GK", "DEF", "MID", "FWD"]
TOP5_LEAGUES: list[str] = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
LOWER4_LEAGUES: list[str] = ["Eredivisie", "Liga Portugal", "Belgian Pro League", "Süper Lig"]
LEAGUE_ORDER: list[str] = TOP5_LEAGUES + LOWER4_LEAGUES
SEASON_ORDER: list[str] = ["2021-22", "2022-23", "2023-24", "2024-25"]

# Tier colors (colorblind-safe): top-5 vs lower-4.
TIER_COLORS: dict[int, str] = {1: "#0173B2", 2: "#DE8F05"}


def fig_dir() -> Path:
    """Return ``reports/figures/`` (created if missing)."""
    d = project_root() / "reports" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_eda_style() -> None:
    """Apply the project-wide EDA matplotlib/seaborn theme. Idempotent."""
    sns.set_theme(style="whitegrid", palette="colorblind", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })


def save_fig(fig, name: str) -> Path:
    """Save ``fig`` to ``reports/figures/<name>.png`` and return the path."""
    path = fig_dir() / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path)
    return path


def euro_millions(value: float) -> str:
    """Format euros compactly: 15_000_000 → '€15.0M', 800_000 → '€0.8M'."""
    return f"€{value / 1e6:.1f}M"
