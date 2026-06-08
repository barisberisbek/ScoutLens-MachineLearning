"""M2 — composite domain features.

Domain-knowledge-injected scores built from per-90 rates. Components NaN-propagate (a
composite needing an unavailable input becomes NaN), EXCEPT `attacking_output`, whose xAG
term is an optional bonus (coalesced xag→understat_xa→0 so the goals+assists core survives
in leagues without xG). `composite_completeness` records the fraction of composites computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

_COMPOSITES = ["goal_threat", "creative_threat", "defensive_actions",
               "shooting_efficiency", "attacking_output"]


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 5 composite scores + a `composite_completeness` fraction."""
    df = df.copy()

    df["goal_threat"] = df["xg_per_90"] + 0.5 * df["shots_per_90"]
    df["creative_threat"] = df["xag_per_90"].where(df["xag_per_90"].notna(), df["understat_xa_per_90"])
    df["defensive_actions"] = df["tackles_won_per_90"] + df["interceptions_per_90"]

    # Threshold: only meaningful with real shot volume; avoids spurious-high values for GK/DEF.
    shots = df["shots_per_90"].to_numpy(dtype="float64")
    goals = df["goals_per_90"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["shooting_efficiency"] = np.where(shots >= 0.5, goals / shots, np.nan)

    # attacking_output: goals + assists + optional 0.5 * (xag or understat_xa, else 0 bonus).
    xag_bonus = df["xag_per_90"].where(df["xag_per_90"].notna(), df["understat_xa_per_90"]).fillna(0.0)
    df["attacking_output"] = df["goals_per_90"] + df["assists_per_90"] + 0.5 * xag_bonus

    df["composite_completeness"] = df[_COMPOSITES].notna().mean(axis=1)
    logger.info("composites: added %d columns", len(_COMPOSITES) + 1)
    return df
