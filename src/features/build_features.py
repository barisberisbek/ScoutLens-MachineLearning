"""M10 — feature-engineering orchestrator.

No logic of its own: chains the per-module transforms in a fixed order. per-90s and
composites come before the lag module (so they exist to be lagged); the FIFA module comes
after lag (it consumes `fifa_rating_lag1`).
"""

from __future__ import annotations

import pandas as pd

from src.features.age_curve import add_age_curve_features
from src.features.categorical import add_categorical_features
from src.features.composites import add_composite_features
from src.features.contract import add_contract_features
from src.features.fifa import add_fifa_features
from src.features.lag import add_lag_features
from src.features.multipliers import add_multiplier_features
from src.features.per_90 import add_per_90_features
from src.features.z_scores import add_z_score_features
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Run all feature modules on the unified panel; returns panel + engineered columns."""
    n_in = panel.shape[1]
    df = panel.copy()
    df = add_per_90_features(df)        # M1
    df = add_composite_features(df)     # M2
    df = add_z_score_features(df)       # M3
    df = add_age_curve_features(df)     # M4
    df = add_contract_features(df)      # M5
    df = add_lag_features(df)           # M6 — after per-90/composites; resorts by (player_id, season)
    df = add_multiplier_features(df)    # M7
    df = add_categorical_features(df)   # M8
    df = add_fifa_features(df)          # M9 — after lag (uses fifa_rating_lag1)
    logger.info("build_features: %d → %d columns (+%d)", n_in, df.shape[1], df.shape[1] - n_in)
    return df
