"""M5 — contract features.

Contract remaining time is one of the highest-ROI valuation signals (an expiring contract
collapses a fee). `contract_remaining_months` / `has_contract_date` come from Phase 2; this
adds a years view and expiry/long-term flags. ~36% of rows have no contract date → NaN
(flags resolve to False, never imputed).
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def add_contract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add contract_years_until_expiry + is_expiring (≤6mo) + is_long_contract (>36mo)."""
    df = df.copy()
    months = df["contract_remaining_months"]

    df["contract_years_until_expiry"] = months / 12.0
    df["is_expiring"] = (months <= 6).fillna(False).astype(bool)
    df["is_long_contract"] = (months > 36).fillna(False).astype(bool)

    logger.info("contract: added 3 columns")
    return df
