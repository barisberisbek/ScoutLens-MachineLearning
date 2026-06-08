"""Load the best trained Stage-1 / Stage-2 model per slot (cached).

Best model is taken from the saved metrics parquets (lowest validation MAE, simpler-model
tie-break) — the same selection the reports use. Fitted sklearn pipelines expose
`feature_names_in_`, so inference passes exactly the columns each model was trained on.
"""

from __future__ import annotations

from functools import lru_cache

from src.models.stage1 import persist as s1_persist
from src.models.stage1.evaluate import select_best as s1_select_best
from src.models.stage2 import persist as s2_persist
from src.models.stage2.evaluate import select_best as s2_select_best
from src.utils.io import load_parquet, project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def best_stage1() -> dict:
    """{(position, target): fitted_pipeline} for the best Stage-1 model of each target."""
    m = load_parquet(project_root() / "data" / "processed" / "stage1_metrics.parquet")
    best = s1_select_best(m)
    out = {(r.position, r.target): s1_persist.load(r.position, r.target, r.model)
           for r in best.itertuples(index=False)}
    logger.info("Loaded %d best Stage-1 models", len(out))
    return out


@lru_cache(maxsize=1)
def best_stage2() -> dict:
    """{position: fitted_pipeline} for the best Stage-2 model of each position."""
    m = load_parquet(project_root() / "data" / "processed" / "stage2_metrics.parquet")
    best = s2_select_best(m)
    out = {r.position: s2_persist.load(r.position, r.model) for r in best.itertuples(index=False)}
    logger.info("Loaded %d best Stage-2 models: %s", len(out),
                {r.position: r.model for r in best.itertuples(index=False)})
    return out


@lru_cache(maxsize=1)
def best_stage2_names() -> dict:
    """{position: model_name} — for reporting/tests."""
    m = load_parquet(project_root() / "data" / "processed" / "stage2_metrics.parquet")
    return {r.position: r.model for r in s2_select_best(m).itertuples(index=False)}
