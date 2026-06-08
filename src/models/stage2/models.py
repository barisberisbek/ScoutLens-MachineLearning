"""Stage-2 model wrappers — reuse Stage-1 Ridge/XGBoost; tightened MLP (Phase-5 MLP overfit).

Ridge and XGBoost(cpu) are identical to Stage 1. MLP uses stronger L2 (alpha 0.1–10) and
smaller nets, since Stage-1 MLP was worse than baseline on every target.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.stage1.models import RANDOM_STATE, RidgeModel, XGBoostModel, _BaseStage1Model


class MLPModel(_BaseStage1Model):
    name = "MLP"

    def _pipeline(self):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("model", MLPRegressor(max_iter=500, early_stopping=True,
                                                random_state=RANDOM_STATE))])

    def _param_grid(self):
        # Tightened vs Stage 1 (alpha 0.001/0.01 → 0.1/1/10; smaller nets) to curb overfit.
        return {"model__hidden_layer_sizes": [(64,), (32,), (64, 32)],
                "model__alpha": [0.1, 1.0, 10.0]}


BASE_MODELS: dict[str, type] = {
    "Ridge": RidgeModel, "XGBoost": XGBoostModel, "MLP": MLPModel,
}
