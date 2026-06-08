"""Stage-1 base model wrappers (CPU): Ridge, XGBoost, MLP.

Each wraps a sklearn Pipeline (median impute → optional scale → estimator) and tunes it with
an inner GridSearchCV (MAE scoring). `fit` exposes the tuned `best_estimator_` (the fitted
Pipeline) for persistence and stacking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42


class _BaseStage1Model:
    name = "Base"
    _grid_n_jobs = -1  # parallelize the grid; estimators stay single-threaded

    def _pipeline(self) -> Pipeline:
        raise NotImplementedError

    def _param_grid(self) -> dict:
        raise NotImplementedError

    def _cv(self) -> int:
        return 3

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_BaseStage1Model":
        gs = GridSearchCV(
            self._pipeline(), self._param_grid(), cv=self._cv(),
            scoring="neg_mean_absolute_error", n_jobs=self._grid_n_jobs,
        )
        gs.fit(X, y)
        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.cv_mae_ = float(-gs.best_score_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.best_estimator_.predict(X)


class RidgeModel(_BaseStage1Model):
    name = "Ridge"

    def _pipeline(self):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()), ("model", Ridge(random_state=RANDOM_STATE))])

    def _param_grid(self):
        return {"model__alpha": [0.01, 0.1, 1.0, 10.0]}

    def _cv(self):
        return 5


class XGBoostModel(_BaseStage1Model):
    name = "XGBoost"
    _grid_n_jobs = 1  # XGBoost itself uses all cores (n_jobs=-1); avoid nested oversubscription

    def _pipeline(self):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("model", XGBRegressor(device="cpu", tree_method="hist",
                                                n_jobs=-1, random_state=RANDOM_STATE))])

    def _param_grid(self):
        return {"model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.05, 0.1],
                "model__n_estimators": [200, 500]}


class MLPModel(_BaseStage1Model):
    name = "MLP"

    def _pipeline(self):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("model", MLPRegressor(max_iter=500, early_stopping=True,
                                                random_state=RANDOM_STATE))])

    def _param_grid(self):
        return {"model__hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
                "model__alpha": [0.001, 0.01]}


BASE_MODELS: dict[str, type[_BaseStage1Model]] = {
    "Ridge": RidgeModel, "XGBoost": XGBoostModel, "MLP": MLPModel,
}
