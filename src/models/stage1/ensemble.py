"""Stacked ensemble: a Ridge meta-learner over the tuned base pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge


class StackedModel:
    """StackingRegressor over the already-tuned base estimators (no hyperparameter re-grid)."""

    name = "Stacked"

    def fit(self, X: pd.DataFrame, y: pd.Series, base_estimators: dict) -> "StackedModel":
        estimators = [(name, est) for name, est in base_estimators.items()]
        # StackingRegressor clones + refits each base on CV folds, preserving tuned params.
        self.model = StackingRegressor(
            estimators=estimators, final_estimator=Ridge(), cv=3, n_jobs=1,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
