"""Save/load fitted Stage-1 pipelines via joblib → models/stage1/{POS}/{target}_{Model}.pkl."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.utils.io import project_root


def model_path(position: str, target: str, model_name: str) -> Path:
    return project_root() / "models" / "stage1" / position / f"{target}_{model_name}.pkl"


def save(estimator, position: str, target: str, model_name: str) -> Path:
    path = model_path(position, target, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)
    return path


def load(position: str, target: str, model_name: str):
    return joblib.load(model_path(position, target, model_name))
