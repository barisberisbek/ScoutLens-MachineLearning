"""Save/load fitted Stage-2 pipelines → models/stage2/{POS}/{Model}.pkl."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.utils.io import project_root


def model_path(position: str, model_name: str) -> Path:
    return project_root() / "models" / "stage2" / position / f"{model_name}.pkl"


def save(estimator, position: str, model_name: str) -> Path:
    path = model_path(position, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)
    return path


def load(position: str, model_name: str):
    return joblib.load(model_path(position, model_name))
