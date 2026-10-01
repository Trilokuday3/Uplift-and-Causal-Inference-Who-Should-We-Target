from __future__ import annotations

import os
from pathlib import Path

import joblib


def load_model(artifacts_dir: str | Path):
    """Load the best model. If MLFLOW_MODEL_URI is set (e.g. runs:/<id>/best_model.joblib, with
    MLFLOW_TRACKING_URI pointing at the server), fetch it from MLflow; otherwise read the joblib file."""
    uri = os.environ.get("MLFLOW_MODEL_URI")
    if uri:
        import mlflow  # lazy: only needed when loading from the tracking server

        return joblib.load(mlflow.artifacts.download_artifacts(uri))
    return joblib.load(Path(artifacts_dir) / "best_model.joblib")
