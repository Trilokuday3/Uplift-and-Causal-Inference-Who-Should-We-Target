from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow import MlflowClient

from uplift.constants import FEATURE_COLS


class UpliftPyfunc(mlflow.pyfunc.PythonModel):
    """Wraps a joblib-saved UpliftModel so MLflow can register, version and serve it."""

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input, params=None):
        return np.asarray(self.model.predict_uplift(model_input[FEATURE_COLS]), dtype=float)


class _RegisteredModel:
    """Gives a loaded pyfunc model the same predict_uplift interface as the in-process models."""

    def __init__(self, pyfunc_model):
        self._model = pyfunc_model

    def predict_uplift(self, X):
        return np.asarray(self._model.predict(X[FEATURE_COLS])).ravel()


def register_model(model_path: str | Path, name: str, tracking_uri: str, alias: str = "production") -> str:
    """Log a saved model to MLflow, register it under `name`, and point `alias` at the new version."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("uplift")
    with mlflow.start_run(run_name=f"register-{name}"):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=UpliftPyfunc(),
            artifacts={"model": str(model_path)},
            registered_model_name=name,
            pip_requirements=["joblib", "numpy", "pandas", "lightgbm", "econml", "scikit-uplift"],
        )
    client = MlflowClient(tracking_uri=tracking_uri)
    version = str(info.registered_model_version)
    client.set_registered_model_alias(name, alias, version)
    return version


def load_registered_model(model_uri: str, tracking_uri: str) -> _RegisteredModel:
    mlflow.set_tracking_uri(tracking_uri)
    return _RegisteredModel(mlflow.pyfunc.load_model(model_uri))
