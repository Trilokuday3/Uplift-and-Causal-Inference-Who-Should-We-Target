from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from uplift.constants import FEATURE_COLS
from uplift.policy import interpolate_curve


class Features(BaseModel):
    """Exactly the model's features. Extra fields (e.g. `exposure`, `visit`) are rejected."""

    model_config = ConfigDict(extra="forbid")
    f0: float
    f1: float
    f2: float
    f3: float
    f4: float
    f5: float
    f6: float
    f7: float
    f8: float
    f9: float
    f10: float
    f11: float


class ScoreRequest(BaseModel):
    features: Features


def create_app(model=None, threshold: float | None = None, policy: dict | None = None, model_name: str = "unknown") -> FastAPI:
    app = FastAPI(title="Uplift targeting API", version="0.1.0")

    @app.get("/health")
    def health():
        ready = model is not None and threshold is not None
        return {"status": "ok" if ready else "degraded", "model": model_name if model is not None else None,
                "policy_loaded": policy is not None}

    @app.post("/score")
    def score(request: ScoreRequest):
        if model is None or threshold is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        row = pd.DataFrame([request.features.model_dump()])[FEATURE_COLS]
        uplift = float(np.asarray(model.predict_uplift(row)).ravel()[0])
        return {"uplift": uplift, "treat": uplift >= threshold, "threshold": threshold, "model": model_name}

    @app.get("/policy")
    def policy_lookup(budget: float = Query(..., gt=0, le=1, description="fraction of users to target")):
        if policy is None:
            raise HTTPException(status_code=503, detail="policy results not loaded")
        body = {"budget": budget, "assumptions": policy["assumptions"], "population": policy.get("population")}
        for strategy, data in policy["strategies"].items():
            body[strategy] = interpolate_curve(data["curve"], budget)
        return body

    return app


def create_app_from_env() -> FastAPI:
    artifacts = Path(os.environ.get("ARTIFACTS_DIR", "models"))
    policy_path = Path(os.environ.get("POLICY_PATH", "docs/policy_results.json"))
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else None
    model = threshold = None
    name = "unknown"
    has_model = (artifacts / "best_model.joblib").exists() or os.environ.get("MODEL_URI")
    if has_model and (artifacts / "model_meta.json").exists():
        meta = json.loads((artifacts / "model_meta.json").read_text())
        name = meta["model_name"]
        model_uri = os.environ.get("MODEL_URI")
        if model_uri:
            from uplift.registry import load_registered_model

            model = load_registered_model(model_uri, os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        else:
            model = joblib.load(artifacts / "best_model.joblib")
        fraction = float(os.environ.get("TREAT_FRACTION", policy["operating_point"]["fraction"] if policy else 0.3))
        scores = pd.read_parquet(artifacts / "test_scores.parquet")[name]
        threshold = float(np.quantile(scores.to_numpy(), 1 - fraction))
    return create_app(model, threshold, policy, name)


app = create_app_from_env()
