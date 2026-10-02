from __future__ import annotations

import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from typing import Annotated
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from uplift.constants import FEATURE_COLS
from uplift.policy import interpolate_curve


def _drop_non_finite(value):
    """Replace NaN/Infinity with None so Starlette's strict JSON encoder doesn't crash on them."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _drop_non_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_drop_non_finite(v) for v in value]
    return value


Number = Annotated[float, Field(allow_inf_nan=False)]


class Features(BaseModel):
    """Exactly the model's features, as real numbers. Extra fields (e.g. `exposure`, `visit`),
    booleans, strings and NaN/inf are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True)
    f0: Number
    f1: Number
    f2: Number
    f3: Number
    f4: Number
    f5: Number
    f6: Number
    f7: Number
    f8: Number
    f9: Number
    f10: Number
    f11: Number


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: Features


def create_app(model=None, threshold: float | None = None, policy: dict | None = None, model_name: str = "unknown") -> FastAPI:
    app = FastAPI(title="Uplift targeting API", version="0.1.0")

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        # A rejected NaN/Infinity in the request body ends up echoed into exc.errors() as
        # a raw float; Starlette's JSONResponse rejects non-finite floats outright, which
        # would otherwise turn a 422 into an unhandled 500.
        content = _drop_non_finite(jsonable_encoder({"detail": exc.errors()}))
        return JSONResponse(status_code=422, content=content)

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
    ready = (artifacts / "model_meta.json").exists() and (artifacts / "test_scores.parquet").exists()
    if has_model and ready:
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
