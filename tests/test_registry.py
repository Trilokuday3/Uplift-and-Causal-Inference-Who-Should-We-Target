import joblib
import numpy as np
import pytest

from uplift.models import build_model, feature_matrix
from uplift.registry import load_registered_model, register_model


@pytest.fixture
def saved_model(tmp_path, uplift_frame):
    df = uplift_frame(1500, seed=31)
    model = build_model("response_model", seed=0).fit(feature_matrix(df), df["treatment"].to_numpy(), df["visit"].to_numpy())
    path = tmp_path / "best_model.joblib"
    joblib.dump(model, path)
    return path, model, df


def test_register_then_load_by_alias_reproduces_predictions(saved_model, tmp_path):
    path, model, df = saved_model
    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    version = register_model(path, "uplift", uri, alias="production")
    assert int(version) >= 1
    loaded = load_registered_model("models:/uplift@production", uri)
    expected = model.predict_uplift(feature_matrix(df))
    assert np.allclose(loaded.predict_uplift(feature_matrix(df)), expected)


def test_loaded_model_ignores_non_feature_columns(saved_model, tmp_path):
    path, model, df = saved_model
    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    register_model(path, "uplift", uri)
    loaded = load_registered_model("models:/uplift@production", uri)
    with_extra = df.copy()
    assert np.allclose(loaded.predict_uplift(with_extra), model.predict_uplift(feature_matrix(df)))


def test_scorer_and_api_load_the_model_from_the_registry(saved_model, tmp_path, monkeypatch, uplift_frame):
    import json

    import pandas as pd

    from api.main import create_app_from_env
    from streaming.scorer import load_model

    path, model, df = saved_model
    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    register_model(path, "uplift", uri)
    expected = model.predict_uplift(feature_matrix(df))

    loaded = load_model(str(tmp_path), model_uri="models:/uplift@production", tracking_uri=uri)
    assert np.allclose(loaded.predict_uplift(feature_matrix(df)), expected)

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "model_meta.json").write_text(json.dumps({"model_name": "response_model", "outcome": "visit"}))
    pd.DataFrame({"response_model": expected}).to_parquet(artifacts / "test_scores.parquet")
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("POLICY_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setenv("MODEL_URI", "models:/uplift@production")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    from fastapi.testclient import TestClient

    client = TestClient(create_app_from_env())
    assert client.get("/health").json()["status"] == "ok"
    row = {f"f{i}": float(df.iloc[0][f"f{i}"]) for i in range(12)}
    assert client.post("/score", json={"features": row}).json()["uplift"] == pytest.approx(float(expected[0]))
