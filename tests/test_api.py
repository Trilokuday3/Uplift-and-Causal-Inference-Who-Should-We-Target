import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app

FEATURES = {f"f{i}": 0.0 for i in range(12)}


class FakeModel:
    def predict_uplift(self, X):
        return X["f0"].to_numpy() * 0.1


POLICY = {
    "assumptions": {"cost_per_impression": 0.5, "value_per_outcome": 20.0, "currency": "INR"},
    "population": 1000,
    "strategies": {
        name: {
            "curve": [
                {"fraction": 0.5, "incremental_outcomes": inc, "profit": profit, "uplift": 0.04, "n_targeted": 500,
                 "cost": 250.0, "revenue": 500.0},
                {"fraction": 1.0, "incremental_outcomes": 10.0, "profit": -100.0, "uplift": 0.01, "n_targeted": 1000,
                 "cost": 500.0, "revenue": 200.0},
            ]
        }
        for name, inc, profit in (("uplift", 20.0, 250.0), ("response_model", 15.0, 50.0), ("random", 5.0, -150.0))
    },
}


@pytest.fixture
def client():
    return TestClient(create_app(model=FakeModel(), threshold=0.05, policy=POLICY, model_name="fake"))


def test_health_reports_model_and_policy_loaded(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model": "fake", "policy_loaded": True}


def test_score_returns_uplift_and_treat_decision(client):
    high = client.post("/score", json={"features": {**FEATURES, "f0": 2.0}}).json()
    low = client.post("/score", json={"features": {**FEATURES, "f0": -2.0}}).json()
    assert high["uplift"] == pytest.approx(0.2) and high["treat"] is True
    assert low["treat"] is False and low["threshold"] == 0.05


def test_score_rejects_missing_or_leaky_fields(client):
    missing = {k: v for k, v in FEATURES.items() if k != "f3"}
    assert client.post("/score", json={"features": missing}).status_code == 422
    assert client.post("/score", json={"features": {**FEATURES, "exposure": 1}}).status_code == 422
    assert client.post("/score", json={"features": {**FEATURES, "visit": 1}}).status_code == 422


def test_policy_returns_expected_lift_for_budget_with_baselines(client):
    body = client.get("/policy", params={"budget": 0.5}).json()
    assert body["budget"] == 0.5
    assert body["uplift"]["incremental_outcomes"] == pytest.approx(20.0)
    assert body["response_model"]["incremental_outcomes"] == pytest.approx(15.0)
    assert body["random"]["incremental_outcomes"] == pytest.approx(5.0)
    assert body["assumptions"]["currency"] == "INR"


@pytest.mark.parametrize("budget", [0, -0.1, 1.5])
def test_policy_rejects_out_of_range_budget(client, budget):
    assert client.get("/policy", params={"budget": budget}).status_code == 422


def test_endpoints_return_503_when_nothing_is_loaded():
    empty = TestClient(create_app(model=None, threshold=None, policy=None))
    assert empty.get("/health").json()["status"] == "degraded"
    assert empty.post("/score", json={"features": FEATURES}).status_code == 503
    assert empty.get("/policy", params={"budget": 0.3}).status_code == 503


def test_score_rejects_extra_top_level_fields_and_non_numeric_values(client):
    good = {"features": FEATURES}
    assert client.post("/score", json={**good, "exposure": 1}).status_code == 422
    assert client.post("/score", json={"features": {**FEATURES, "f0": True}}).status_code == 422
    assert client.post("/score", json={"features": {**FEATURES, "f0": "2"}}).status_code == 422
    assert client.post("/score", json={"features": {**FEATURES, "f0": 2}}).status_code == 200  # ints are fine
    raw = '{"features": {' + ", ".join(f'"f{i}": {"NaN" if i == 0 else 0.0}' for i in range(12)) + "}}"
    assert client.post("/score", content=raw, headers={"Content-Type": "application/json"}).status_code == 422


def test_create_app_from_env_degrades_instead_of_crashing_when_scores_are_missing(tmp_path, monkeypatch):
    import json

    import joblib

    from api.main import create_app_from_env

    (tmp_path / "model_meta.json").write_text(json.dumps({"model_name": "fake"}))
    joblib.dump({"not": "a real model"}, tmp_path / "best_model.joblib")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("POLICY_PATH", str(tmp_path / "missing.json"))
    monkeypatch.delenv("MODEL_URI", raising=False)
    assert TestClient(create_app_from_env()).get("/health").json()["status"] == "degraded"
