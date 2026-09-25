import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

import uplift.models as models_module
from uplift.constants import FEATURE_COLS
from uplift.models import SEARCH_SPACES
from uplift.train import (
    fit_all,
    log_run,
    score_all,
    stratified_subsample,
    summarize,
    tune_model,
)


def _cell_shares(df):
    return df.groupby(["treatment", "visit"]).size() / len(df)


def test_subsample_preserves_strata_and_is_deterministic(uplift_frame):
    df = uplift_frame(5000, seed=4)
    a = stratified_subsample(df, 1000, seed=1)
    b = stratified_subsample(df, 1000, seed=1)
    assert abs(len(a) - 1000) <= 4
    assert a.equals(b)
    diff = (_cell_shares(a) - _cell_shares(df)).abs().max()
    assert diff < 0.02


def test_subsample_larger_than_frame_returns_frame(uplift_frame):
    df = uplift_frame(300)
    assert len(stratified_subsample(df, 10_000, seed=0)) == 300


def test_fit_all_returns_requested_models(uplift_frame):
    df = uplift_frame(1200)
    fitted = fit_all(df, "visit", seed=0, names=["random", "response_model", "s_learner"])
    assert sorted(fitted) == ["random", "response_model", "s_learner"]


def test_fit_all_only_passes_feature_columns(uplift_frame, monkeypatch):
    seen = {}

    class Spy:
        def __init__(self, seed=0, **params):
            pass

        def fit(self, X, treatment, y):
            seen["cols"] = list(X.columns)
            return self

        def predict_uplift(self, X):
            return np.zeros(len(X))

    monkeypatch.setitem(models_module.MODEL_REGISTRY, "spy", Spy)
    fit_all(uplift_frame(200), "visit", seed=0, names=["spy"])
    assert seen["cols"] == FEATURE_COLS


def test_score_all_and_summarize(uplift_frame):
    train, test = uplift_frame(2000, seed=1), uplift_frame(1500, seed=2)
    fitted = fit_all(train, "visit", seed=0, names=["random", "s_learner"])
    scores = score_all(fitted, test)
    metrics = summarize(test["visit"].to_numpy(), test["treatment"].to_numpy(), scores["s_learner"])
    assert set(metrics) == {"qini_auc", "uplift_at_10", "uplift_at_20", "uplift_at_30"}
    random_metrics = summarize(test["visit"].to_numpy(), test["treatment"].to_numpy(), scores["random"])
    assert metrics["qini_auc"] > random_metrics["qini_auc"]


def test_log_run_writes_metrics_to_local_store(tmp_path):
    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    log_run("s_learner", {"n_estimators": 10}, {"qini_auc": 0.12}, uri)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("uplift")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.metrics["qini_auc"] == pytest.approx(0.12)
    assert runs[0].data.params["n_estimators"] == "10"


def test_tune_model_returns_params_for_its_search_space(uplift_frame):
    train, val = uplift_frame(1200, seed=1), uplift_frame(800, seed=2)
    best = tune_model("s_learner", train, val, "visit", n_trials=3, seed=0)
    assert set(best) == set(SEARCH_SPACES["s_learner"])


def test_tune_model_rejects_untunable_model(uplift_frame):
    df = uplift_frame(200)
    with pytest.raises(ValueError):
        tune_model("random", df, df, "visit", n_trials=1, seed=0)


def test_tune_model_signature_has_no_test_frame():
    import inspect

    params = inspect.signature(tune_model).parameters
    assert not any("test" in p for p in params)


def test_subsample_is_shuffled_not_grouped_by_stratum(uplift_frame):
    sampled = stratified_subsample(uplift_frame(5000, seed=6), 2000, seed=1)
    head = sampled.head(200)
    assert head["treatment"].nunique() == 2 and head["visit"].nunique() == 2
