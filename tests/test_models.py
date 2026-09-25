import numpy as np
import pytest

from uplift.constants import FEATURE_COLS
from uplift.models import MODEL_REGISTRY, build_model, feature_matrix

ALL_MODELS = [
    "random",
    "response_model",
    "s_learner",
    "t_learner",
    "x_learner",
    "class_transformation",
    "causal_forest",
]


def test_registry_has_all_seven_models():
    assert sorted(MODEL_REGISTRY) == sorted(ALL_MODELS)


def test_feature_matrix_excludes_leaky_columns(uplift_frame):
    X = feature_matrix(uplift_frame(50))
    assert list(X.columns) == FEATURE_COLS
    for leaky in ("exposure", "treatment", "visit", "conversion"):
        assert leaky not in X.columns


def test_random_model_shape_finite_and_seeded(uplift_frame):
    df = uplift_frame(200)
    X, t, y = feature_matrix(df), df["treatment"].to_numpy(), df["visit"].to_numpy()
    a = build_model("random", seed=1).fit(X, t, y).predict_uplift(X)
    b = build_model("random", seed=1).fit(X, t, y).predict_uplift(X)
    c = build_model("random", seed=2).fit(X, t, y).predict_uplift(X)
    assert a.shape == (200,) and np.isfinite(a).all()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_response_model_ignores_treatment(uplift_frame):
    df = uplift_frame(600)
    X, y = feature_matrix(df), df["visit"].to_numpy()
    t = df["treatment"].to_numpy()
    scores = build_model("response_model", seed=0).fit(X, t, y).predict_uplift(X)
    assert scores.shape == (600,)
    assert ((scores >= 0) & (scores <= 1)).all()
    flipped = build_model("response_model", seed=0).fit(X, 1 - t, y).predict_uplift(X)
    assert np.allclose(scores, flipped)


@pytest.mark.parametrize(
    "name",
    ["s_learner", "t_learner", "x_learner", "class_transformation", "causal_forest"],
)
def test_uplift_model_is_directionally_correct(name, uplift_frame):
    train, test = uplift_frame(2000, seed=1), uplift_frame(1000, seed=2)
    model = build_model(name, seed=0)
    model.fit(feature_matrix(train), train["treatment"].to_numpy(), train["visit"].to_numpy())
    scores = np.asarray(model.predict_uplift(feature_matrix(test))).ravel()
    assert scores.shape == (len(test),)
    assert np.isfinite(scores).all()
    high = scores[(test["f0"] > 0.5).to_numpy()].mean()
    low = scores[(test["f0"] < -0.5).to_numpy()].mean()
    assert high > low
