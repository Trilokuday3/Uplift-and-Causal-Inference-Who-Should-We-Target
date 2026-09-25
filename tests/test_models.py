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


def test_class_transformation_ranks_by_effect_not_baseline_under_85_15_split():
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 20000
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(12)})
    t = rng.binomial(1, 0.85, size=n)
    baseline = 0.05 + 0.4 * (X["f1"] > 0)  # response depends on f1 only
    effect = 0.15 * (X["f0"] > 0)  # uplift depends on f0 only
    y = rng.binomial(1, np.clip(baseline + t * effect, 0, 1))

    scores = build_model("class_transformation", seed=0).fit(X, t, y).predict_uplift(X)
    corr_effect = np.corrcoef(scores, (X["f0"] > 0).astype(float))[0, 1]
    corr_baseline = np.corrcoef(scores, (X["f1"] > 0).astype(float))[0, 1]
    assert corr_effect > abs(corr_baseline)
