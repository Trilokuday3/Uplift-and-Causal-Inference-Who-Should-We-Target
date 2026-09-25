from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.metalearners import SLearner, TLearner, XLearner
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklift.models import ClassTransformation

from uplift.constants import FEATURE_COLS

_LGBM_DEFAULTS = {
    "n_estimators": 60,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 20,
}
_FOREST_DEFAULTS = {"n_estimators": 100, "min_samples_leaf": 20}


class UpliftModel(Protocol):
    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "UpliftModel": ...

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray: ...


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLS]


def _lgbm_kwargs(seed: int, params: dict) -> dict:
    kwargs = {**_LGBM_DEFAULTS, "verbose": -1, "random_state": seed, "n_jobs": 4}
    kwargs.update({k: v for k, v in params.items() if k in _LGBM_DEFAULTS})
    return kwargs


def _regressor(seed: int, params: dict) -> LGBMRegressor:
    return LGBMRegressor(**_lgbm_kwargs(seed, params))


def _classifier(seed: int, params: dict) -> LGBMClassifier:
    return LGBMClassifier(**_lgbm_kwargs(seed, params))


class RandomModel:
    def __init__(self, seed: int = 42, **params):
        self.seed = seed

    def fit(self, X, treatment, y):
        return self

    def predict_uplift(self, X):
        return np.random.default_rng(self.seed).random(len(X))


class ResponseModel:
    """The naive 'who converts' classifier: ignores treatment entirely."""

    def __init__(self, seed: int = 42, **params):
        self.model = _classifier(seed, params)

    def fit(self, X, treatment, y):
        self.model.fit(X, y)
        return self

    def predict_uplift(self, X):
        return self.model.predict_proba(X)[:, 1]


class _EconmlModel:
    """Adapter for EconML estimators, which use fit(Y, T, X=X) and effect(X)."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, treatment, y):
        self.estimator.fit(y, treatment, X=X)
        return self

    def predict_uplift(self, X):
        return np.asarray(self.estimator.effect(X)).ravel()


def _s_learner(seed: int = 42, **params):
    return _EconmlModel(SLearner(overall_model=_regressor(seed, params)))


def _t_learner(seed: int = 42, **params):
    return _EconmlModel(TLearner(models=[_regressor(seed, params), _regressor(seed, params)]))


def _x_learner(seed: int = 42, **params):
    return _EconmlModel(
        XLearner(
            models=[_regressor(seed, params), _regressor(seed, params)],
            cate_models=[_regressor(seed, params), _regressor(seed, params)],
            propensity_model=LogisticRegression(max_iter=1000),
        )
    )


def _causal_forest(seed: int = 42, **params):
    forest = {**_FOREST_DEFAULTS, **{k: v for k, v in params.items() if k in _FOREST_DEFAULTS}}
    return _EconmlModel(
        CausalForestDML(
            model_y=_regressor(seed, {}),
            model_t=_classifier(seed, {}),
            discrete_treatment=True,
            n_estimators=int(forest["n_estimators"]),
            min_samples_leaf=int(forest["min_samples_leaf"]),
            cv=2,
            random_state=seed,
        )
    )


class ClassTransformationModel:
    def __init__(self, seed: int = 42, **params):
        self.model = ClassTransformation(estimator=_classifier(seed, params))

    def fit(self, X, treatment, y):
        self.model.fit(X, y, treatment)
        return self

    def predict_uplift(self, X):
        return np.asarray(self.model.predict(X)).ravel()


MODEL_REGISTRY: dict[str, Callable[..., UpliftModel]] = {
    "random": RandomModel,
    "response_model": ResponseModel,
    "s_learner": _s_learner,
    "t_learner": _t_learner,
    "x_learner": _x_learner,
    "class_transformation": ClassTransformationModel,
    "causal_forest": _causal_forest,
}


def build_model(name: str, seed: int = 42, **params) -> UpliftModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](seed=seed, **params)


_LGBM_SPACE = {
    "num_leaves": ("int", 7, 63),
    "learning_rate": ("float_log", 0.02, 0.2),
    "n_estimators": ("int", 30, 200),
    "min_child_samples": ("int", 10, 200),
}

# Optuna search spaces for the uplift models (not the baselines). Entries are
# (kind, low, high[, step]); causal-forest n_estimators must stay a multiple of 4.
SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "s_learner": _LGBM_SPACE,
    "t_learner": _LGBM_SPACE,
    "x_learner": _LGBM_SPACE,
    "class_transformation": _LGBM_SPACE,
    "causal_forest": {
        "n_estimators": ("int", 48, 200, 4),
        "min_samples_leaf": ("int", 10, 200),
    },
}
