import math

import pandas as pd


def test_compute_ate_basic():
    from uplift.ab_test import compute_ate

    df = pd.DataFrame(
        {"treatment": [1, 1, 1, 0, 0, 0], "visit": [1, 1, 0, 0, 0, 0]}
    )
    result = compute_ate(df, outcome_col="visit")

    assert math.isclose(result["treated_mean"], 2 / 3, rel_tol=1e-9)
    assert result["control_mean"] == 0.0
    assert math.isclose(result["ate"], 2 / 3, rel_tol=1e-9)
    assert result["n_treated"] == 3
    assert result["n_control"] == 3


def test_ci_normal_approx_contains_true_effect_on_large_sample(synthetic_criteo_pandas):
    from uplift.ab_test import ci_normal_approx

    df = synthetic_criteo_pandas(n=200_000)
    result = ci_normal_approx(df, outcome_col="visit")

    assert result["ci_low"] < result["ate"] < result["ci_high"]
    assert result["se"] > 0
