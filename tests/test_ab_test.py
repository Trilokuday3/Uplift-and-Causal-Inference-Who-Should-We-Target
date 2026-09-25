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


def test_ci_bootstrap_spark_is_deterministic(spark, synthetic_criteo_pandas):
    from uplift.ab_test import ci_bootstrap_spark

    pdf = synthetic_criteo_pandas(n=2_000)
    sdf = spark.createDataFrame(pdf)

    result1 = ci_bootstrap_spark(sdf, outcome_col="visit", n_boot=20, seed=42)
    result2 = ci_bootstrap_spark(sdf, outcome_col="visit", n_boot=20, seed=42)

    assert result1 == result2


def test_ci_bootstrap_spark_brackets_normal_approx(spark, synthetic_criteo_pandas):
    from uplift.ab_test import ci_bootstrap_spark, ci_normal_approx

    pdf = synthetic_criteo_pandas(n=5_000)
    sdf = spark.createDataFrame(pdf)

    normal_result = ci_normal_approx(pdf, outcome_col="visit")
    boot_result = ci_bootstrap_spark(sdf, outcome_col="visit", n_boot=50, seed=42)

    # the two CIs should substantially overlap on the same data
    overlap_low = max(normal_result["ci_low"], boot_result["ci_low"])
    overlap_high = min(normal_result["ci_high"], boot_result["ci_high"])
    assert overlap_low < overlap_high


def test_cuped_reduces_variance(synthetic_criteo_pandas):
    from uplift.ab_test import cuped_adjustment

    df = synthetic_criteo_pandas(n=50_000)
    # make visit correlated with f0 so CUPED has something to adjust for
    df["visit"] = (df["visit"] + (df["f0"] > 1.5).astype(int)).clip(0, 1)

    result = cuped_adjustment(df, outcome_col="visit", covariate_cols=[f"f{i}" for i in range(12)])

    assert result["variance_after"] < result["variance_before"]
    assert result["variance_reduction_pct"] > 0


def test_cuped_preserves_ate_unbiasedness(synthetic_criteo_pandas):
    from uplift.ab_test import compute_ate, cuped_adjustment

    df = synthetic_criteo_pandas(n=50_000, seed=7)
    raw_ate = compute_ate(df, outcome_col="visit")["ate"]

    result = cuped_adjustment(df, outcome_col="visit", covariate_cols=[f"f{i}" for i in range(12)])
    adjusted_ate = compute_ate(result["adjusted_df"], outcome_col="visit_cuped")["ate"]

    assert abs(adjusted_ate - raw_ate) < 0.005  # point estimate barely moves; only variance drops
