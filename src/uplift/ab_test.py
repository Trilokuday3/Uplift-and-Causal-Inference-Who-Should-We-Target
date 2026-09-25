import math

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from scipy import stats


def compute_ate(df: pd.DataFrame, outcome_col: str, treatment_col: str = "treatment") -> dict:
    treated = df.loc[df[treatment_col] == 1, outcome_col]
    control = df.loc[df[treatment_col] == 0, outcome_col]
    treated_mean = treated.mean()
    control_mean = control.mean()
    ate = treated_mean - control_mean
    relative_lift = ate / control_mean if control_mean != 0 else float("nan")
    return {
        "treated_mean": treated_mean,
        "control_mean": control_mean,
        "ate": ate,
        "relative_lift": relative_lift,
        "n_treated": int(treated.shape[0]),
        "n_control": int(control.shape[0]),
    }


def ci_normal_approx(
    df: pd.DataFrame, outcome_col: str, treatment_col: str = "treatment", alpha: float = 0.05
) -> dict:
    ate_result = compute_ate(df, outcome_col, treatment_col)
    treated = df.loc[df[treatment_col] == 1, outcome_col]
    control = df.loc[df[treatment_col] == 0, outcome_col]
    se = math.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    z = stats.norm.ppf(1 - alpha / 2)
    ate = ate_result["ate"]
    return {"ate": ate, "se": se, "ci_low": ate - z * se, "ci_high": ate + z * se, "alpha": alpha}


def ci_bootstrap_spark(
    spark_df: SparkDataFrame,
    outcome_col: str,
    treatment_col: str = "treatment",
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    estimates = []
    for i in range(n_boot):
        sample = spark_df.sample(withReplacement=True, fraction=1.0, seed=seed + i)
        rows = sample.groupBy(treatment_col).avg(outcome_col).collect()
        means = {r[treatment_col]: r[f"avg({outcome_col})"] for r in rows}
        if 1 in means and 0 in means:
            estimates.append(means[1] - means[0])
    estimates = np.array(estimates)
    ci_low, ci_high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "ate_mean": float(estimates.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_boot": int(len(estimates)),
        "seed": seed,
    }
