import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pyspark.sql import DataFrame as SparkDataFrame
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


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


def cuped_adjustment(df: pd.DataFrame, outcome_col: str, covariate_cols: list[str]) -> dict:
    """Regression-adjustment CUPED. No true pre-experiment period exists in this dataset, so
    theta is estimated by regressing the outcome on pre-treatment covariates f0-f11 (pooled
    across both arms, ignoring treatment assignment) rather than classic pre-period CUPED."""
    X = sm.add_constant(df[covariate_cols])
    y = df[outcome_col]
    model = sm.OLS(y, X).fit()
    theta = model.params[covariate_cols]
    fitted_covariate_part = X[covariate_cols] @ theta
    adjusted = df[outcome_col] - (fitted_covariate_part - fitted_covariate_part.mean())

    variance_before = df[outcome_col].var(ddof=1)
    variance_after = adjusted.var(ddof=1)
    variance_reduction_pct = 100 * (1 - variance_after / variance_before)

    out_df = df.copy()
    out_df[f"{outcome_col}_cuped"] = adjusted
    return {
        "theta": theta.to_dict(),
        "variance_before": float(variance_before),
        "variance_after": float(variance_after),
        "variance_reduction_pct": float(variance_reduction_pct),
        "adjusted_df": out_df,
    }


def _cohens_h_to_p2(effect_size: float, p1: float) -> float:
    # effect_size = phi2 - phi1 (matches proportion_effectsize(p2, p1) convention used
    # below for observed_effect_size), so a positive effect_size is an increase over p1.
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = phi1 + effect_size
    return float(np.sin(phi2 / 2) ** 2)


def power_analysis(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment",
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict:
    ate_result = compute_ate(df, outcome_col, treatment_col)
    n_control = ate_result["n_control"]
    n_treated = ate_result["n_treated"]
    ratio = n_treated / n_control

    analysis = NormalIndPower()
    # solve_power's power curve is symmetric in effect_size, so the solver can return
    # either sign; abs() gives the minimum detectable *magnitude*, and _cohens_h_to_p2
    # applies it as an increase over control (see its docstring comment).
    mde_effect_size = abs(
        analysis.solve_power(effect_size=None, nobs1=n_control, alpha=alpha, power=power, ratio=ratio)
    )
    p1 = ate_result["control_mean"]
    mde_p2 = _cohens_h_to_p2(mde_effect_size, p1)

    observed_effect_size = proportion_effectsize(ate_result["treated_mean"], ate_result["control_mean"])
    n_required = analysis.solve_power(
        effect_size=observed_effect_size, nobs1=None, alpha=alpha, power=power, ratio=1.0
    )

    return {
        "alpha": alpha,
        "power_target": power,
        "n_control": n_control,
        "n_treated": n_treated,
        "mde_effect_size_cohens_h": float(mde_effect_size),
        "mde_absolute_lift": float(mde_p2 - p1),
        "observed_effect_size_cohens_h": float(observed_effect_size),
        "n_required_per_group_for_observed_effect": float(n_required),
    }
