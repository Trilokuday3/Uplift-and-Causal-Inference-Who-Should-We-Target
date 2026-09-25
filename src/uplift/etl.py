import json

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType
from scipy import stats

from uplift.constants import (
    CONVERSION_COL,
    EXPECTED_TREATMENT_RATE,
    EXPOSURE_COL,
    FEATURE_COLS,
    TREATMENT_COL,
    VISIT_COL,
)


class SRMCheckFailed(Exception):
    """Raised when the observed treatment/control split doesn't match the design ratio."""


class CovariateBalanceFailed(Exception):
    """Raised when a covariate's standardized mean difference exceeds the balance threshold."""


CRITEO_SCHEMA = StructType(
    # nullable=True on every field: Spark's CSV data source always marks columns
    # nullable regardless of a declared schema's nullability, so a stricter schema
    # here would never match what read_csv_to_spark actually produces.
    [StructField(c, DoubleType(), True) for c in FEATURE_COLS]
    + [
        StructField(TREATMENT_COL, IntegerType(), True),
        StructField(CONVERSION_COL, IntegerType(), True),
        StructField(VISIT_COL, IntegerType(), True),
        StructField(EXPOSURE_COL, IntegerType(), True),
    ]
)


def read_csv_to_spark(spark: SparkSession, csv_path: str) -> DataFrame:
    return spark.read.csv(csv_path, header=True, schema=CRITEO_SCHEMA)


def write_parquet(spark: SparkSession, df: DataFrame, output_path: str) -> dict:
    rows_in = df.count()
    df.write.mode("overwrite").parquet(output_path)
    rows_out = spark.read.parquet(output_path).count()
    if rows_out != rows_in:
        raise ValueError(f"Row count mismatch after Parquet write: {rows_in} in, {rows_out} out")
    return {"rows_in": rows_in, "rows_out": rows_out}


def compute_srm(
    df: DataFrame,
    treatment_col: str = TREATMENT_COL,
    expected_treatment_rate: float = EXPECTED_TREATMENT_RATE,
) -> dict:
    n_total = df.count()
    n_treated = df.filter(F.col(treatment_col) == 1).count()
    n_control = n_total - n_treated
    expected_treated = n_total * expected_treatment_rate
    expected_control = n_total * (1 - expected_treatment_rate)
    chi2, p_value = stats.chisquare(
        f_obs=[n_treated, n_control], f_exp=[expected_treated, expected_control]
    )
    return {
        "n_total": n_total,
        "n_treated": n_treated,
        "n_control": n_control,
        "observed_treatment_rate": n_treated / n_total,
        "expected_treatment_rate": expected_treatment_rate,
        "chi2": float(chi2),
        "p_value": float(p_value),
    }


def assert_srm_ok(srm_result: dict, alpha: float = 0.01) -> None:
    if srm_result["p_value"] <= alpha:
        raise SRMCheckFailed(
            f"SRM check failed: p={srm_result['p_value']:.4f} <= {alpha}. "
            f"Observed treatment rate {srm_result['observed_treatment_rate']:.4f} vs "
            f"expected {srm_result['expected_treatment_rate']:.4f}."
        )


def compute_covariate_balance(
    df: DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    treatment_col: str = TREATMENT_COL,
) -> pd.DataFrame:
    agg_exprs = []
    for c in feature_cols:
        agg_exprs += [F.mean(c).alias(f"{c}_mean"), F.variance(c).alias(f"{c}_var")]
    stats_df = df.groupBy(treatment_col).agg(*agg_exprs).toPandas().set_index(treatment_col)

    rows = []
    for c in feature_cols:
        mean_t, mean_c = stats_df.loc[1, f"{c}_mean"], stats_df.loc[0, f"{c}_mean"]
        var_t, var_c = stats_df.loc[1, f"{c}_var"], stats_df.loc[0, f"{c}_var"]
        pooled_sd = ((var_t + var_c) / 2) ** 0.5
        smd = abs((mean_t - mean_c) / pooled_sd) if pooled_sd > 0 else 0.0
        rows.append({"feature": c, "mean_treated": mean_t, "mean_control": mean_c, "smd": smd})
    return pd.DataFrame(rows)


def assert_balanced(balance_df: pd.DataFrame, threshold: float = 0.1) -> None:
    violations = balance_df[balance_df["smd"] >= threshold]
    if not violations.empty:
        raise CovariateBalanceFailed(
            f"Covariate balance check failed for: {violations['feature'].tolist()} (SMD >= {threshold})"
        )


def stratified_split(
    df: DataFrame,
    treatment_col: str = TREATMENT_COL,
    outcome_col: str = VISIT_COL,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[DataFrame, DataFrame, DataFrame]:
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1.0"
    train_cut = ratios[0] * 100
    val_cut = (ratios[0] + ratios[1]) * 100

    bucketed = df.withColumn("_row_id", F.monotonically_increasing_id()).withColumn(
        "_bucket", F.pmod(F.hash(F.col("_row_id"), F.lit(seed)), F.lit(100))
    )
    train_df = bucketed.filter(F.col("_bucket") < train_cut).drop("_row_id", "_bucket")
    val_df = bucketed.filter((F.col("_bucket") >= train_cut) & (F.col("_bucket") < val_cut)).drop(
        "_row_id", "_bucket"
    )
    test_df = bucketed.filter(F.col("_bucket") >= val_cut).drop("_row_id", "_bucket")
    return train_df, val_df, test_df


def write_split_meta(
    splits: dict[str, DataFrame],
    seed: int,
    output_path: str,
    treatment_col: str = TREATMENT_COL,
    outcome_col: str = VISIT_COL,
) -> dict:
    meta = {"seed": seed, "splits": {}}
    for name, split_df in splits.items():
        n = split_df.count()
        treat_rate = split_df.filter(F.col(treatment_col) == 1).count() / n
        outcome_rate = split_df.filter(F.col(outcome_col) == 1).count() / n
        meta["splits"][name] = {"n_rows": n, "treatment_rate": treat_rate, "outcome_rate": outcome_rate}
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta
