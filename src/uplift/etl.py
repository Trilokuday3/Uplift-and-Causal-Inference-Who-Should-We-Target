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
