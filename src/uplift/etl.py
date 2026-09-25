from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from uplift.constants import (
    CONVERSION_COL,
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
