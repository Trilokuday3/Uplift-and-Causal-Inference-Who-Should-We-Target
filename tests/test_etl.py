import pytest


def test_write_parquet_round_trips_row_count(spark, tmp_path, synthetic_criteo_pandas):
    from uplift.etl import write_parquet

    pdf = synthetic_criteo_pandas(n=200)
    sdf = spark.createDataFrame(pdf)
    out_path = str(tmp_path / "out.parquet")

    result = write_parquet(spark, sdf, out_path)

    assert result == {"rows_in": 200, "rows_out": 200}
    assert spark.read.parquet(out_path).count() == 200


def test_read_csv_to_spark_applies_schema(spark, tmp_path):
    from uplift.etl import read_csv_to_spark, CRITEO_SCHEMA

    csv_path = tmp_path / "raw.csv"
    header = ",".join([f"f{i}" for i in range(12)] + ["treatment", "conversion", "visit", "exposure"])
    row = ",".join(["0.1"] * 12 + ["1", "0", "1", "1"])
    csv_path.write_text(header + "\n" + row + "\n")

    df = read_csv_to_spark(spark, str(csv_path))

    assert df.schema == CRITEO_SCHEMA
    assert df.count() == 1
