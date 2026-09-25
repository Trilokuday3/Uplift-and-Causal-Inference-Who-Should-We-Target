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


def test_srm_passes_on_clean_split(spark, synthetic_criteo_pandas):
    from uplift.etl import assert_srm_ok, compute_srm

    pdf = synthetic_criteo_pandas(n=20_000, treatment_rate=0.846)
    sdf = spark.createDataFrame(pdf)

    result = compute_srm(sdf)

    assert result["p_value"] > 0.01
    assert_srm_ok(result)  # must not raise


def test_srm_detects_break(spark, synthetic_criteo_pandas):
    from uplift.etl import SRMCheckFailed, assert_srm_ok, compute_srm

    pdf = synthetic_criteo_pandas(n=20_000, treatment_rate=0.90)  # broken: should be 0.846
    sdf = spark.createDataFrame(pdf)

    result = compute_srm(sdf, expected_treatment_rate=0.846)

    assert result["p_value"] <= 0.01
    with pytest.raises(SRMCheckFailed):
        assert_srm_ok(result)


def test_covariate_balance_all_features_reported(spark, synthetic_criteo_pandas):
    from uplift.etl import compute_covariate_balance

    pdf = synthetic_criteo_pandas(n=20_000)
    sdf = spark.createDataFrame(pdf)

    balance_df = compute_covariate_balance(sdf)

    assert len(balance_df) == 12
    assert set(balance_df["feature"]) == {f"f{i}" for i in range(12)}
    assert (balance_df["smd"] < 0.1).all()  # synthetic features are treatment-independent


def test_covariate_balance_detects_violation(spark, synthetic_criteo_pandas):
    from uplift.etl import CovariateBalanceFailed, assert_balanced, compute_covariate_balance

    pdf = synthetic_criteo_pandas(n=20_000)
    pdf.loc[pdf["treatment"] == 1, "f0"] += 5.0  # inject a large imbalance
    sdf = spark.createDataFrame(pdf)

    balance_df = compute_covariate_balance(sdf)

    assert balance_df.loc[balance_df["feature"] == "f0", "smd"].iloc[0] >= 0.1
    with pytest.raises(CovariateBalanceFailed):
        assert_balanced(balance_df)
