import pytest
from pyspark.sql import functions as F


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

    pdf = synthetic_criteo_pandas(n=20_000, treatment_rate=0.85)
    sdf = spark.createDataFrame(pdf)

    result = compute_srm(sdf)

    assert result["p_value"] > 0.01
    assert_srm_ok(result)  # must not raise


def test_srm_detects_break(spark, synthetic_criteo_pandas):
    from uplift.etl import SRMCheckFailed, assert_srm_ok, compute_srm

    pdf = synthetic_criteo_pandas(n=20_000, treatment_rate=0.90)  # broken: should be 0.85
    sdf = spark.createDataFrame(pdf)

    result = compute_srm(sdf, expected_treatment_rate=0.85)

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


def test_split_ratios_are_approximately_correct(spark, synthetic_criteo_pandas):
    from uplift.etl import stratified_split

    pdf = synthetic_criteo_pandas(n=50_000)
    sdf = spark.createDataFrame(pdf)

    train, val, test = stratified_split(sdf, seed=42)
    n = sdf.count()

    assert abs(train.count() / n - 0.6) < 0.02
    assert abs(val.count() / n - 0.2) < 0.02
    assert abs(test.count() / n - 0.2) < 0.02


def test_split_is_stratified_on_treatment_and_visit(spark, synthetic_criteo_pandas):
    from uplift.etl import stratified_split

    pdf = synthetic_criteo_pandas(n=50_000)
    sdf = spark.createDataFrame(pdf)
    full_treat_rate = pdf["treatment"].mean()
    full_visit_rate = pdf["visit"].mean()

    train, val, test = stratified_split(sdf, seed=42)

    for split_df in (train, val, test):
        n = split_df.count()
        treat_rate = split_df.filter(F.col("treatment") == 1).count() / n
        visit_rate = split_df.filter(F.col("visit") == 1).count() / n
        assert abs(treat_rate - full_treat_rate) < 0.02
        assert abs(visit_rate - full_visit_rate) < 0.02


def test_split_determinism(spark, synthetic_criteo_pandas):
    from uplift.etl import stratified_split

    pdf = synthetic_criteo_pandas(n=10_000)
    sdf = spark.createDataFrame(pdf)

    train1, val1, test1 = stratified_split(sdf, seed=42)
    train2, val2, test2 = stratified_split(sdf, seed=42)

    assert train1.count() == train2.count()
    assert val1.count() == val2.count()
    assert test1.count() == test2.count()


def test_split_is_truly_stratified_per_cell(spark):
    """A random hash split only balances strata approximately, by luck of large numbers.
    True stratification gives exact per-cell proportions. Build cells with clean divisors
    so the correct answer is an exact integer, not just close."""
    from uplift.etl import stratified_split
    import pandas as pd

    rows = []
    for treatment in [0, 1]:
        for visit in [0, 1]:
            for i in range(200):
                rows.append({"treatment": treatment, "visit": visit, "id": f"{treatment}-{visit}-{i}"})
    pdf = pd.DataFrame(rows)
    sdf = spark.createDataFrame(pdf)

    train, val, test = stratified_split(sdf, seed=42, ratios=(0.6, 0.2, 0.2))

    for treatment in [0, 1]:
        for visit in [0, 1]:
            cell = F.col("treatment") == treatment
            cell = cell & (F.col("visit") == visit)
            assert train.filter(cell).count() == 120
            assert val.filter(cell).count() == 40
            assert test.filter(cell).count() == 40


def test_split_determinism_independent_of_partitioning(spark, synthetic_criteo_pandas):
    """monotonically_increasing_id() depends on how Spark partitions the input, so the
    same seed on a differently-partitioned copy of the same data must still assign every
    row to the same split for the split to be reproducible on another machine/run."""
    from uplift.etl import stratified_split

    pdf = synthetic_criteo_pandas(n=2_000)
    sdf_one_partition = spark.createDataFrame(pdf).repartition(1)
    sdf_many_partitions = spark.createDataFrame(pdf).repartition(8)

    train1, _, _ = stratified_split(sdf_one_partition, seed=42)
    train2, _, _ = stratified_split(sdf_many_partitions, seed=42)

    cols = sorted(train1.columns)
    train1_rows = {tuple(r) for r in train1.select(*cols).collect()}
    train2_rows = {tuple(r) for r in train2.select(*cols).collect()}
    assert train1_rows == train2_rows


def test_write_split_meta(spark, tmp_path, synthetic_criteo_pandas):
    from uplift.etl import stratified_split, write_split_meta
    import json

    pdf = synthetic_criteo_pandas(n=10_000)
    sdf = spark.createDataFrame(pdf)
    train, val, test = stratified_split(sdf, seed=42)
    out_path = str(tmp_path / "split_meta.json")

    meta = write_split_meta({"train": train, "val": val, "test": test}, seed=42, output_path=out_path)

    assert meta["seed"] == 42
    assert set(meta["splits"].keys()) == {"train", "val", "test"}
    with open(out_path) as f:
        assert json.load(f) == meta
