def test_constants_importable():
    from uplift.constants import (
        FEATURE_COLS,
        TREATMENT_COL,
        EXPOSURE_COL,
        VISIT_COL,
        CONVERSION_COL,
        EXPECTED_TREATMENT_RATE,
    )

    assert FEATURE_COLS == [f"f{i}" for i in range(12)]
    assert TREATMENT_COL == "treatment"
    assert EXPOSURE_COL == "exposure"
    assert VISIT_COL == "visit"
    assert CONVERSION_COL == "conversion"
    assert 0 < EXPECTED_TREATMENT_RATE < 1


def test_synthetic_fixture_shape(synthetic_criteo_pandas):
    df = synthetic_criteo_pandas(n=500)
    assert len(df) == 500
    for col in ["f0", "f11", "treatment", "visit", "conversion", "exposure"]:
        assert col in df.columns


def test_spark_session_works(spark):
    df = spark.createDataFrame([(1,), (2,)], ["x"])
    assert df.count() == 2
