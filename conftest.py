import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).parent / "src"))

from uplift.spark_env import LOCAL_SPARK_CONFIG, apply_windows_spark_defaults  # noqa: E402

apply_windows_spark_defaults()


@pytest.fixture(scope="session")
def spark():
    builder = (
        SparkSession.builder.master("local[2]")
        .appName("uplift-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
    )
    for key, value in LOCAL_SPARK_CONFIG.items():
        builder = builder.config(key, value)
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture
def synthetic_criteo_pandas():
    def _make(n: int = 1000, treatment_rate: float = 0.846, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        treatment = rng.binomial(1, treatment_rate, size=n)
        features = {f"f{i}": rng.normal(size=n) for i in range(12)}
        uplift_signal = 0.02 * features["f0"]
        base_visit_rate = 0.04
        visit_p = np.clip(base_visit_rate + treatment * (0.01 + uplift_signal), 0.001, 0.99)
        visit = rng.binomial(1, visit_p)
        exposure = np.where(treatment == 1, rng.binomial(1, 0.9, size=n), 0)
        conversion = rng.binomial(1, np.clip(visit * 0.05, 0.0, 1.0))
        return pd.DataFrame(
            {
                **features,
                "treatment": treatment,
                "visit": visit,
                "conversion": conversion,
                "exposure": exposure,
            }
        )

    return _make
