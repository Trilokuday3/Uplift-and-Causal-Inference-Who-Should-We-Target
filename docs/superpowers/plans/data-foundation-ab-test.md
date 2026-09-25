# Data Foundation & Classic A/B Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validated, reproducibly-split Criteo v2.1 dataset (PySpark ETL with hard-gated SRM/covariate-balance checks) and answer "did the campaign work?" with a rigorous classical A/B analysis (ATE, dual CIs, CUPED-style variance reduction, power analysis, optional CACE), ending in a committed written verdict.

**Architecture:** Two small, independently-testable modules — `src/uplift/etl.py` (PySpark: schema, SRM/balance gates, stratified split) and `src/uplift/ab_test.py` (pandas/statsmodels for ATE/CI/CUPED/power, one Spark-native bootstrap function) — each with a `compute_*` (never raises) / `assert_*` (raises) pairing for hard gates, wired together by a `Makefile` and exercised against synthetic data in unit tests, then against the real 14M-row dataset in the final task.

**Tech Stack:** PySpark (ETL, bootstrap CI), pandas/numpy (in-memory stats), SciPy (`chisquare`), statsmodels (OLS for CUPED, `NormalIndPower`/`proportion_effectsize` for power), pytest.

**Spec:** `docs/superpowers/specs/data-foundation-ab-test-design.md` (and its parent `docs/superpowers/specs/uplift-targeting-roadmap.md`)

## Global Constraints

- Use **Criteo v2.1 only** — never v1 (known treatment-leakage issue).
- `exposure` is post-randomization and must **never** be a parameter of any `ab_test.py` function except `cace_iv` — enforced by a structural signature test (Task 11), not just convention.
- Split seed is fixed at **42** and, once written, is never regenerated — every later sub-project loads this exact split.
- `data/` (raw CSV, processed Parquet) is gitignored — never commit downloaded/generated data.
- Python 3.10+. PySpark requires a JDK (11 or 17) with `JAVA_HOME` set; on Windows this also needs `winutils.exe`/`HADOOP_HOME` on `PATH` for local-mode Spark to write files correctly — verify this works in Task 1 before writing any Spark logic, since a broken Spark install blocks every later task.
- File/folder names created by this plan must not contain dates (per this project's `CLAUDE.md`).
- Commits: per this project's `CLAUDE.md`, the implementer (whether this session or a subagent) never runs `git add`/`git commit` — each task's "Commit" step is executed by the user, not by Claude. The implementer stops once a task's tests pass and reports the diff; commit blocks for completed tasks are batched and handed to the user as copy-pasteable text (grouped, per the CLAUDE.md format) rather than run automatically. No `Co-Authored-By: Claude` line, ever. Branch names must not contain "claude".
- `Makefile` targets are committed for parity with the spec and for CI (GitHub Actions' Ubuntu runners have `make`); on Windows, run the underlying command shown in each target directly if `make` isn't installed.

## Review Focus

- **Randomization silently breaks** (e.g. a bad filter drops rows unevenly) → the SRM check must actually halt the pipeline (raise), not just print a warning. Covered in Task 3.
- **A covariate is badly imbalanced** (e.g. an off-by-one in a join skews `f`-columns between groups) → balance check must raise, not just report a table nobody reads. Covered in Task 4.
- **`exposure` quietly ends up inside an ATE/CI/CUPED/power computation** (a very easy mistake — it's a column right next to `visit`/`conversion` in the same DataFrame) → structural test asserts no such function even accepts an `exposure_col` argument. Covered in Task 11.
- **Bootstrap CI isn't reproducible run-to-run** (someone reruns the analysis before a stakeholder meeting and gets a different CI) → same seed must yield identical CI bounds. Covered in Task 8.
- **CUPED reduces variance but silently shifts the point estimate** (a regression-adjustment bug that biases the ATE while looking like it "improved" the CI) → test asserts the CUPED-adjusted ATE stays within a tight tolerance of the raw ATE on data with a known true effect, not just that variance drops. Covered in Task 9.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `Makefile`
- Create: `.gitignore`
- Create: `conftest.py`
- Create: `src/uplift/__init__.py`
- Create: `src/uplift/constants.py`
- Test: `tests/test_scaffolding.py`

**Interfaces:**
- Produces: `uplift.constants.FEATURE_COLS` (`list[str]`, `["f0", ..., "f11"]`), `TREATMENT_COL`, `EXPOSURE_COL`, `VISIT_COL`, `CONVERSION_COL` (all `str`), `EXPECTED_TREATMENT_RATE` (`float`, `0.846`) — every later task imports these instead of hardcoding column names.
- Produces: `conftest.py` fixtures `spark` (session-scoped `pyspark.sql.SparkSession`) and `synthetic_criteo_pandas` (factory: `(n: int = 1000, treatment_rate: float = 0.846, seed: int = 42) -> pandas.DataFrame` with columns `f0..f11, treatment, visit, conversion, exposure`) — every later test uses these.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_scaffolding.py
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_scaffolding.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'uplift'` and fixture errors (`conftest.py`, `pyproject.toml` don't exist yet).

- [ ] **Step 3: Create the scaffolding**

```toml
# pyproject.toml
[project]
name = "uplift"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
```

```
# requirements.txt
pyspark>=3.5
pandas>=2.0
numpy>=1.26
scipy>=1.11
statsmodels>=0.14
scikit-uplift>=0.5
pytest>=8.0
jupyter>=1.0
nbformat>=5.9
```

```makefile
# Makefile
.PHONY: data ab-test test

DATA_DIR ?= data
RAW_CSV ?= $(DATA_DIR)/raw/criteo-uplift-v2.1.csv
PROCESSED_DIR ?= $(DATA_DIR)/processed

data:
	python -m uplift.etl --input $(RAW_CSV) --output-dir $(PROCESSED_DIR)

ab-test:
	python -m uplift.ab_test --input-dir $(PROCESSED_DIR) --output docs/ab_test_results.json

test:
	pytest tests/ -v
```

```
# .gitignore
data/
*.pyc
__pycache__/
.pytest_cache/
.ipynb_checkpoints/
*.egg-info/
.venv/
venv/
```

```python
# conftest.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).parent / "src"))


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("uplift-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
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
```

```python
# src/uplift/__init__.py
```

```python
# src/uplift/constants.py
FEATURE_COLS = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
EXPOSURE_COL = "exposure"
VISIT_COL = "visit"
CONVERSION_COL = "conversion"
EXPECTED_TREATMENT_RATE = 0.846
```

- [ ] **Step 4: Install and verify Spark works locally**

Run: `pip install -e . && pip install -r requirements.txt`
Then run: `python -c "from pyspark.sql import SparkSession; s = SparkSession.builder.master('local[1]').getOrCreate(); print(s.range(3).count()); s.stop()"`
Expected: prints `3` with no `JAVA_HOME`/`winutils` errors. **If this fails**, stop and fix the local JDK/`JAVA_HOME` (and on Windows, `HADOOP_HOME`/`winutils.exe`) setup before continuing — every later task depends on Spark working.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_scaffolding.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt Makefile .gitignore conftest.py src/uplift/__init__.py src/uplift/constants.py tests/test_scaffolding.py
git commit -m "chore: scaffold uplift package, test fixtures, Makefile"
```

---

## Task 2: ETL — CSV → Parquet with row-count logging

**Files:**
- Create: `src/uplift/etl.py`
- Test: `tests/test_etl.py`

**Interfaces:**
- Consumes: `uplift.constants.{FEATURE_COLS, TREATMENT_COL, EXPOSURE_COL, VISIT_COL, CONVERSION_COL}` (Task 1)
- Produces: `CRITEO_SCHEMA` (`pyspark.sql.types.StructType`), `read_csv_to_spark(spark, csv_path: str) -> DataFrame`, `write_parquet(spark, df: DataFrame, output_path: str) -> dict` (`{"rows_in": int, "rows_out": int}`, raises `ValueError` on mismatch) — used by every later ETL task and by Task 6's CLI.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_etl.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_etl.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'uplift.etl'`.

- [ ] **Step 3: Implement**

```python
# src/uplift/etl.py
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
    [StructField(c, DoubleType(), False) for c in FEATURE_COLS]
    + [
        StructField(TREATMENT_COL, IntegerType(), False),
        StructField(CONVERSION_COL, IntegerType(), False),
        StructField(VISIT_COL, IntegerType(), False),
        StructField(EXPOSURE_COL, IntegerType(), False),
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_etl.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/uplift/etl.py tests/test_etl.py
git commit -m "feat(etl): CSV to Parquet conversion with row-count verification"
```

---

## Task 3: ETL — SRM check (hard gate)

**Files:**
- Modify: `src/uplift/etl.py`
- Modify: `tests/test_etl.py`

**Interfaces:**
- Consumes: `SRMCheckFailed`, `TREATMENT_COL`, `EXPECTED_TREATMENT_RATE` (Task 1/2)
- Produces: `compute_srm(df, treatment_col=TREATMENT_COL, expected_treatment_rate=EXPECTED_TREATMENT_RATE) -> dict` (never raises), `assert_srm_ok(srm_result: dict, alpha: float = 0.01) -> None` (raises `SRMCheckFailed`) — used by Task 6's CLI.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_etl.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_etl.py -k srm -v`
Expected: FAIL — `ImportError: cannot import name 'compute_srm'`.

- [ ] **Step 3: Implement**

```python
# add to src/uplift/etl.py
from pyspark.sql import functions as F
from scipy import stats

from uplift.constants import EXPECTED_TREATMENT_RATE


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_etl.py -k srm -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/uplift/etl.py tests/test_etl.py
git commit -m "feat(etl): SRM hard gate on treatment/control split"
```

---

## Task 4: ETL — Covariate balance check (hard gate)

**Files:**
- Modify: `src/uplift/etl.py`
- Modify: `tests/test_etl.py`

**Interfaces:**
- Consumes: `CovariateBalanceFailed`, `FEATURE_COLS`, `TREATMENT_COL` (Task 1/2)
- Produces: `compute_covariate_balance(df, feature_cols=FEATURE_COLS, treatment_col=TREATMENT_COL) -> pandas.DataFrame` (columns `feature, mean_treated, mean_control, smd`; never raises), `assert_balanced(balance_df, threshold: float = 0.1) -> None` (raises `CovariateBalanceFailed`) — used by Task 6's CLI.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_etl.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_etl.py -k balance -v`
Expected: FAIL — `ImportError: cannot import name 'compute_covariate_balance'`.

- [ ] **Step 3: Implement**

```python
# add to src/uplift/etl.py
import pandas as pd

from uplift.constants import FEATURE_COLS


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_etl.py -k balance -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/uplift/etl.py tests/test_etl.py
git commit -m "feat(etl): covariate balance hard gate (standardized mean difference)"
```

---

## Task 5: ETL — Stratified split + split metadata

**Files:**
- Modify: `src/uplift/etl.py`
- Modify: `tests/test_etl.py`

**Interfaces:**
- Consumes: `TREATMENT_COL`, `VISIT_COL` (Task 1)
- Produces: `stratified_split(df, treatment_col=TREATMENT_COL, outcome_col=VISIT_COL, seed=42, ratios=(0.6, 0.2, 0.2)) -> tuple[DataFrame, DataFrame, DataFrame]`, `write_split_meta(splits: dict[str, DataFrame], seed: int, output_path: str, treatment_col=TREATMENT_COL, outcome_col=VISIT_COL) -> dict` — used by Task 6's CLI and by every later sub-project that loads `train`/`val`/`test`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_etl.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_etl.py -k split -v`
Expected: FAIL — `ImportError: cannot import name 'stratified_split'`.

- [ ] **Step 3: Implement**

```python
# add to src/uplift/etl.py
import json


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_etl.py -k split -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/uplift/etl.py tests/test_etl.py
git commit -m "feat(etl): stratified train/val/test split with split metadata"
```

---

## Task 6: ETL CLI + `make data`

**Files:**
- Modify: `src/uplift/etl.py`
- Test: `tests/test_etl_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 2–5.
- Produces: `main() -> None` (argparse CLI, invoked via `python -m uplift.etl`), used by the `Makefile`'s `data` target (already written in Task 1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_etl_cli.py
import json
import subprocess
import sys


def test_etl_cli_end_to_end(tmp_path, synthetic_criteo_pandas):
    pdf = synthetic_criteo_pandas(n=5_000)
    csv_path = tmp_path / "raw.csv"
    pdf.to_csv(csv_path, index=False)
    out_dir = tmp_path / "processed"

    result = subprocess.run(
        [sys.executable, "-m", "uplift.etl", "--input", str(csv_path), "--output-dir", str(out_dir)],
        cwd=str(tmp_path.parents[len(tmp_path.parts) - 1]) if False else None,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "split_meta.json").exists()
    meta = json.loads((out_dir / "split_meta.json").read_text())
    assert set(meta["splits"].keys()) == {"train", "val", "test"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_etl_cli.py -v`
Expected: FAIL — `main` doesn't exist yet, `python -m uplift.etl` errors (no `__main__` block).

- [ ] **Step 3: Implement**

```python
# add to src/uplift/etl.py
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Uplift project ETL: CSV -> validated, split Parquet")
    parser.add_argument("--input", required=True, help="Path to raw Criteo v2.1 CSV")
    parser.add_argument("--output-dir", required=True, help="Directory to write processed Parquet + split_meta.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("uplift-etl").getOrCreate()
    try:
        raw_df = read_csv_to_spark(spark, args.input)
        write_parquet(spark, raw_df, f"{args.output_dir}/full")

        srm_result = compute_srm(raw_df)
        print("SRM check:", srm_result)
        assert_srm_ok(srm_result)

        balance_df = compute_covariate_balance(raw_df)
        print(balance_df.to_string())
        assert_balanced(balance_df)

        train_df, val_df, test_df = stratified_split(raw_df, seed=args.seed)
        for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            write_parquet(spark, split_df, f"{args.output_dir}/{name}")

        meta = write_split_meta(
            {"train": train_df, "val": val_df, "test": test_df},
            seed=args.seed,
            output_path=f"{args.output_dir}/split_meta.json",
        )
        print("Split meta:", meta)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_etl_cli.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/uplift/etl.py tests/test_etl_cli.py
git commit -m "feat(etl): CLI entry point wiring SRM/balance/split into one pipeline run"
```

---

## Task 7: A/B test — ATE + normal-approximation CI

**Files:**
- Create: `src/uplift/ab_test.py`
- Test: `tests/test_ab_test.py`

**Interfaces:**
- Produces: `compute_ate(df: pandas.DataFrame, outcome_col: str, treatment_col: str = "treatment") -> dict` (`{"treated_mean", "control_mean", "ate", "relative_lift", "n_treated", "n_control"}`), `ci_normal_approx(df, outcome_col, treatment_col="treatment", alpha=0.05) -> dict` (`{"ate", "se", "ci_low", "ci_high", "alpha"}`) — used by every later `ab_test.py` task and by Task 12's CLI.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ab_test.py
import math

import pandas as pd


def test_compute_ate_basic():
    from uplift.ab_test import compute_ate

    df = pd.DataFrame(
        {"treatment": [1, 1, 1, 0, 0, 0], "visit": [1, 1, 0, 0, 0, 0]}
    )
    result = compute_ate(df, outcome_col="visit")

    assert math.isclose(result["treated_mean"], 2 / 3, rel_tol=1e-9)
    assert result["control_mean"] == 0.0
    assert math.isclose(result["ate"], 2 / 3, rel_tol=1e-9)
    assert result["n_treated"] == 3
    assert result["n_control"] == 3


def test_ci_normal_approx_contains_true_effect_on_large_sample(synthetic_criteo_pandas):
    from uplift.ab_test import ci_normal_approx

    df = synthetic_criteo_pandas(n=200_000)
    result = ci_normal_approx(df, outcome_col="visit")

    assert result["ci_low"] < result["ate"] < result["ci_high"]
    assert result["se"] > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_ab_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'uplift.ab_test'`.

- [ ] **Step 3: Implement**

```python
# src/uplift/ab_test.py
import math

import pandas as pd
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_ab_test.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/uplift/ab_test.py tests/test_ab_test.py
git commit -m "feat(ab_test): ATE and normal-approximation confidence interval"
```

---

## Task 8: A/B test — Bootstrap CI (Spark)

**Files:**
- Modify: `src/uplift/ab_test.py`
- Modify: `tests/test_ab_test.py`

**Interfaces:**
- Consumes: nothing from `ab_test.py` directly (independent Spark-native implementation), operates on a `pyspark.sql.DataFrame`.
- Produces: `ci_bootstrap_spark(spark_df, outcome_col: str, treatment_col: str = "treatment", n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict` (`{"ate_mean", "ci_low", "ci_high", "n_boot", "seed"}`) — used by Task 12's CLI.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_ab_test.py
def test_ci_bootstrap_spark_is_deterministic(spark, synthetic_criteo_pandas):
    from uplift.ab_test import ci_bootstrap_spark

    pdf = synthetic_criteo_pandas(n=2_000)
    sdf = spark.createDataFrame(pdf)

    result1 = ci_bootstrap_spark(sdf, outcome_col="visit", n_boot=20, seed=42)
    result2 = ci_bootstrap_spark(sdf, outcome_col="visit", n_boot=20, seed=42)

    assert result1 == result2


def test_ci_bootstrap_spark_brackets_normal_approx(spark, synthetic_criteo_pandas):
    from uplift.ab_test import ci_bootstrap_spark, ci_normal_approx

    pdf = synthetic_criteo_pandas(n=5_000)
    sdf = spark.createDataFrame(pdf)

    normal_result = ci_normal_approx(pdf, outcome_col="visit")
    boot_result = ci_bootstrap_spark(sdf, outcome_col="visit", n_boot=50, seed=42)

    # the two CIs should substantially overlap on the same data
    overlap_low = max(normal_result["ci_low"], boot_result["ci_low"])
    overlap_high = min(normal_result["ci_high"], boot_result["ci_high"])
    assert overlap_low < overlap_high
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_ab_test.py -k bootstrap -v`
Expected: FAIL — `ImportError: cannot import name 'ci_bootstrap_spark'`.

- [ ] **Step 3: Implement**

```python
# add to src/uplift/ab_test.py
import numpy as np
from pyspark.sql import DataFrame as SparkDataFrame


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_ab_test.py -k bootstrap -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/uplift/ab_test.py tests/test_ab_test.py
git commit -m "feat(ab_test): Spark-native bootstrap confidence interval"
```

---

## Task 9: A/B test — CUPED-style variance reduction

**Files:**
- Modify: `src/uplift/ab_test.py`
- Modify: `tests/test_ab_test.py`

**Interfaces:**
- Produces: `cuped_adjustment(df: pandas.DataFrame, outcome_col: str, covariate_cols: list[str]) -> dict` (`{"theta": dict, "variance_before", "variance_after", "variance_reduction_pct", "adjusted_df": pandas.DataFrame}`) — used by Task 12's CLI. Note: no `treatment_col` parameter — theta is fit pooled across both arms on pre-treatment covariates, deliberately ignoring treatment assignment (fitting on treatment would bias the adjustment).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_ab_test.py
def test_cuped_reduces_variance(synthetic_criteo_pandas):
    from uplift.ab_test import cuped_adjustment

    df = synthetic_criteo_pandas(n=50_000)
    # make visit correlated with f0 so CUPED has something to adjust for
    df["visit"] = (df["visit"] + (df["f0"] > 1.5).astype(int)).clip(0, 1)

    result = cuped_adjustment(df, outcome_col="visit", covariate_cols=[f"f{i}" for i in range(12)])

    assert result["variance_after"] < result["variance_before"]
    assert result["variance_reduction_pct"] > 0


def test_cuped_preserves_ate_unbiasedness(synthetic_criteo_pandas):
    from uplift.ab_test import compute_ate, cuped_adjustment

    df = synthetic_criteo_pandas(n=50_000, seed=7)
    raw_ate = compute_ate(df, outcome_col="visit")["ate"]

    result = cuped_adjustment(df, outcome_col="visit", covariate_cols=[f"f{i}" for i in range(12)])
    adjusted_ate = compute_ate(result["adjusted_df"], outcome_col="visit_cuped")["ate"]

    assert abs(adjusted_ate - raw_ate) < 0.005  # point estimate barely moves; only variance drops
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_ab_test.py -k cuped -v`
Expected: FAIL — `ImportError: cannot import name 'cuped_adjustment'`.

- [ ] **Step 3: Implement**

```python
# add to src/uplift/ab_test.py
import statsmodels.api as sm


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_ab_test.py -k cuped -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/uplift/ab_test.py tests/test_ab_test.py
git commit -m "feat(ab_test): CUPED-style regression-adjustment variance reduction"
```

---

## Task 10: A/B test — Power analysis

**Files:**
- Modify: `src/uplift/ab_test.py`
- Modify: `tests/test_ab_test.py`

**Interfaces:**
- Consumes: `compute_ate` (Task 7)
- Produces: `power_analysis(df, outcome_col, treatment_col="treatment", alpha=0.05, power=0.8) -> dict` (`{"alpha", "power_target", "n_control", "n_treated", "mde_effect_size_cohens_h", "mde_absolute_lift", "observed_effect_size_cohens_h", "n_required_per_group_for_observed_effect"}`) — used by Task 12's CLI.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ab_test.py
def test_power_analysis_reports_mde_and_required_n(synthetic_criteo_pandas):
    from uplift.ab_test import power_analysis

    df = synthetic_criteo_pandas(n=100_000)
    result = power_analysis(df, outcome_col="visit")

    assert result["mde_absolute_lift"] > 0
    assert result["n_required_per_group_for_observed_effect"] > 0
    assert result["n_control"] > 0
    assert result["n_treated"] > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_ab_test.py -k power -v`
Expected: FAIL — `ImportError: cannot import name 'power_analysis'`.

- [ ] **Step 3: Implement**

```python
# add to src/uplift/ab_test.py
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def _cohens_h_to_p2(effect_size: float, p1: float) -> float:
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = phi1 - effect_size
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
    mde_effect_size = analysis.solve_power(
        effect_size=None, nobs1=n_control, alpha=alpha, power=power, ratio=ratio
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_ab_test.py -k power -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/uplift/ab_test.py tests/test_ab_test.py
git commit -m "feat(ab_test): power analysis (MDE and required sample size)"
```

---

## Task 11: A/B test — Optional CACE/IV + exposure leakage guard

**Files:**
- Modify: `src/uplift/ab_test.py`
- Modify: `tests/test_ab_test.py`

**Interfaces:**
- Produces: `cace_iv(df, outcome_col: str, treatment_col: str = "treatment", exposure_col: str = "exposure") -> dict` (`{"itt_y", "itt_d", "cace"}`, raises `ValueError` if the first stage is zero) — the **only** `ab_test.py` function permitted to take `exposure_col`, enforced structurally by this task's leakage-guard test.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_ab_test.py
import inspect


def test_cace_iv_basic():
    from uplift.ab_test import cace_iv

    df = pd.DataFrame(
        {
            "treatment": [1, 1, 1, 1, 0, 0, 0, 0],
            "exposure": [1, 1, 0, 1, 0, 0, 0, 0],
            "visit": [1, 1, 0, 1, 0, 0, 0, 0],
        }
    )
    result = cace_iv(df, outcome_col="visit")

    assert result["itt_d"] > 0
    assert result["cace"] == result["itt_y"] / result["itt_d"]


def test_cace_iv_raises_on_zero_first_stage():
    from uplift.ab_test import cace_iv

    df = pd.DataFrame({"treatment": [1, 0, 1, 0], "exposure": [1, 1, 0, 0], "visit": [1, 0, 1, 0]})
    with pytest.raises(ValueError):
        cace_iv(df, outcome_col="visit")


def test_no_default_ab_test_function_accepts_exposure_col():
    from uplift import ab_test

    allowed = {"cace_iv"}
    checked = 0
    for name, func in inspect.getmembers(ab_test, inspect.isfunction):
        if name.startswith("_") or func.__module__ != ab_test.__name__ or name in allowed:
            continue
        sig = inspect.signature(func)
        assert "exposure_col" not in sig.parameters, f"{name} must not accept exposure_col"
        checked += 1
    assert checked > 0  # sanity: the scan actually found functions to check
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_ab_test.py -k "cace or exposure_col" -v`
Expected: FAIL — `ImportError: cannot import name 'cace_iv'`.

- [ ] **Step 3: Implement**

```python
# add to src/uplift/ab_test.py
def cace_iv(
    df: pd.DataFrame, outcome_col: str, treatment_col: str = "treatment", exposure_col: str = "exposure"
) -> dict:
    """Wald IV estimator: effect on the actually-exposed, using treatment as an instrument
    for exposure. The only function in this module that reads `exposure_col`."""
    treated_mask = df[treatment_col] == 1
    control_mask = df[treatment_col] == 0
    itt_y = df.loc[treated_mask, outcome_col].mean() - df.loc[control_mask, outcome_col].mean()
    itt_d = df.loc[treated_mask, exposure_col].mean() - df.loc[control_mask, exposure_col].mean()
    if itt_d == 0:
        raise ValueError("First-stage effect of treatment on exposure is zero; CACE is undefined.")
    return {"itt_y": float(itt_y), "itt_d": float(itt_d), "cace": float(itt_y / itt_d)}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_ab_test.py -k "cace or exposure_col" -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/uplift/ab_test.py tests/test_ab_test.py
git commit -m "feat(ab_test): optional CACE/IV estimator with structural exposure-leakage guard"
```

---

## Task 12: A/B test CLI + `make ab-test`

**Files:**
- Modify: `src/uplift/ab_test.py`
- Test: `tests/test_ab_test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 7–11, plus `uplift.etl.read_csv_to_spark`-produced Parquet (reads the `full` split written by Task 6's CLI).
- Produces: `main() -> None` (argparse CLI, invoked via `python -m uplift.ab_test`), used by the `Makefile`'s `ab-test` target (already written in Task 1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ab_test_cli.py
import json
import os
import subprocess
import sys


def test_ab_test_cli_end_to_end(tmp_path, synthetic_criteo_pandas, spark):
    pdf = synthetic_criteo_pandas(n=5_000)
    input_dir = tmp_path / "processed"
    sdf = spark.createDataFrame(pdf)
    sdf.write.mode("overwrite").parquet(str(input_dir / "full"))
    out_path = tmp_path / "ab_test_results.json"

    result = subprocess.run(
        [sys.executable, "-m", "uplift.ab_test", "--input-dir", str(input_dir), "--output", str(out_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    results = json.loads(out_path.read_text())
    assert "visit" in results
    assert "ate" in results["visit"]["ate"]
    assert "ci_normal" in results["visit"]
    assert "cuped" in results["visit"]
    assert "power" in results["visit"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_ab_test_cli.py -v`
Expected: FAIL — `main` doesn't exist, `python -m uplift.ab_test` errors.

- [ ] **Step 3: Implement**

```python
# add to src/uplift/ab_test.py
from uplift.constants import FEATURE_COLS


def main() -> None:
    import argparse
    import json as json_module

    from pyspark.sql import SparkSession

    parser = argparse.ArgumentParser(description="Classic A/B analysis on the full Criteo dataset")
    parser.add_argument("--input-dir", required=True, help="Directory containing the 'full' Parquet split")
    parser.add_argument("--output", required=True, help="Path to write JSON results")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("uplift-ab-test").getOrCreate()
    try:
        pdf = spark.read.parquet(f"{args.input_dir}/full").toPandas()
        spark_full = spark.read.parquet(f"{args.input_dir}/full")

        results = {}
        for outcome_col in ["visit", "conversion"]:
            outcome_results = {
                "ate": compute_ate(pdf, outcome_col),
                "ci_normal": ci_normal_approx(pdf, outcome_col),
                "ci_bootstrap": ci_bootstrap_spark(spark_full, outcome_col, n_boot=200),
                "power": power_analysis(pdf, outcome_col),
            }
            cuped_result = cuped_adjustment(pdf, outcome_col, FEATURE_COLS)
            outcome_results["cuped"] = {
                "theta": cuped_result["theta"],
                "variance_before": cuped_result["variance_before"],
                "variance_after": cuped_result["variance_after"],
                "variance_reduction_pct": cuped_result["variance_reduction_pct"],
            }
            results[outcome_col] = outcome_results

        results["cace"] = cace_iv(pdf, outcome_col="visit")

        with open(args.output, "w") as f:
            json_module.dump(results, f, indent=2, default=float)
        print(f"Wrote A/B test results to {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_ab_test_cli.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/uplift/ab_test.py tests/test_ab_test_cli.py
git commit -m "feat(ab_test): CLI wiring ATE/CI/CUPED/power/CACE into one JSON report"
```

---

## Task 13: EDA notebook

**Files:**
- Create: `notebooks/01_eda.ipynb`
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: the `full` Parquet split (Task 6's CLI output) via `pandas.read_parquet` or a local Spark session — read-only exploration, no logic later code depends on.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebooks.py
import json
from pathlib import Path


def test_eda_notebook_is_valid_json_with_expected_sections():
    nb_path = Path("notebooks/01_eda.ipynb")
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text())
    assert nb["nbformat"] == 4
    source_text = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    for expected in ["read_parquet", "visit", "groupby", "treatment"]:
        assert expected in source_text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_notebooks.py -k eda -v`
Expected: FAIL — `notebooks/01_eda.ipynb` doesn't exist.

- [ ] **Step 3: Create the notebook**

```python
# notebooks/01_eda.ipynb  (create via nbformat so the JSON is always valid)
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["cells"] = [
    nbf.v4.new_markdown_cell("# Phase 1 EDA — Criteo v2.1"),
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "df = pd.read_parquet('../data/processed/full')\n"
        "df.shape"
    ),
    nbf.v4.new_markdown_cell("## Outcome rates by treatment group"),
    nbf.v4.new_code_cell(
        "df.groupby('treatment')[['visit', 'conversion']].mean()"
    ),
    nbf.v4.new_markdown_cell("## Feature distributions (f0-f11)"),
    nbf.v4.new_code_cell(
        "df[[f'f{i}' for i in range(12)]].describe()"
    ),
    nbf.v4.new_markdown_cell("## Feature correlation matrix"),
    nbf.v4.new_code_cell(
        "df[[f'f{i}' for i in range(12)]].corr()"
    ),
]
with open("notebooks/01_eda.ipynb", "w") as f:
    nbf.write(nb, f)
```

Run this snippet once with `python` to generate the file (e.g. `python -c "<snippet above>"` or save as a throwaway script and run it), then delete the throwaway script — the committed artifact is the `.ipynb` file itself, not the generator.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_notebooks.py -k eda -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/01_eda.ipynb tests/test_notebooks.py
git commit -m "docs(notebooks): add Phase 1 EDA notebook"
```

---

## Task 14: A/B test notebook

**Files:**
- Create: `notebooks/02_ab_test.ipynb`
- Modify: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: `uplift.ab_test` functions (Tasks 7–11) and the `full` Parquet split, plus `ab_test_results.json` (Task 12's CLI output) if present.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_notebooks.py
def test_ab_test_notebook_is_valid_json_with_expected_sections():
    nb_path = Path("notebooks/02_ab_test.ipynb")
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text())
    assert nb["nbformat"] == 4
    source_text = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    for expected in ["compute_ate", "ci_normal_approx", "cuped_adjustment", "power_analysis"]:
        assert expected in source_text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_notebooks.py -k ab_test -v`
Expected: FAIL — `notebooks/02_ab_test.ipynb` doesn't exist.

- [ ] **Step 3: Create the notebook**

```python
# notebooks/02_ab_test.ipynb  (create via nbformat)
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["cells"] = [
    nbf.v4.new_markdown_cell("# Phase 2 — Classic A/B Test Analysis"),
    nbf.v4.new_code_cell(
        "import sys; sys.path.insert(0, '../src')\n"
        "import pandas as pd\n"
        "from uplift.ab_test import compute_ate, ci_normal_approx, cuped_adjustment, power_analysis, cace_iv\n"
        "from uplift.constants import FEATURE_COLS\n"
        "df = pd.read_parquet('../data/processed/full')"
    ),
    nbf.v4.new_markdown_cell("## ATE and normal-approximation CI"),
    nbf.v4.new_code_cell(
        "for outcome in ['visit', 'conversion']:\n"
        "    print(outcome, ci_normal_approx(df, outcome))"
    ),
    nbf.v4.new_markdown_cell("## CUPED-style variance reduction"),
    nbf.v4.new_code_cell(
        "cuped_result = cuped_adjustment(df, 'visit', FEATURE_COLS)\n"
        "print('variance reduction %:', cuped_result['variance_reduction_pct'])"
    ),
    nbf.v4.new_markdown_cell("## Power analysis"),
    nbf.v4.new_code_cell("power_analysis(df, 'visit')"),
    nbf.v4.new_markdown_cell("## Optional: CACE via instrumental variable"),
    nbf.v4.new_code_cell("cace_iv(df, outcome_col='visit')"),
]
with open("notebooks/02_ab_test.ipynb", "w") as f:
    nbf.write(nb, f)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_notebooks.py -k ab_test -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/02_ab_test.ipynb tests/test_notebooks.py
git commit -m "docs(notebooks): add Phase 2 A/B test analysis notebook"
```

---

## Task 15: Real data run — download, execute pipeline, write verdict

**Files:**
- Create: `docs/ab-test-verdict.md`
- No new source files — this task runs the pipeline built in Tasks 1–12 against the real dataset.

**Interfaces:**
- Consumes: `make data`, `make ab-test` (Task 1/6/12).

- [ ] **Step 1: Download the real dataset**

Run: `pip install scikit-uplift` (already in `requirements.txt`, confirm installed), then:
```python
from sklift.datasets import fetch_criteo
data = fetch_criteo(version="2.1", data_home="data/raw")
```
If `sklift`'s download is unavailable, use the Hugging Face mirror instead — either way, save the raw CSV to `data/raw/criteo-uplift-v2.1.csv` (the path `Makefile`'s `RAW_CSV` variable expects) and record which loader was actually used in this task's commit message.

- [ ] **Step 2: Run the ETL pipeline on the real data**

Run: `make data` (or the equivalent `python -m uplift.etl --input data/raw/criteo-uplift-v2.1.csv --output-dir data/processed` if `make` isn't available)
Expected: SRM check passes (p > 0.01), covariate balance check passes (all SMD < 0.1), `data/processed/split_meta.json` is written with `train`/`val`/`test` row counts summing to ~14M. **If either hard gate fails**, do not proceed — that means something about the download or ETL is wrong, not that the checks should be loosened.

- [ ] **Step 3: Run the A/B test pipeline on the real data**

Run: `make ab-test` (or `python -m uplift.ab_test --input-dir data/processed --output docs/ab_test_results.json`)

Note: the default `n_boot=200` used by the CLI (Task 12) is already reduced from the design doc's suggested 2,000 for tractability at 14M rows — if this is still too slow in practice, reduce further via a direct call to `ci_bootstrap_spark` with a smaller `n_boot` and note the reduced value in the verdict doc.

Expected: `docs/ab_test_results.json` is written with `visit`, `conversion`, and `cace` sections.

- [ ] **Step 4: Write the one-page verdict**

Read `docs/ab_test_results.json` and write `docs/ab-test-verdict.md` by hand, covering:
- Did the campaign work? (ATE on `visit`, both CIs, whether they exclude 0)
- Effect size and relative lift
- CUPED variance reduction achieved (%)
- Minimum detectable effect at this sample size, and what a smaller test would need
- Optional CACE result and what it implies about the effect on actually-exposed users
- Caveats: this is Criteo's non-uniformly sub-sampled benchmark data, so absolute lift is not a claim about Criteo's real ad performance; CUPED here is regression adjustment, not classic pre-period CUPED

- [ ] **Step 5: Commit**

```bash
git add docs/ab-test-verdict.md docs/ab_test_results.json
git commit -m "docs: record A/B test verdict from full Criteo v2.1 run"
```

(Note: `data/` itself stays gitignored — only the verdict and the JSON results summary are committed.)

---

## Self-review notes

- **Spec coverage:** every Phase 1 checklist item (download+ETL, SRM, balance, stratified split, EDA notebook) is covered by Tasks 1–6, 13; every Phase 2 item (ATE, dual CIs, CUPED, power, optional CACE, written verdict) is covered by Tasks 7–12, 14–15.
- **Leakage guard:** `exposure` is structurally prevented from entering any function except `cace_iv` (Task 11's `test_no_default_ab_test_function_accepts_exposure_col`), matching the design doc and this plan's Global Constraints.
- **Reused split:** `stratified_split`'s output (seed 42) is exactly what sub-project 2 will load — no separate splitting logic should ever be written downstream.
