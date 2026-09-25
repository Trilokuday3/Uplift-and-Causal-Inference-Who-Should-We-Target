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


def test_ab_test_cli_respects_n_boot_override(tmp_path, synthetic_criteo_pandas, spark):
    pdf = synthetic_criteo_pandas(n=2_000)
    input_dir = tmp_path / "processed"
    sdf = spark.createDataFrame(pdf)
    sdf.write.mode("overwrite").parquet(str(input_dir / "full"))
    out_path = tmp_path / "ab_test_results.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "uplift.ab_test",
            "--input-dir",
            str(input_dir),
            "--output",
            str(out_path),
            "--n-boot",
            "5",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 0, result.stderr
    results = json.loads(out_path.read_text())
    assert results["visit"]["ci_bootstrap"]["n_boot"] == 5


def test_ab_test_cli_cuped_reports_narrower_adjusted_ci(tmp_path, synthetic_criteo_pandas, spark):
    """Spec success criterion 4: CUPED must report the narrower resulting CI, not just
    the variance numbers - a stakeholder reading the verdict needs the actual adjusted
    interval, not just a percentage they have to trust was applied correctly."""
    pdf = synthetic_criteo_pandas(n=20_000)
    # inject a covariate-outcome correlation shared by both arms (not just within
    # treated, as the fixture's own uplift_signal is) - CUPED needs real, both-arms
    # signal to reliably narrow the CI at this sample size; same technique as
    # test_cuped_reduces_variance in test_ab_test.py.
    pdf["visit"] = (pdf["visit"] + (pdf["f0"] > 1.5).astype(int)).clip(0, 1)
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
    results = json.loads(out_path.read_text())
    cuped = results["visit"]["cuped"]
    raw_ci = results["visit"]["ci_normal"]

    assert "adjusted_ci_low" in cuped
    assert "adjusted_ci_high" in cuped
    assert "adjusted_ate" in cuped
    raw_width = raw_ci["ci_high"] - raw_ci["ci_low"]
    adjusted_width = cuped["adjusted_ci_high"] - cuped["adjusted_ci_low"]
    assert adjusted_width < raw_width
