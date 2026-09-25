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
