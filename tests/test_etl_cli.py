import json
import os
import subprocess
import sys


def test_etl_cli_end_to_end(tmp_path, synthetic_criteo_pandas):
    pdf = synthetic_criteo_pandas(n=5_000)
    csv_path = tmp_path / "raw.csv"
    pdf.to_csv(csv_path, index=False)
    out_dir = tmp_path / "processed"

    result = subprocess.run(
        [sys.executable, "-m", "uplift.etl", "--input", str(csv_path), "--output-dir", str(out_dir)],
        cwd=None,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "split_meta.json").exists()
    meta = json.loads((out_dir / "split_meta.json").read_text())
    assert set(meta["splits"].keys()) == {"train", "val", "test"}


def test_etl_cli_srm_failure_publishes_nothing(tmp_path, synthetic_criteo_pandas):
    """A failed SRM gate must not leave a `full` dataset behind for ab_test.py to
    unknowingly analyze - the gate has to run before anything is published, not after."""
    pdf = synthetic_criteo_pandas(n=5_000, treatment_rate=0.5)  # badly broken vs the 0.85 default
    csv_path = tmp_path / "raw.csv"
    pdf.to_csv(csv_path, index=False)
    out_dir = tmp_path / "processed"

    result = subprocess.run(
        [sys.executable, "-m", "uplift.etl", "--input", str(csv_path), "--output-dir", str(out_dir)],
        cwd=None,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode != 0
    assert not (out_dir / "full").exists()
    assert not (out_dir / "split_meta.json").exists()
