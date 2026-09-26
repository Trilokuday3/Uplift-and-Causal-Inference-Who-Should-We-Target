import json
import os
import subprocess
import sys

from mlflow.tracking import MlflowClient


def _write_splits(tmp_path, uplift_frame):
    input_dir = tmp_path / "processed"
    for i, name in enumerate(["train", "val", "test"]):
        split_dir = input_dir / name
        split_dir.mkdir(parents=True)
        uplift_frame(2500, seed=10 + i).to_parquet(split_dir / "part-0.parquet")
    return input_dir


def test_train_cli_end_to_end(tmp_path, uplift_frame):
    input_dir = _write_splits(tmp_path, uplift_frame)
    out_path = tmp_path / "uplift_results.json"
    db_uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()

    result = subprocess.run(
        [
            sys.executable, "-m", "uplift.train",
            "--input-dir", str(input_dir),
            "--output", str(out_path),
            "--models", "random,response_model,s_learner,t_learner",
            "--sample-size", "1500",
            "--n-trials", "2",
            "--n-boot", "20",
            "--mlflow-uri", db_uri,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src", "MLFLOW_DISABLE_AGENT_HINT": "1"},
    )

    assert result.returncode == 0, result.stderr
    results = json.loads(out_path.read_text())
    assert set(results["test_metrics"]) == {"random", "response_model", "s_learner", "t_learner"}
    best = results["best_model"]
    assert best in {"s_learner", "t_learner"}
    assert results["test_metrics"][best]["qini_auc"] > results["test_metrics"]["random"]["qini_auc"]
    assert results["test_metrics"][best]["ci_low"] <= results["test_metrics"][best]["ci_high"]
    assert len(results["decile_table"]) == 10
    assert {row["segment"] for row in results["segment_table"]} == {
        "persuadables", "sure_things", "lost_causes", "sleeping_dogs"
    }
    assert len(results["tuned_params"]) == 2
    assert all(len(c["x"]) == len(c["y"]) <= 200 for c in results["qini_curves"].values())

    client = MlflowClient(tracking_uri=db_uri)
    runs = client.search_runs([client.get_experiment_by_name("uplift").experiment_id])
    assert runs
    for run in runs:
        assert "qini_auc" in run.data.metrics
        assert "exposure" not in " ".join(run.data.params)


def test_train_cli_reports_paired_difference_against_response_model(tmp_path, uplift_frame):
    input_dir = _write_splits(tmp_path, uplift_frame)
    out_path = tmp_path / "uplift_results.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "uplift.train",
            "--input-dir", str(input_dir), "--output", str(out_path),
            "--models", "response_model,s_learner", "--sample-size", "1500",
            "--n-trials", "1", "--n-boot", "20",
            "--mlflow-uri", "sqlite:///" + (tmp_path / "mlflow.db").as_posix(),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src", "MLFLOW_DISABLE_AGENT_HINT": "1"},
    )
    assert result.returncode == 0, result.stderr
    results = json.loads(out_path.read_text())
    assert results["reference_model"] == "response_model"
    assert set(results["test_metrics"]["s_learner"]["diff_vs_reference"]) == {"estimate", "ci_low", "ci_high"}
    assert "untuned_val_metrics" in results


def test_train_cli_saves_model_and_test_scores(tmp_path, uplift_frame):
    import joblib
    import pandas as pd

    input_dir = _write_splits(tmp_path, uplift_frame)
    artifacts = tmp_path / "artifacts"
    result = subprocess.run(
        [
            sys.executable, "-m", "uplift.train",
            "--input-dir", str(input_dir), "--output", str(tmp_path / "r.json"),
            "--models", "random,response_model,s_learner", "--sample-size", "1500",
            "--n-trials", "1", "--n-boot", "10",
            "--mlflow-uri", "sqlite:///" + (tmp_path / "mlflow.db").as_posix(),
            "--artifacts-dir", str(artifacts),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src", "MLFLOW_DISABLE_AGENT_HINT": "1"},
    )
    assert result.returncode == 0, result.stderr
    meta = json.loads((artifacts / "model_meta.json").read_text())
    assert meta["model_name"] == "s_learner" and meta["outcome"] == "visit"
    assert "exposure" not in meta["feature_cols"]
    model = joblib.load(artifacts / "best_model.joblib")
    scores = pd.read_parquet(artifacts / "test_scores.parquet")
    assert {"treatment", "outcome", "random", "response_model", "s_learner"} <= set(scores.columns)
    assert len(model.predict_uplift(scores.iloc[:5].assign(**{f"f{i}": 0.0 for i in range(12)})[[f"f{i}" for i in range(12)]])) == 5
