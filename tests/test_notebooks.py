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


def _uplift_results_stub():
    curve = {"x": [0.0, 50.0, 100.0], "y": [0.0, 4.0, 5.0]}
    metrics = {"qini_auc": 0.1, "uplift_at_10": 0.1, "uplift_at_20": 0.1, "uplift_at_30": 0.1,
               "auc": 0.1, "ci_low": 0.05, "ci_high": 0.15}
    return {
        "best_model": "s_learner",
        "test_metrics": {"random": metrics, "s_learner": metrics},
        "qini_curves": {"random": curve, "s_learner": curve},
        "decile_table": [
            {"decile": d, "n": 10, "treated_rate": 0.1, "control_rate": 0.05, "uplift": 0.05}
            for d in range(1, 11)
        ],
        "segment_table": [
            {"segment": s, "n": 10, "share": 0.25, "observed_uplift": 0.01}
            for s in ("persuadables", "sure_things", "lost_causes", "sleeping_dogs")
        ],
    }


def test_uplift_notebook_code_runs_against_results_json(tmp_path, monkeypatch):
    nb_path = Path("notebooks/03_uplift.ipynb").resolve()
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text())
    assert nb["nbformat"] == 4
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")

    (tmp_path / "docs").mkdir()
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "docs" / "uplift_results.json").write_text(json.dumps(_uplift_results_stub()))
    monkeypatch.chdir(tmp_path / "notebooks")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace: dict = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)


def test_policy_notebook_code_runs_against_policy_json(tmp_path, monkeypatch):
    nb_path = Path("notebooks/04_policy.ipynb").resolve()
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text())
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    curve = [
        {"fraction": f, "n_targeted": int(f * 100), "uplift": 0.05, "incremental_outcomes": f * 5,
         "cost": f * 50, "revenue": f * 100, "profit": f * 50}
        for f in (0.5, 1.0)
    ]
    stub = {
        "assumptions": {"cost_per_impression": 0.5, "value_per_outcome": 20.0, "currency": "INR", "note": "x"},
        "strategies": {s: {"curve": curve} for s in ("uplift", "response_model", "random")},
        "operating_point": {"strategy": "uplift", "fraction": 0.5, "profit": 25.0},
        "segment_savings": [{"segment": "do_not_target_total", "n": 1, "net_saving": 1.0}],
    }
    (tmp_path / "docs").mkdir()
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "docs" / "policy_results.json").write_text(json.dumps(stub))
    monkeypatch.chdir(tmp_path / "notebooks")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace: dict = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
