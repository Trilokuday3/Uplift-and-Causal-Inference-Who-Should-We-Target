import json
from pathlib import Path

import pandas as pd
import pytest

from app.data import ab_verdict, decile_frame, load_json, policy_at_budget, qini_frame

DASHBOARD = str(Path(__file__).resolve().parent.parent / "app" / "dashboard.py")
UPLIFT = {
    "best_model": "s_learner",
    "qini_curves": {
        "s_learner": {"x": [0, 50, 100], "y": [0, 4, 5]},
        "random": {"x": [0, 60, 100], "y": [0, 3, 5]},
    },
    "decile_table": [{"decile": d, "uplift": 0.1 - d * 0.01} for d in range(1, 11)],
}
AB = {"visit": {"ate": {"ate": 0.0103, "relative_lift": 0.27, "n_treated": 100, "n_control": 20},
                "ci_normal": {"ci_low": 0.0100, "ci_high": 0.0106}}}
POLICY = {"strategies": {"uplift": {"curve": [{"fraction": 0.5, "incremental_outcomes": 10.0, "profit": 4.0},
                                              {"fraction": 1.0, "incremental_outcomes": 12.0, "profit": 1.0}]}},
          "assumptions": {"currency": "INR"}}


def test_load_json_returns_none_when_file_is_missing(tmp_path):
    assert load_json(tmp_path, "nope.json") is None
    (tmp_path / "x.json").write_text(json.dumps({"a": 1}))
    assert load_json(tmp_path, "x.json") == {"a": 1}


def test_qini_frame_puts_all_models_on_one_shared_x_axis():
    frame = qini_frame(UPLIFT)
    assert set(frame.columns) == {"s_learner", "random"}
    assert frame.index.is_monotonic_increasing and frame.notna().all().all()


def test_decile_frame_and_ab_verdict_and_policy_lookup():
    assert list(decile_frame(UPLIFT)["decile"]) == list(range(1, 11))
    verdict = ab_verdict(AB)
    assert verdict["worked"] is True and "1.03" in verdict["headline"]
    assert policy_at_budget(POLICY, 0.75)["incremental_outcomes"] == pytest.approx(11.0)


def test_ab_verdict_says_inconclusive_when_ci_includes_zero():
    ab = {"visit": {"ate": {"ate": 0.0001, "relative_lift": 0.01, "n_treated": 10, "n_control": 10},
                    "ci_normal": {"ci_low": -0.001, "ci_high": 0.001}}}
    assert ab_verdict(ab)["worked"] is False


def test_dashboard_script_renders_without_exceptions_when_results_exist(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    (tmp_path / "uplift_results.json").write_text(json.dumps(UPLIFT))
    (tmp_path / "ab_test_results.json").write_text(json.dumps(AB))
    (tmp_path / "policy_results.json").write_text(json.dumps(POLICY))
    monkeypatch.setenv("DOCS_DIR", str(tmp_path))
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not at.exception
    assert len(at.slider) == 1
    at.slider[0].set_value(0.75).run()
    assert not at.exception


def test_dashboard_script_degrades_gracefully_with_no_results(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DOCS_DIR", str(tmp_path))
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not at.exception
    assert len(at.warning) >= 1
