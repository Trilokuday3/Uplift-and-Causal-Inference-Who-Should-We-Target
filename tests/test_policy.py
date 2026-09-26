import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from uplift.policy import budget_curve, business_impact, operating_point, segment_savings


@pytest.fixture
def scored(uplift_frame):
    df = uplift_frame(6000, seed=5)
    rng = np.random.default_rng(1)
    return {
        "y": df["visit"].to_numpy(),
        "t": df["treatment"].to_numpy(),
        "oracle": df["f0"].to_numpy(),
        "noise": rng.random(len(df)),
    }


FRACTIONS = [0.1, 0.3, 0.5, 1.0]


def test_budget_curve_oracle_beats_random_at_partial_budgets_and_ties_at_full(scored):
    y, t = scored["y"], scored["t"]
    oracle = budget_curve(y, t, scored["oracle"], FRACTIONS).set_index("fraction")
    noise = budget_curve(y, t, scored["noise"], FRACTIONS).set_index("fraction")
    assert oracle.loc[0.3, "incremental_outcomes"] > noise.loc[0.3, "incremental_outcomes"]
    assert oracle.loc[1.0, "incremental_outcomes"] == pytest.approx(noise.loc[1.0, "incremental_outcomes"])
    assert list(oracle["n_targeted"]) == [int(round(f * len(y))) for f in FRACTIONS]


def test_budget_curve_is_independent_of_row_order(scored):
    y, t, s = scored["y"], scored["t"], scored["oracle"]
    order = np.argsort(t, kind="stable")
    a = budget_curve(y, t, s, FRACTIONS)
    b = budget_curve(y[order], t[order], s[order], FRACTIONS)
    assert np.allclose(a["incremental_outcomes"], b["incremental_outcomes"])


def test_business_impact_arithmetic_and_operating_point():
    curve = pd.DataFrame(
        {
            "fraction": [0.5, 1.0],
            "n_targeted": [100, 200],
            "uplift": [0.05, 0.01],
            "incremental_outcomes": [5.0, 2.0],
        }
    )
    impact = business_impact(curve, cost_per_impression=0.5, value_per_outcome=20.0)
    first = impact.iloc[0]
    assert (first["cost"], first["revenue"], first["profit"]) == (50.0, 100.0, 50.0)
    assert impact.iloc[1]["profit"] == pytest.approx(2.0 * 20 - 200 * 0.5)
    assert operating_point(impact)["fraction"] == 0.5


def test_segment_savings_identities(scored):
    table = segment_savings(
        scored["y"], scored["t"], scored["oracle"], scored["noise"], cost_per_impression=0.5, value_per_outcome=20.0
    )
    assert set(table["segment"]) == {"sure_things", "sleeping_dogs", "do_not_target_total"}
    row = table.set_index("segment").loc["sure_things"]
    assert row["impression_cost_saved"] == pytest.approx(row["n"] * 0.5)
    assert row["net_saving"] == pytest.approx(row["impression_cost_saved"] - row["value_forgone"])
    total = table.set_index("segment").loc["do_not_target_total"]
    parts = table[table["segment"] != "do_not_target_total"]
    assert total["n"] == parts["n"].sum()
    assert total["net_saving"] == pytest.approx(parts["net_saving"].sum())


def test_policy_cli_end_to_end(tmp_path, scored):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    n = len(scored["y"])
    pd.DataFrame(
        {
            "treatment": scored["t"],
            "outcome": scored["y"],
            "random": scored["noise"],
            "response_model": scored["noise"][::-1],
            "s_learner": scored["oracle"],
        }
    ).to_parquet(artifacts / "test_scores.parquet")
    (artifacts / "model_meta.json").write_text(json.dumps({"model_name": "s_learner", "outcome": "visit"}))
    out = tmp_path / "policy.json"
    result = subprocess.run(
        [sys.executable, "-m", "uplift.policy", "--artifacts-dir", str(artifacts), "--output", str(out),
         "--cost-per-impression", "0.5", "--value-per-outcome", "20"],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    res = json.loads(out.read_text())
    assert res["assumptions"]["cost_per_impression"] == 0.5
    assert set(res["strategies"]) == {"uplift", "response_model", "random"}
    assert res["operating_point"]["strategy"] == "uplift"
    assert res["strategies"]["uplift"]["curve"][0]["fraction"] == pytest.approx(0.05)
    assert {r["segment"] for r in res["segment_savings"]} >= {"do_not_target_total"}
    assert n == len(scored["y"])


def test_interpolate_curve_is_linear_between_points_and_anchored_at_origin():
    from uplift.policy import interpolate_curve

    curve = [
        {"fraction": 0.5, "incremental_outcomes": 10.0, "profit": 4.0},
        {"fraction": 1.0, "incremental_outcomes": 16.0, "profit": 2.0},
    ]
    assert interpolate_curve(curve, 0.5)["incremental_outcomes"] == pytest.approx(10.0)
    assert interpolate_curve(curve, 0.75)["incremental_outcomes"] == pytest.approx(13.0)
    assert interpolate_curve(curve, 0.25)["profit"] == pytest.approx(2.0)  # halfway from origin (0,0)
    with pytest.raises(ValueError):
        interpolate_curve(curve, 1.5)


def test_operating_point_is_chosen_on_validation_and_profit_reported_on_test(tmp_path, uplift_frame):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    def frame(seed, flip):
        df = uplift_frame(6000, seed=seed)
        rng = np.random.default_rng(seed)
        signal = df["f0"].to_numpy()
        return pd.DataFrame({
            "treatment": df["treatment"], "outcome": df["visit"], "random": rng.random(len(df)),
            "response_model": rng.random(len(df)), "s_learner": -signal if flip else signal,
        })

    frame(1, flip=False).to_parquet(artifacts / "val_scores.parquet")
    frame(2, flip=False).to_parquet(artifacts / "test_scores.parquet")
    (artifacts / "model_meta.json").write_text(json.dumps({"model_name": "s_learner", "outcome": "visit"}))
    out = tmp_path / "policy.json"
    result = subprocess.run(
        [sys.executable, "-m", "uplift.policy", "--artifacts-dir", str(artifacts), "--output", str(out)],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    res = json.loads(out.read_text())
    op = res["operating_point"]
    assert op["selected_on"] == "validation" and op["reported_on"] == "test"
    test_curve = pd.DataFrame(res["strategies"]["uplift"]["curve"]).set_index("fraction")
    assert op["profit"] == pytest.approx(test_curve.loc[op["fraction"], "profit"])
    assert "in_sample_best_on_test" in res  # kept only as a clearly labelled optimistic reference
    assert op["profit"] <= res["in_sample_best_on_test"]["uplift"]["profit"] + 1e-9
