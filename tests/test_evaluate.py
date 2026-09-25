import numpy as np
import pytest

from uplift.evaluate import (
    bootstrap_qini_ci,
    decile_table,
    qini_auc,
    qini_curve,
    segment_table,
    uplift_at_k,
)


@pytest.fixture
def scored(uplift_frame):
    df = uplift_frame(4000, seed=3)
    rng = np.random.default_rng(0)
    return {
        "y": df["visit"].to_numpy(),
        "t": df["treatment"].to_numpy(),
        "oracle": df["f0"].to_numpy(),
        "reversed": -df["f0"].to_numpy(),
        "noise": rng.random(len(df)),
    }


def test_qini_auc_ranks_oracle_above_noise_above_reversed(scored):
    y, t = scored["y"], scored["t"]
    assert qini_auc(y, t, scored["oracle"]) > qini_auc(y, t, scored["noise"])
    assert qini_auc(y, t, scored["noise"]) > qini_auc(y, t, scored["reversed"])


def test_qini_curve_returns_matching_arrays(scored):
    x, y_curve = qini_curve(scored["y"], scored["t"], scored["oracle"])
    assert len(x) == len(y_curve) > 2


def test_uplift_at_k_oracle_beats_reversed(scored):
    y, t = scored["y"], scored["t"]
    assert uplift_at_k(y, t, scored["oracle"], 0.3) > uplift_at_k(y, t, scored["reversed"], 0.3)


def test_decile_table_shape_and_ordering(scored):
    table = decile_table(scored["y"], scored["t"], scored["oracle"])
    assert len(table) == 10
    assert table["n"].sum() == len(scored["y"])
    assert table["uplift"].iloc[0] > table["uplift"].iloc[-1]


def test_decile_table_survives_constant_scores(scored):
    table = decile_table(scored["y"], scored["t"], np.zeros(len(scored["y"])))
    assert len(table) == 10 and table["n"].sum() == len(scored["y"])


def test_metrics_reject_single_arm_input():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        qini_auc(y, np.ones(4, dtype=int), np.arange(4.0))


def test_segment_table_has_four_segments_and_sane_uplift(scored):
    response = scored["noise"]
    table = segment_table(scored["y"], scored["t"], scored["oracle"], response)
    assert set(table["segment"]) == {"persuadables", "sure_things", "lost_causes", "sleeping_dogs"}
    assert table["n"].sum() == len(scored["y"])
    by = table.set_index("segment")["observed_uplift"]
    assert by["persuadables"] > 0.1
    assert by["sleeping_dogs"] < by["persuadables"]


def test_bootstrap_ci_separates_oracle_from_noise(scored):
    result = bootstrap_qini_ci(
        scored["y"], scored["t"], {"oracle": scored["oracle"], "noise": scored["noise"]}, n_boot=60, seed=1
    )
    assert result["oracle"]["ci_low"] > result["noise"]["ci_high"]
    assert result["oracle"]["ci_low"] <= result["oracle"]["auc"] <= result["oracle"]["ci_high"]


def test_bootstrap_ci_is_deterministic(scored):
    args = (scored["y"], scored["t"], {"m": scored["oracle"]})
    assert bootstrap_qini_ci(*args, n_boot=20, seed=5) == bootstrap_qini_ci(*args, n_boot=20, seed=5)


def test_bootstrap_ci_skips_resamples_missing_an_arm():
    y = np.array([1, 0, 1, 0, 1, 0])
    t = np.array([1, 0, 0, 0, 0, 0])
    scores = {"m": np.arange(6.0)}
    result = bootstrap_qini_ci(y, t, scores, n_boot=30, seed=0)
    assert np.isfinite(result["m"]["auc"])


def test_metrics_reject_constant_outcome():
    with pytest.raises(ValueError):
        qini_auc(np.zeros(6, dtype=int), np.array([1, 0, 1, 0, 1, 0]), np.arange(6.0))
