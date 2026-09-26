import numpy as np
import pandas as pd
import pytest

from streaming.events import build_events, merged_events
from streaming.producer import ListSink
from streaming.scorer import benchmark_scoring, run_scorer, score_events, threshold_for_fraction


class LinearModel:
    def predict_uplift(self, X):
        return X["f0"].to_numpy()


@pytest.fixture
def exposure_events(uplift_frame):
    exposures, outcomes = build_events(uplift_frame(300, seed=9), seed=1)
    return [e for e in merged_events(exposures, outcomes) if e["topic"] == "exposures"]


def test_threshold_for_fraction_treats_that_share_of_scores():
    scores = np.random.default_rng(0).normal(size=10_000)
    threshold = threshold_for_fraction(scores, 0.3)
    assert (scores >= threshold).mean() == pytest.approx(0.3, abs=0.01)


def test_score_events_returns_one_decision_per_event_and_uses_threshold(exposure_events):
    decisions = score_events(LinearModel(), exposure_events, threshold=0.0)
    assert len(decisions) == len(exposure_events)
    for event, decision in zip(exposure_events, decisions):
        assert decision["user_id"] == event["value"]["user_id"]
        assert decision["treat"] == (event["value"]["f0"] >= 0.0)
        assert decision["uplift"] == pytest.approx(event["value"]["f0"])


def test_score_events_ignores_columns_outside_the_feature_set(exposure_events):
    seen = {}

    class Spy:
        def predict_uplift(self, X):
            seen["cols"] = list(X.columns)
            return np.zeros(len(X))

    poisoned = [{**e, "value": {**e["value"], "visit": 1, "exposure": 1}} for e in exposure_events[:5]]
    score_events(Spy(), poisoned, threshold=0.0)
    assert seen["cols"] == [f"f{i}" for i in range(12)]


class FakeConsumer:
    def __init__(self, events, batch_size):
        self.batches = [events[i : i + batch_size] for i in range(0, len(events), batch_size)]

    def poll_batch(self, n, timeout):
        return self.batches.pop(0) if self.batches else []


def test_run_scorer_micro_batches_and_publishes_all_decisions(exposure_events):
    sink = ListSink()
    processed = run_scorer(FakeConsumer(exposure_events, 100), sink, LinearModel(), 0.0, batch_size=100, max_idle_polls=1)
    assert processed == len(exposure_events)
    assert {topic for topic, _, _ in sink.messages} == {"decisions"}
    assert len(sink.messages) == len(exposure_events)
    assert all(key == str(value["user_id"]) for _, key, value in sink.messages)


def test_benchmark_scoring_reports_throughput_and_latency(uplift_frame):
    stats = benchmark_scoring(LinearModel(), uplift_frame(2000, seed=3), batch_size=500, n_batches=8)
    assert stats["events_per_sec"] > 0
    assert 0 <= stats["p95_batch_ms"] and stats["p95_event_ms"] >= stats["p95_batch_ms"] / 500


def _drifting_events(uplift_frame, n=1200, shift=4.0):
    from streaming.events import inject_drift

    exposures, outcomes = build_events(uplift_frame(n, seed=21), seed=1)
    exposures = inject_drift(exposures, "f0", shift=shift, after_fraction=0.5)
    return [e for e in merged_events(exposures, outcomes) if e["topic"] == "exposures"]


def test_run_scorer_publishes_drift_alert_within_two_windows_of_the_shift(uplift_frame):
    from streaming.drift import DriftMonitor
    from streaming.scorer import DRIFT_TOPIC

    events = _drifting_events(uplift_frame)
    reference = uplift_frame(5000, seed=22)[[f"f{i}" for i in range(12)]]
    monitor = DriftMonitor(reference, list(reference.columns), window_size=200)
    sink = ListSink()
    run_scorer(FakeConsumer(events, 100), sink, LinearModel(), 0.0, batch_size=100, max_idle_polls=1, drift_monitor=monitor)
    alerts = [v for topic, _, v in sink.messages if topic == DRIFT_TOPIC]
    assert alerts, "shift was never flagged"
    assert all(a["features"] == ["f0"] for a in alerts)
    assert all(a["n_processed"] > 600 for a in alerts)  # nothing flagged before the shift starts
    assert alerts[0]["n_processed"] <= 600 + 2 * 200  # flagged within two windows


def test_run_scorer_raises_no_drift_alert_on_a_stationary_stream(uplift_frame):
    from streaming.drift import DriftMonitor
    from streaming.scorer import DRIFT_TOPIC

    events = _drifting_events(uplift_frame, shift=0.0)
    reference = uplift_frame(5000, seed=22)[[f"f{i}" for i in range(12)]]
    monitor = DriftMonitor(reference, list(reference.columns), window_size=200)
    sink = ListSink()
    run_scorer(FakeConsumer(events, 100), sink, LinearModel(), 0.0, batch_size=100, max_idle_polls=1, drift_monitor=monitor)
    assert not [1 for topic, _, _ in sink.messages if topic == DRIFT_TOPIC]
