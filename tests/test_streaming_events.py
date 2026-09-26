import numpy as np
import pytest

from streaming.events import build_events, inject_drift, merged_events
from streaming.producer import ListSink, replay


@pytest.fixture
def frame(uplift_frame):
    df = uplift_frame(400, seed=8)
    df["conversion"] = 0
    return df


def test_exposure_events_never_carry_outcomes_or_exposure_flag(frame):
    exposures, _ = build_events(frame, seed=1)
    for leaky in ("visit", "conversion", "exposure"):
        assert leaky not in exposures.columns
    assert {"user_id", "event_time", "treatment", "f0", "f11"} <= set(exposures.columns)


def test_exposures_are_shuffled_unique_and_paced_by_eps(frame):
    exposures, _ = build_events(frame, seed=1, eps=100)
    assert exposures["user_id"].is_unique and len(exposures) == len(frame)
    assert np.allclose(np.diff(exposures["event_time"]), 1 / 100, atol=1e-5)
    assert not exposures["user_id"].is_monotonic_increasing


def test_outcomes_arrive_after_their_exposure_within_lag_range(frame):
    exposures, outcomes = build_events(frame, seed=1, lag_range=(2.0, 5.0))
    joined = outcomes.merge(exposures[["user_id", "event_time"]], on="user_id", suffixes=("_out", "_exp"))
    lag = joined["event_time_out"] - joined["event_time_exp"]
    assert len(outcomes) == len(frame)
    assert lag.min() >= 2.0 and lag.max() <= 5.0
    assert set(outcomes.columns) == {"user_id", "event_time", "visit", "conversion"}


def test_build_events_is_deterministic_per_seed(frame):
    a, b = build_events(frame, seed=3)[0], build_events(frame, seed=3)[0]
    assert a.equals(b)
    assert not a["user_id"].equals(build_events(frame, seed=4)[0]["user_id"])


def test_merged_events_are_time_ordered_and_outcome_never_precedes_exposure(frame):
    exposures, outcomes = build_events(frame, seed=1, lag_range=(1.0, 3.0))
    events = list(merged_events(exposures, outcomes))
    times = [e["value"]["event_time"] for e in events]
    assert times == sorted(times)
    assert {e["topic"] for e in events} == {"exposures", "outcomes"}
    seen = set()
    for e in events:
        if e["topic"] == "exposures":
            seen.add(e["key"])
        else:
            assert e["key"] in seen


def test_inject_drift_shifts_only_late_events(frame):
    exposures, _ = build_events(frame, seed=1)
    drifted = inject_drift(exposures, "f0", shift=5.0, after_fraction=0.5)
    half = len(exposures) // 2
    assert np.allclose(drifted["f0"].iloc[:half], exposures["f0"].iloc[:half])
    assert np.allclose(drifted["f0"].iloc[half:], exposures["f0"].iloc[half:] + 5.0)


def test_replay_paces_events_on_simulated_clock_and_delivers_all(frame):
    exposures, outcomes = build_events(frame.head(50), seed=1, eps=10, lag_range=(0.5, 1.0))
    events = list(merged_events(exposures, outcomes))
    sink, clock = ListSink(), {"t": 0.0}

    def fake_sleep(seconds):
        clock["t"] += seconds

    replay(events, sink, speed=1.0, sleep=fake_sleep, now=lambda: clock["t"])
    assert len(sink.messages) == len(events)
    assert clock["t"] == pytest.approx(events[-1]["value"]["event_time"] - events[0]["value"]["event_time"], abs=1e-6)


def test_write_event_files_emits_parquet_dirs_and_batch_ate_meta(frame, tmp_path):
    import json

    import pandas as pd

    from streaming.producer import write_event_files

    exposures, outcomes = build_events(frame, seed=1, eps=50)
    write_event_files(exposures, outcomes, tmp_path)
    assert len(pd.read_parquet(tmp_path / "exposures")) == len(frame)
    assert len(pd.read_parquet(tmp_path / "outcomes")) == len(frame)
    meta = json.loads((tmp_path / "meta.json").read_text())
    rates = frame.groupby("treatment")["visit"].mean()
    assert meta["batch_ate"] == pytest.approx(rates[1] - rates[0])
    assert meta["n_events"] == len(frame)
