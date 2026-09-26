import numpy as np
import pandas as pd
import pytest

from streaming.ab_stream import WindowAccumulator, consistency_check, run_stream_job
from streaming.events import build_events


def test_window_accumulator_running_ate_and_total():
    acc = WindowAccumulator(window_seconds=60)
    acc.update(pd.DataFrame({"exposure_time": [1.0, 2.0, 3.0, 4.0, 65.0, 66.0],
                             "treatment": [1, 1, 0, 0, 1, 0],
                             "visit": [1, 0, 0, 0, 1, 0]}))
    running = acc.running_ate()
    assert list(running["window_start"]) == [0, 60]
    assert running["n_treated"].tolist() == [2, 3] and running["n_control"].tolist() == [2, 3]
    assert running["ate"].iloc[0] == pytest.approx(0.5 - 0.0)
    assert acc.total_ate()["ate"] == pytest.approx(2 / 3 - 0.0)


def test_consistency_check_uses_relative_tolerance():
    assert consistency_check(0.0101, 0.0100, tol=0.02)
    assert not consistency_check(0.0120, 0.0100, tol=0.01)


def test_stream_join_matches_batch_ate(spark, uplift_frame, tmp_path):
    df = uplift_frame(1500, seed=12)
    df["conversion"] = 0
    exposures, outcomes = build_events(df, seed=2, eps=20, lag_range=(0.5, 3.0))
    exposures.to_parquet(tmp_path / "exposures.parquet")
    outcomes.to_parquet(tmp_path / "outcomes.parquet")
    (tmp_path / "exp").mkdir(); (tmp_path / "out").mkdir()
    exposures.to_parquet(tmp_path / "exp" / "part-0.parquet")
    outcomes.to_parquet(tmp_path / "out" / "part-0.parquet")

    acc = run_stream_job(spark, str(tmp_path / "exp"), str(tmp_path / "out"), str(tmp_path / "ckpt"))

    batch = df.groupby("treatment")["visit"].mean()
    batch_ate = batch[1] - batch[0]
    assert acc.n_rows == len(df)
    assert consistency_check(acc.total_ate()["ate"], batch_ate, tol=0.01)
    assert len(acc.running_ate()) >= 2
