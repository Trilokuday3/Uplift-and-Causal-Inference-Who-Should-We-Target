import numpy as np
import pandas as pd
import pytest

from streaming.drift import DriftMonitor, detect_drift
from streaming.sequential import msprt_confidence_sequence, naive_pvalue_path


def _simulate(n, effect, seed):
    rng = np.random.default_rng(seed)
    t = rng.binomial(1, 0.5, size=n)
    y = rng.binomial(1, 0.10 + effect * t)
    return y, t


CHECKPOINTS = list(range(100, 2001, 50))


def test_peeking_inflates_false_positives_but_msprt_does_not():
    naive_hits = msprt_hits = 0
    runs = 200
    for seed in range(runs):
        y, t = _simulate(2000, effect=0.0, seed=seed)
        naive = naive_pvalue_path(y, t, CHECKPOINTS)
        cs = msprt_confidence_sequence(y, t, CHECKPOINTS, alpha=0.05, tau=0.05)
        naive_hits += (naive["p_value"] < 0.05).any()
        msprt_hits += ((cs["ci_low"] > 0) | (cs["ci_high"] < 0)).any()
    assert naive_hits / runs > 0.15
    assert msprt_hits / runs < 0.08


def test_msprt_eventually_detects_a_real_effect():
    y, t = _simulate(20000, effect=0.05, seed=1)
    cs = msprt_confidence_sequence(y, t, list(range(500, 20001, 500)), alpha=0.05, tau=0.05)
    assert (cs["ci_low"] > 0).any()
    assert (cs["ci_low"] <= cs["ate"]).all() and (cs["ate"] <= cs["ci_high"]).all()


def test_sequential_paths_have_one_row_per_checkpoint():
    y, t = _simulate(1000, 0.02, seed=2)
    assert len(naive_pvalue_path(y, t, [100, 500, 1000])) == 3
    assert len(msprt_confidence_sequence(y, t, [100, 500, 1000])) == 3


@pytest.fixture
def reference():
    rng = np.random.default_rng(0)
    return pd.DataFrame({f"f{i}": rng.normal(size=5000) for i in range(3)})


def test_detect_drift_flags_only_the_shifted_feature(reference):
    rng = np.random.default_rng(1)
    window = pd.DataFrame({f"f{i}": rng.normal(size=1000) for i in range(3)})
    assert detect_drift(reference, window, list(reference.columns)) == []
    window["f1"] += 1.0
    assert detect_drift(reference, window, list(reference.columns)) == ["f1"]


def test_drift_monitor_flags_injected_shift_within_two_windows(reference):
    rng = np.random.default_rng(2)
    monitor = DriftMonitor(reference, list(reference.columns), window_size=500)
    stream = pd.DataFrame({f"f{i}": rng.normal(size=4000) for i in range(3)})
    stream.loc[stream.index[2000:], "f0"] += 2.0  # shift starts at row 2000 = start of window 5
    flagged_windows = []
    for start in range(0, 4000, 500):
        result = monitor.update(stream.iloc[start : start + 500])
        if result:
            flagged_windows.append((start // 500, result))
    assert flagged_windows and flagged_windows[0][0] <= 5 + 1
    assert all(w >= 4 for w, _ in flagged_windows)
    assert flagged_windows[0][1] == ["f0"]


def test_drift_demo_flags_only_after_shift():
    import numpy as np
    import pandas as pd

    from streaming.drift_demo import run_demo
    from uplift.constants import FEATURE_COLS

    rng = np.random.default_rng(0)
    make = lambda n: pd.DataFrame({**{c: rng.normal(size=n) for c in FEATURE_COLS},
                                   "treatment": rng.integers(0, 2, n), "visit": rng.integers(0, 2, n),
                                   "conversion": 0, "exposure": 0})
    res = run_demo(make(5000), make(4000), "f0", 3.0, limit=4000, window=500)
    assert res["first_flagged_row"] is not None and res["first_flagged_row"] >= 1500
