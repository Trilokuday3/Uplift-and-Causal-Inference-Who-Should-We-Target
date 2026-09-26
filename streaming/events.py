from __future__ import annotations

import heapq
from typing import Iterator

import numpy as np
import pandas as pd

from uplift.constants import CONVERSION_COL, FEATURE_COLS, TREATMENT_COL, VISIT_COL

# A realistic epoch (Nov 2023) rather than 0: Spark treats event times at/before its initial
# watermark as late, which silently dropped the first event when the simulated clock started at 0.
SIM_EPOCH = 1_700_000_000.0
EXPOSURES_TOPIC = "exposures"
OUTCOMES_TOPIC = "outcomes"


def build_events(
    df: pd.DataFrame,
    seed: int = 42,
    eps: float = 5000.0,
    lag_range: tuple[float, float] = (0.5, 5.0),
    start_time: float = SIM_EPOCH,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn a (test-split) frame into a shuffled exposure stream plus delayed outcome events.

    Exposure events carry only what is known at serving time: user id, the randomized
    `treatment` assignment and the features. `visit`/`conversion` (and the post-randomization
    `exposure` flag) appear only in the later outcome events or not at all."""
    n = len(df) if limit is None else min(limit, len(df))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))[:n]
    picked = df.iloc[order].reset_index(drop=True)
    event_time = start_time + np.arange(n) / eps

    exposures = pd.DataFrame(
        {"user_id": order, "event_time": event_time, TREATMENT_COL: picked[TREATMENT_COL].to_numpy()}
    )
    for col in FEATURE_COLS:
        exposures[col] = picked[col].to_numpy()

    lags = rng.uniform(lag_range[0], lag_range[1], size=n)
    outcomes = pd.DataFrame(
        {
            "user_id": order,
            "event_time": event_time + lags,
            VISIT_COL: picked[VISIT_COL].to_numpy(),
            CONVERSION_COL: picked[CONVERSION_COL].to_numpy() if CONVERSION_COL in picked else 0,
        }
    ).sort_values("event_time", kind="stable").reset_index(drop=True)
    return exposures, outcomes


def merged_events(exposures: pd.DataFrame, outcomes: pd.DataFrame) -> Iterator[dict]:
    def stream(frame: pd.DataFrame, topic: str, priority: int):
        for record in frame.to_dict("records"):
            yield (record["event_time"], priority, {"topic": topic, "key": str(int(record["user_id"])), "value": record})

    merged = heapq.merge(
        stream(exposures, EXPOSURES_TOPIC, 0), stream(outcomes, OUTCOMES_TOPIC, 1), key=lambda item: (item[0], item[1])
    )
    for _, _, event in merged:
        yield event


def inject_drift(exposures: pd.DataFrame, feature: str, shift: float, after_fraction: float = 0.5) -> pd.DataFrame:
    drifted = exposures.copy()
    start = int(len(drifted) * after_fraction)
    drifted.loc[drifted.index[start:], feature] = drifted[feature].iloc[start:] + shift
    return drifted
