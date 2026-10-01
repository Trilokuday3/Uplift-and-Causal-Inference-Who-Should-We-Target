from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from streaming.drift import DriftMonitor
from streaming.events import build_events, inject_drift
from uplift.constants import FEATURE_COLS


def run_demo(train: pd.DataFrame, test: pd.DataFrame, feature: str, shift: float, limit: int, window: int) -> dict:
    """Replay `test` with `feature` shifted halfway through and report which windows flag drift."""
    exposures, _ = build_events(test, seed=42, eps=1000.0, lag_range=(0.5, 5.0), limit=limit)
    exposures = inject_drift(exposures, feature, shift)
    monitor = DriftMonitor(train[FEATURE_COLS], FEATURE_COLS, window_size=window)
    windows = []
    for start in range(0, len(exposures) - window + 1, window):
        flagged = monitor.update(exposures.iloc[start:start + window])
        windows.append({"start_row": start, "drifted": flagged})
    first = next((w["start_row"] for w in windows if w["drifted"]), None)
    return {"feature": feature, "shift": shift, "shift_starts_at_row": limit // 2,
            "first_flagged_row": first, "windows": windows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-style drift demo on the replayed test split.")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--feature", default="f0")
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--output", default="docs/drift_demo.json")
    args = parser.parse_args()
    train = pd.read_parquet(f"{args.input_dir}/train").sample(50_000, random_state=42)
    result = run_demo(train, pd.read_parquet(f"{args.input_dir}/test"), args.feature, args.shift, args.limit, args.window)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"shift starts at row {result['shift_starts_at_row']}, first flagged at row {result['first_flagged_row']}")


if __name__ == "__main__":
    main()
