from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from streaming.events import EXPOSURES_TOPIC
from streaming.producer import KafkaSink
from uplift.constants import FEATURE_COLS

DECISIONS_TOPIC = "decisions"
DRIFT_TOPIC = "drift"


def threshold_for_fraction(scores, fraction: float) -> float:
    """Score cut-off that treats the top `fraction` of users."""
    return float(np.quantile(np.asarray(scores, dtype=float), 1 - fraction))


def events_to_frame(events: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([e["value"] for e in events])[FEATURE_COLS]


def score_events(model, events: list[dict], threshold: float) -> list[dict]:
    """Score a micro-batch of exposure events. Only FEATURE_COLS reach the model."""
    frame = events_to_frame(events)
    uplift = np.asarray(model.predict_uplift(frame), dtype=float)
    return [
        {"user_id": int(e["value"]["user_id"]), "event_time": e["value"]["event_time"],
         "uplift": float(u), "treat": bool(u >= threshold)}
        for e, u in zip(events, uplift)
    ]


def run_scorer(consumer, sink, model, threshold: float, batch_size: int = 500, poll_timeout: float = 1.0,
               max_idle_polls: int | None = None, drift_monitor=None) -> int:
    """Consume exposures in micro-batches, score them, publish decisions. Returns events scored.
    With a drift monitor, each full window is tested against the training reference and any
    drifted features are published to the `drift` topic."""
    processed, idle = 0, 0
    while max_idle_polls is None or idle < max_idle_polls:
        events = consumer.poll_batch(batch_size, poll_timeout)
        if not events:
            idle += 1
            continue
        idle = 0
        for decision in score_events(model, events, threshold):
            sink.send(DECISIONS_TOPIC, str(decision["user_id"]), decision)
        processed += len(events)
        if drift_monitor is not None:
            flagged = drift_monitor.update(events_to_frame(events))
            if flagged:
                sink.send(DRIFT_TOPIC, "drift", {
                    "features": flagged, "n_processed": processed, "event_time": events[-1]["value"]["event_time"]})
    sink.flush()
    return processed


def benchmark_scoring(model, frame: pd.DataFrame, batch_size: int = 500, n_batches: int = 20) -> dict[str, float]:
    """Single-process scoring throughput and latency. An event's latency is its whole batch's
    scoring time (it waits for the batch), so p95_event_ms is the conservative per-event figure."""
    batch = frame[FEATURE_COLS].iloc[:batch_size]
    latencies = []
    for _ in range(n_batches):
        start = time.perf_counter()
        model.predict_uplift(batch)
        latencies.append((time.perf_counter() - start) * 1000)
    total_seconds = sum(latencies) / 1000
    p95 = float(np.percentile(latencies, 95))
    return {
        "events_per_sec": len(batch) * n_batches / total_seconds,
        "p95_batch_ms": p95,
        "p95_event_ms": p95,
        "batch_size": len(batch),
    }


class KafkaBatchConsumer:
    def __init__(self, bootstrap_servers: str, topic: str = EXPOSURES_TOPIC, group_id: str = "uplift-scorer"):
        from confluent_kafka import Consumer

        self._consumer = Consumer(
            {"bootstrap.servers": bootstrap_servers, "group.id": group_id, "auto.offset.reset": "earliest"}
        )
        self._consumer.subscribe([topic])

    def poll_batch(self, n: int, timeout: float) -> list[dict]:
        messages = self._consumer.consume(num_messages=n, timeout=timeout)
        return [
            {"topic": m.topic(), "key": m.key().decode(), "value": json.loads(m.value())}
            for m in messages
            if m.error() is None
        ]


class PostgresDecisionSink:
    """Mirrors decisions into Postgres (requires psycopg2, imported lazily)."""

    DDL = (
        "CREATE TABLE IF NOT EXISTS decisions (user_id BIGINT, event_time DOUBLE PRECISION, "
        "uplift DOUBLE PRECISION, treat BOOLEAN)"
    )

    def __init__(self, dsn: str):
        import psycopg2

        self._conn = psycopg2.connect(dsn)
        with self._conn, self._conn.cursor() as cur:
            cur.execute(self.DDL)

    def send(self, topic: str, key: str, value: dict) -> None:
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO decisions VALUES (%s, %s, %s, %s)",
                (value["user_id"], value["event_time"], value["uplift"], value["treat"]),
            )

    def flush(self) -> None:
        pass


class FanOutSink:
    def __init__(self, *sinks):
        self.sinks = sinks

    def send(self, topic, key, value):
        for sink in self.sinks:
            sink.send(topic, key, value)

    def flush(self):
        for sink in self.sinks:
            sink.flush()


def load_model(artifacts_dir: str):
    return joblib.load(Path(artifacts_dir) / "best_model.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score exposure events and publish treat/don't-treat decisions.")
    parser.add_argument("--artifacts-dir", default="models")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--fraction", type=float, default=0.3, help="share of users to treat (from the policy)")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--postgres-dsn", default=None)
    parser.add_argument("--reference-dir", default=None, help="processed data dir; enables drift monitoring")
    parser.add_argument("--drift-window", type=int, default=1000)
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    meta = json.loads((artifacts / "model_meta.json").read_text())
    reference_scores = pd.read_parquet(artifacts / "test_scores.parquet")[meta["model_name"]]
    threshold = threshold_for_fraction(reference_scores, args.fraction)

    sink = KafkaSink(args.bootstrap_servers)
    if args.postgres_dsn:
        sink = FanOutSink(sink, PostgresDecisionSink(args.postgres_dsn))
    monitor = None
    if args.reference_dir:
        import pyarrow.dataset as ds

        from streaming.drift import DriftMonitor

        reference = ds.dataset(f"{args.reference_dir}/train").head(20_000).to_pandas()[FEATURE_COLS]
        monitor = DriftMonitor(reference, FEATURE_COLS, window_size=args.drift_window)
    consumer = KafkaBatchConsumer(args.bootstrap_servers)
    print(f"scoring with {meta['model_name']}, treating top {args.fraction:.0%} (threshold {threshold:.5f})")
    run_scorer(consumer, sink, load_model(args.artifacts_dir), threshold, batch_size=args.batch_size,
               drift_monitor=monitor)


if __name__ == "__main__":
    main()
