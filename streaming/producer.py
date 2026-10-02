from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from streaming.events import EXPOSURES_TOPIC, OUTCOMES_TOPIC, build_events, inject_drift, merged_events


class ListSink:
    """In-memory sink used by tests and dry runs."""

    def __init__(self):
        self.messages: list[tuple[str, str, dict]] = []

    def send(self, topic: str, key: str, value: dict) -> None:
        self.messages.append((topic, key, value))

    def flush(self) -> None:
        pass


class KafkaSink:
    def __init__(self, bootstrap_servers: str):
        from confluent_kafka import Producer

        self._producer = Producer({"bootstrap.servers": bootstrap_servers, "linger.ms": 20})

    def send(self, topic: str, key: str, value: dict) -> None:
        self._producer.produce(topic, key=key, value=json.dumps(value))
        self._producer.poll(0)

    def flush(self) -> None:
        self._producer.flush()


def create_topics(bootstrap_servers: str, exposure_partitions: int = 6) -> None:
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    topics = [
        NewTopic(EXPOSURES_TOPIC, num_partitions=exposure_partitions, replication_factor=1),
        NewTopic(OUTCOMES_TOPIC, num_partitions=exposure_partitions, replication_factor=1),
        NewTopic("decisions", num_partitions=exposure_partitions, replication_factor=1),
        NewTopic("drift", num_partitions=1, replication_factor=1),
    ]
    for future in admin.create_topics(topics).values():
        try:
            future.result()
        except Exception as error:  # topic already exists is fine
            if "already exists" not in str(error):
                raise


def replay(
    events: Iterable[dict],
    sink,
    speed: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> int:
    """Emit events on their simulated clock (event_time / speed seconds after start)."""
    start, sent, origin = now(), 0, None
    for event in events:
        if origin is None:
            origin = event["value"]["event_time"]
        wait = (event["value"]["event_time"] - origin) / speed - (now() - start)
        if wait > 0:
            sleep(wait)
        sink.send(event["topic"], event["key"], event["value"])
        sent += 1
    sink.flush()
    return sent


def write_event_files(exposures: pd.DataFrame, outcomes: pd.DataFrame, out_dir) -> None:
    """File-based stand-in for Kafka: parquet dirs the Spark job can stream, plus the batch ATE of
    exactly these events for the streaming-vs-batch consistency check."""
    out = Path(out_dir)
    for name, frame in (("exposures", exposures), ("outcomes", outcomes)):
        (out / name).mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out / name / "part-0.parquet")
    write_meta(exposures, outcomes, out)


def write_meta(exposures: pd.DataFrame, outcomes: pd.DataFrame, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joined = exposures[["user_id", "treatment"]].merge(outcomes[["user_id", "visit"]], on="user_id")
    rates = joined.groupby("treatment")["visit"].mean()
    meta = {"batch_ate": float(rates[1] - rates[0]), "n_events": int(len(joined))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated real-time replay of the test split into Kafka.")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--eps", type=float, default=5000.0, help="simulated exposures per second")
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--speed", type=float, default=1.0, help="wall-clock speedup of the simulated clock")
    parser.add_argument("--lag-min", type=float, default=0.5)
    parser.add_argument("--lag-max", type=float, default=5.0)
    parser.add_argument("--drift-feature", default=None, help="feature to shift halfway through (drift demo)")
    parser.add_argument("--drift-shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sink", choices=["kafka", "files"], default="kafka")
    parser.add_argument("--out-dir", default="data/stream", help="output for --sink files")
    args = parser.parse_args()

    test = pd.read_parquet(f"{args.input_dir}/test")
    exposures, outcomes = build_events(
        test, seed=args.seed, eps=args.eps, lag_range=(args.lag_min, args.lag_max), limit=args.limit
    )
    if args.drift_feature:
        exposures = inject_drift(exposures, args.drift_feature, args.drift_shift)
    if args.sink == "files":
        write_event_files(exposures, outcomes, args.out_dir)
        print(f"simulated real-time replay (files): wrote {len(exposures)} exposures to {args.out_dir}")
        return
    write_meta(exposures, outcomes, args.out_dir)
    create_topics(args.bootstrap_servers)
    sent = replay(merged_events(exposures, outcomes), KafkaSink(args.bootstrap_servers), speed=args.speed)
    print(f"simulated real-time replay: sent {sent} events")


if __name__ == "__main__":
    main()
