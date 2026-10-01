from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from streaming.scorer import DECISIONS_TOPIC


def main() -> None:
    """Count decisions arriving on the `decisions` topic while `make replay` + `make score` run
    (start this first). Reports end-to-end throughput through Kafka -> scorer -> Kafka."""
    from confluent_kafka import Consumer

    parser = argparse.ArgumentParser(description="End-to-end Kafka throughput of the scoring service.")
    parser.add_argument("--bootstrap-servers", default="localhost:29092")
    parser.add_argument("--expected", type=int, required=True, help="decisions to wait for (= replay --limit)")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", default="docs/load_test.json")
    args = parser.parse_args()

    consumer = Consumer({"bootstrap.servers": args.bootstrap_servers, "group.id": f"load-test-{time.time()}",
                         "auto.offset.reset": "latest"})
    consumer.subscribe([DECISIONS_TOPIC])
    count, first, deadline = 0, None, time.monotonic() + args.timeout
    while count < args.expected and time.monotonic() < deadline:
        for m in consumer.consume(num_messages=1000, timeout=1.0):
            if m.error() is None:
                first = first or time.monotonic()
                count += 1
    elapsed = time.monotonic() - first if first else 0.0
    stats = {"decisions": count, "expected": args.expected, "seconds": elapsed,
             "decisions_per_sec": count / elapsed if elapsed else 0.0,
             "note": "end-to-end through Kafka; single scorer process, single broker"}
    Path(args.output).write_text(json.dumps(stats, indent=2))
    print(stats)


if __name__ == "__main__":
    main()
