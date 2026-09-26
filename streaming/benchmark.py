from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from streaming.scorer import benchmark_scoring, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-process scoring throughput and latency.")
    parser.add_argument("--artifacts-dir", default="models")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument("--output", default="docs/streaming_benchmark.json")
    args = parser.parse_args()

    frame = pd.read_parquet(f"{args.input_dir}/test").head(args.batch_size)
    model = load_model(args.artifacts_dir)
    stats = benchmark_scoring(model, frame, batch_size=args.batch_size, n_batches=args.n_batches)
    stats["model"] = json.loads((Path(args.artifacts_dir) / "model_meta.json").read_text())["model_name"]
    stats["note"] = "single process, in-memory scoring only; excludes Kafka and network overhead"
    Path(args.output).write_text(json.dumps(stats, indent=2))
    print(stats)


if __name__ == "__main__":
    main()
