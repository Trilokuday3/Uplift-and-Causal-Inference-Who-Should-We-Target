from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StructField, StructType

from streaming.events import EXPOSURES_TOPIC, OUTCOMES_TOPIC
from streaming.sequential import msprt_confidence_sequence, naive_pvalue_path
from uplift.constants import FEATURE_COLS
from uplift.spark_env import LOCAL_SPARK_CONFIG, apply_windows_spark_defaults


class WindowAccumulator:
    """Driver-side state for the running A/B test: counts per (time window, arm)."""

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._frames: list[pd.DataFrame] = []

    def update(self, pdf: pd.DataFrame) -> None:
        if len(pdf):
            self._frames.append(pdf.copy())

    @property
    def n_rows(self) -> int:
        return sum(len(f) for f in self._frames)

    def rows(self) -> pd.DataFrame:
        return pd.concat(self._frames, ignore_index=True) if self._frames else pd.DataFrame()

    def running_ate(self) -> pd.DataFrame:
        """Cumulative ATE (visit rate treated - control) with a normal-approx 95% CI, per window."""
        rows = self.rows()
        rows["window_start"] = (np.floor(rows["exposure_time"] / self.window_seconds) * self.window_seconds).astype(int)
        grouped = rows.groupby(["window_start", "treatment"])["visit"].agg(["count", "sum"]).unstack("treatment", fill_value=0)
        out = pd.DataFrame(index=grouped.index)
        out["n_treated"] = grouped[("count", 1)].cumsum()
        out["n_control"] = grouped[("count", 0)].cumsum()
        visits_t, visits_c = grouped[("sum", 1)].cumsum(), grouped[("sum", 0)].cumsum()
        p_t, p_c = visits_t / out["n_treated"], visits_c / out["n_control"]
        out["ate"] = p_t - p_c
        se = np.sqrt(p_t * (1 - p_t) / out["n_treated"] + p_c * (1 - p_c) / out["n_control"])
        out["ci_low"], out["ci_high"] = out["ate"] - 1.96 * se, out["ate"] + 1.96 * se
        return out.reset_index()

    def total_ate(self) -> dict[str, float]:
        return self.running_ate().iloc[-1].to_dict()


def consistency_check(streaming_ate: float, batch_ate: float, tol: float = 0.01) -> bool:
    """Streaming ATE must match the batch ATE from the classic analysis within `tol` (relative)."""
    return abs(streaming_ate - batch_ate) <= tol * abs(batch_ate)


def join_streams(exposures: DataFrame, outcomes: DataFrame, watermark: str = "10 minutes", max_lag_seconds: int = 3600) -> DataFrame:
    """Stream-stream inner join of exposures to their delayed outcomes, bounded by a watermark
    and a maximum lag so Spark can drop old state."""
    exp = (
        exposures.select("user_id", F.col("event_time").alias("exposure_time"), "treatment")
        .withColumn("exposure_ts", F.col("exposure_time").cast("timestamp"))
        .withWatermark("exposure_ts", watermark)
    )
    out = (
        outcomes.select(F.col("user_id").alias("o_user_id"), F.col("event_time").alias("outcome_time"), "visit")
        .withColumn("outcome_ts", F.col("outcome_time").cast("timestamp"))
        .withWatermark("outcome_ts", watermark)
    )
    condition = F.expr(
        f"user_id = o_user_id AND outcome_ts >= exposure_ts AND outcome_ts <= exposure_ts + interval {max_lag_seconds} seconds"
    )
    return exp.join(out, condition, "inner").select("user_id", "exposure_time", "outcome_time", "treatment", "visit")


def read_file_stream(spark: SparkSession, path: str) -> DataFrame:
    return spark.readStream.schema(spark.read.parquet(path).schema).parquet(path)


def _event_schema(kind: str) -> StructType:
    base = [StructField("user_id", LongType()), StructField("event_time", DoubleType())]
    if kind == EXPOSURES_TOPIC:
        return StructType(base + [StructField("treatment", LongType())] + [StructField(c, DoubleType()) for c in FEATURE_COLS])
    return StructType(base + [StructField("visit", LongType()), StructField("conversion", LongType())])


def read_kafka_stream(spark: SparkSession, bootstrap_servers: str, topic: str) -> DataFrame:
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .load()
    )
    return raw.select(F.from_json(F.col("value").cast("string"), _event_schema(topic)).alias("e")).select("e.*")


def run_stream_job(
    spark: SparkSession,
    exposures_path: str,
    outcomes_path: str,
    checkpoint: str,
    window_seconds: int = 60,
) -> WindowAccumulator:
    """Process file-based exposure/outcome streams to completion (Trigger.AvailableNow)."""
    accumulator = WindowAccumulator(window_seconds)
    joined = join_streams(read_file_stream(spark, exposures_path), read_file_stream(spark, outcomes_path))
    query = (
        joined.writeStream.foreachBatch(lambda batch, _id: accumulator.update(batch.toPandas()))
        .option("checkpointLocation", checkpoint)
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination()
    return accumulator


def sequential_report(accumulator: WindowAccumulator, step: int = 500) -> dict:
    """Peeking demo: naive p-value path vs always-valid mSPRT sequence, in outcome-arrival order."""
    rows = accumulator.rows().sort_values("outcome_time")
    checkpoints = list(range(step, len(rows) + 1, step)) or [len(rows)]
    y, t = rows["visit"].to_numpy(), rows["treatment"].to_numpy()
    naive = naive_pvalue_path(y, t, checkpoints)
    sequence = msprt_confidence_sequence(y, t, checkpoints)
    return {
        "naive_first_p_below_0_05_at_n": _first(naive.loc[naive["p_value"] < 0.05, "n"]),
        "msprt_first_excludes_zero_at_n": _first(sequence.loc[(sequence["ci_low"] > 0) | (sequence["ci_high"] < 0), "n"]),
        "naive_path": naive.to_dict(orient="records"),
        "msprt_path": sequence.to_dict(orient="records"),
    }


def _first(series: pd.Series):
    return int(series.iloc[0]) if len(series) else None


def build_spark(kafka: bool) -> SparkSession:
    apply_windows_spark_defaults()
    builder = SparkSession.builder.master("local[2]").appName("uplift-ab-stream").config("spark.ui.enabled", "false")
    builder = builder.config("spark.sql.shuffle.partitions", "4").config("spark.driver.memory", "4g")
    for key, value in LOCAL_SPARK_CONFIG.items():
        builder = builder.config(key, value)
    if kafka:
        scala = "2.13" if int(pyspark.__version__.split(".")[0]) >= 4 else "2.12"
        builder = builder.config("spark.jars.packages", f"org.apache.spark:spark-sql-kafka-0-10_{scala}:{pyspark.__version__}")
    return builder.getOrCreate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming A/B: join exposures to delayed outcomes, running ATE.")
    parser.add_argument("--source", choices=["files", "kafka"], default="kafka")
    parser.add_argument("--exposures-path", default="data/stream/exposures")
    parser.add_argument("--outcomes-path", default="data/stream/outcomes")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--checkpoint", default="data/stream/checkpoint")
    parser.add_argument("--output", default="docs/streaming_results.json")
    parser.add_argument("--timeout", type=int, default=120, help="seconds to run in kafka mode")
    parser.add_argument("--batch-ate", type=float, default=None, help="batch ATE to check consistency against")
    args = parser.parse_args()

    spark = build_spark(kafka=args.source == "kafka")
    spark.sparkContext.setLogLevel("ERROR")
    if args.source == "files":
        accumulator = run_stream_job(spark, args.exposures_path, args.outcomes_path, args.checkpoint)
    else:
        accumulator = WindowAccumulator()
        joined = join_streams(
            read_kafka_stream(spark, args.bootstrap_servers, EXPOSURES_TOPIC),
            read_kafka_stream(spark, args.bootstrap_servers, OUTCOMES_TOPIC),
        )
        query = (
            joined.writeStream.foreachBatch(lambda batch, _id: accumulator.update(batch.toPandas()))
            .option("checkpointLocation", args.checkpoint)
            .trigger(processingTime="5 seconds")
            .start()
        )
        query.awaitTermination(args.timeout)
        query.stop()

    total = accumulator.total_ate()
    results = {
        "label": "simulated real-time replay",
        "rows_joined": accumulator.n_rows,
        "total": total,
        "running_ate": accumulator.running_ate().to_dict(orient="records"),
        "sequential": sequential_report(accumulator),
    }
    if args.batch_ate is not None:
        results["batch_ate"] = args.batch_ate
        results["consistent_with_batch_within_1pct"] = consistency_check(total["ate"], args.batch_ate)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, default=float))
    print(f"streaming ATE {total['ate']:.5f} over {accumulator.n_rows} joined events")


if __name__ == "__main__":
    main()
