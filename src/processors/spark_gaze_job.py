"""PySpark Structured Streaming job for the FLASH-POINT pipeline.

Reads gaze events from the 'gaze_events' Kafka topic, applies the spatial
gaze-hit UDF, and writes trigger messages to 'intervention_trigger'.

Run:
    PYTHONPATH=. spark-submit \\
      --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.0 \\
      --py-files src/processors/gaze_udf.py \\
      src/processors/spark_gaze_job.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    FloatType,
    StringType,
    StructField,
    StructType,
)

from src.utils.kafka_config import (
    BOOTSTRAP_SERVERS,
    GAZE_EVENTS_TOPIC,
    TRIGGER_TOPIC,
)
from src.processors.gaze_udf import (
    SENSITIVE_LABELS,
    CONFIDENCE_THRESHOLD,
    _is_gaze_in_sensitive_bbox,
)

# ── Schema for messages on gaze_events ────────────────────────────────────────

DETECTION_SCHEMA = StructType([
    StructField("label", StringType()),
    StructField("bbox", ArrayType(FloatType())),
    StructField("confidence", FloatType()),
])

GAZE_EVENT_SCHEMA = StructType([
    StructField("timestamp", FloatType()),
    StructField("gaze_x", FloatType()),
    StructField("gaze_y", FloatType()),
    StructField("confidence", FloatType()),
    StructField("detections", ArrayType(DETECTION_SCHEMA)),
])

# ── UDF ────────────────────────────────────────────────────────────────────────

def _hit_check(gaze_x, gaze_y, label, bbox, confidence):
    """Return True when gaze circle (radius=150) hits a sensitive detection."""
    if gaze_x is None or gaze_y is None or label is None or bbox is None:
        return False
    if label not in SENSITIVE_LABELS:
        return False
    try:
        if float(confidence) < CONFIDENCE_THRESHOLD:
            return False
        if len(bbox) < 4:
            return False
    except (TypeError, ValueError):
        return False
    return _is_gaze_in_sensitive_bbox(gaze_x, gaze_y, *bbox[:4], radius=150.0)


def main():
    spark = (
        SparkSession.builder
        .appName("FlashPoint-GazeHitDetector")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    hit_udf = F.udf(_hit_check, BooleanType())

    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
        .option("subscribe", GAZE_EVENTS_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = raw.select(
        F.from_json(F.col("value").cast("string"), GAZE_EVENT_SCHEMA).alias("e")
    ).select("e.*")

    # Explode detections so each row is one (gaze, detection) pair.
    exploded = parsed.select(
        "timestamp",
        "gaze_x",
        "gaze_y",
        "confidence",
        F.explode_outer("detections").alias("det"),
    ).select(
        "timestamp",
        "gaze_x",
        "gaze_y",
        "confidence",
        F.col("det.label").alias("label"),
        F.col("det.bbox").alias("bbox"),
        F.col("det.confidence").alias("det_confidence"),
    )

    hits = exploded.filter(
        hit_udf(
            F.col("gaze_x"),
            F.col("gaze_y"),
            F.col("label"),
            F.col("bbox"),
            F.col("det_confidence"),
        )
    )

    trigger_messages = hits.select(
        F.to_json(
            F.struct(
                F.col("timestamp"),
                F.col("label").alias("matched_label"),
                F.col("det_confidence").alias("confidence"),
                F.lit("flash").alias("action"),
            )
        ).alias("value")
    )

    query = (
        trigger_messages.writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
        .option("topic", TRIGGER_TOPIC)
        .option("checkpointLocation", "/tmp/flash_point_checkpoint")
        .outputMode("append")
        .start()
    )

    print(f"spark_gaze_job: streaming {GAZE_EVENTS_TOPIC} → {TRIGGER_TOPIC}")
    query.awaitTermination()


if __name__ == "__main__":
    main()
