"""
spark_analytics.py — Flash Point Spark Structured Streaming Processor

Consumes gaze_events from Kafka, applies a UDF to check whether the user's
gaze falls inside a bounding box of any sensitive-label object, and publishes
violation records to the intervention_trigger topic.

Expected input schema (JSON in Kafka value):
{
  "timestamp":        float,        # Unix epoch seconds
  "gaze_x":           float,        # Gaze x-coordinate in frame pixels
  "gaze_y":           float,        # Gaze y-coordinate in frame pixels
  "detected_objects": [             # List of objects detected by vision_node.py
    {
      "label":      str,            # e.g. "head"
      "bbox":       [x1, y1, x2, y2],  # Pixel coords (top-left, bottom-right)
      "confidence": float           # 0.0 – 1.0
    }
  ]
}

Output schema published to intervention_trigger (JSON):
{
  "timestamp":     float,
  "violation":     true,
  "gaze_x":        float,
  "gaze_y":        float,
  "matched_label": str,
  "confidence":    float
}

Run with:
  spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
    spark_analytics.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, lit, struct, to_json, udf
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    FloatType,
    StringType,
    StructField,
    StructType,
)

from gaze_udf import (
    _is_gaze_in_sensitive_bbox,
    _matched_confidence,
    _matched_label,
)

# ─── Configuration ────────────────────────────────────────────────────────────

KAFKA_BROKER = "localhost:9092"
INPUT_TOPIC = "gaze_events"
OUTPUT_TOPIC = "intervention_trigger"
CHECKPOINT_DIR = "/tmp/flash_point_checkpoint"

# ─── Input Schema ─────────────────────────────────────────────────────────────

detected_object_schema = StructType([
    StructField("label", StringType(), nullable=False),
    StructField("bbox", ArrayType(FloatType()), nullable=False),
    StructField("confidence", FloatType(), nullable=False),
])

gaze_event_schema = StructType([
    StructField("timestamp", FloatType(), nullable=False),
    StructField("gaze_x", FloatType(), nullable=False),
    StructField("gaze_y", FloatType(), nullable=False),
    StructField("detected_objects", ArrayType(detected_object_schema), nullable=False),
])

# ─── UDF wrappers (logic in gaze_udf.py) ─────────────────────────────────────

is_gaze_in_sensitive_bbox = udf(_is_gaze_in_sensitive_bbox, BooleanType())
matched_label_udf = udf(_matched_label, StringType())
matched_confidence_udf = udf(_matched_confidence, FloatType())


def main():
    spark = (
        SparkSession.builder.appName("FlashPoint-GazeAnalytics")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", INPUT_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = (
        raw_stream.selectExpr("CAST(value AS STRING) AS json_str")
        .select(from_json(col("json_str"), gaze_event_schema).alias("evt"))
        .select("evt.*")
    )

    annotated = (
        parsed.withColumn(
            "violation",
            is_gaze_in_sensitive_bbox(
                col("gaze_x"), col("gaze_y"), col("detected_objects")
            ),
        )
        .withColumn(
            "matched_label",
            matched_label_udf(col("gaze_x"), col("gaze_y"), col("detected_objects")),
        )
        .withColumn(
            "matched_confidence",
            matched_confidence_udf(
                col("gaze_x"), col("gaze_y"), col("detected_objects")
            ),
        )
    )

    violations = annotated.filter(col("violation"))  # boolean column

    violation_output = violations.select(
        to_json(
            struct(
                col("timestamp"),
                lit(True).alias("violation"),
                col("gaze_x"),
                col("gaze_y"),
                col("matched_label"),
                col("matched_confidence").alias("confidence"),
            )
        ).alias("value")
    )

    violation_output.writeStream.format("kafka").option(
        "kafka.bootstrap.servers", KAFKA_BROKER
    ).option("topic", OUTPUT_TOPIC).option(
        "checkpointLocation", CHECKPOINT_DIR + "/kafka"
    ).outputMode("append").start()

    annotated.select(
        "timestamp",
        "gaze_x",
        "gaze_y",
        "violation",
        "matched_label",
        "matched_confidence",
    ).writeStream.format("console").option("truncate", False).option(
        "checkpointLocation", CHECKPOINT_DIR + "/console"
    ).outputMode("append").start()

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
