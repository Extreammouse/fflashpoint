
# spark_analytics.py
# PySpark Structured Streaming: Gaze-Contingent Spatial Join

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, explode
from pyspark.sql.types import StructType, StructField, DoubleType, ArrayType, StringType, IntegerType

# Define schema for incoming JSON
detected_object_schema = StructType([
	StructField("label", StringType(), True),
	StructField("bbox", ArrayType(DoubleType()), True)  # [x_min, y_min, x_max, y_max]
])

event_schema = StructType([
	StructField("gaze_x", DoubleType(), True),
	StructField("gaze_y", DoubleType(), True),
	StructField("detected_objects", ArrayType(detected_object_schema), True)
])

# UDF to check if gaze is inside any sensitive bbox
def gaze_in_sensitive_bbox(gaze_x, gaze_y, detected_objects):
	if not detected_objects:
		return False
	for obj in detected_objects:
		if obj.get('label') == 'sensitive':
			bbox = obj.get('bbox', [])
			if len(bbox) == 4:
				x_min, y_min, x_max, y_max = bbox
				if x_min <= gaze_x <= x_max and y_min <= gaze_y <= y_max:
					return True
	return False

gaze_in_sensitive_bbox_udf = udf(gaze_in_sensitive_bbox, StringType())

if __name__ == "__main__":
	spark = SparkSession.builder \
		.appName("FlashPointSpatialJoin") \
		.getOrCreate()
	spark.sparkContext.setLogLevel("WARN")

	# Read from Kafka topic 'gaze_events'
	df_raw = spark.readStream \
		.format("kafka") \
		.option("kafka.bootstrap.servers", "localhost:9092") \
		.option("subscribe", "gaze_events") \
		.option("startingOffsets", "latest") \
		.load()

	# Parse JSON
	df_parsed = df_raw.select(from_json(col("value").cast("string"), event_schema).alias("data")).select("data.*")

	# Filter for gaze in any sensitive bbox
	df_filtered = df_parsed.withColumn(
		"violation",
		gaze_in_sensitive_bbox_udf(col("gaze_x"), col("gaze_y"), col("detected_objects"))
	).filter(col("violation") == "True")

	# Write to console
	query_console = df_filtered.writeStream \
		.outputMode("append") \
		.format("console") \
		.option("truncate", False) \
		.start()

	# Write to Kafka topic 'intervention_trigger'
	df_to_kafka = df_filtered.selectExpr("to_json(struct(*)) AS value")
	query_kafka = df_to_kafka.writeStream \
		.format("kafka") \
		.option("kafka.bootstrap.servers", "localhost:9092") \
		.option("topic", "intervention_trigger") \
		.option("checkpointLocation", "./data/logs/spark_checkpoint") \
		.outputMode("append") \
		.start()

	query_console.awaitTermination()
	query_kafka.awaitTermination()
