### Purpose

This file gives concise, repo-specific guidance for AI coding agents working on this project. Focus on discoverable, actionable knowledge: architecture, key files, developer workflows, and project-specific conventions.

### Big picture
- The repository is a small vision/data pipeline with three logical components:
  - Producers: `src/producers/` (e.g. `vision_node.py`) capture or simulate sensor/frame data.
  - Consumers: `src/consumers/` (e.g. `flash_trigger.py`) react to processed events.
  - Processors/Analytics: `src/processors/` contains data transformations and Spark UDFs (e.g. `gaze_udf.py`, `spark_analytics.py`).
- Messaging layer: components exchange messages via Kafka; configuration lives in `src/utils/kafka_config.py`.
- ML pipeline: training and evaluation code are in `ml/` (see `ml/train.py`, `ml/evaluate.py`) and model weights are stored in `models/` (e.g. `yolov8n.pt`).

### Key files to reference
- Project-level: `requirements.txt`, `docker/docker-compose.yml`, `README.md`.
- Producers: `src/producers/vision_node.py`, `src/producers/stub_producer.py`.
- Consumers: `src/consumers/flash_trigger.py`, `src/consumers/stub_consumer.py`.
- Processors: `src/processors/gaze_udf.py`, `src/processors/spark_analytics.py`.
- Utils/integration: `src/utils/kafka_config.py`.
- ML: `ml/train.py`, `ml/evaluate.py`, `ml/data/`.
- Tests: `tests/test_udf.py` demonstrates how UDFs are validated.

### Developer workflows (discoverable / runnable)
- Install deps: `pip install -r requirements.txt`.
- Bring up local infra (Kafka, etc.): `docker-compose -f docker/docker-compose.yml up`.
- Run a producer locally: `python src/producers/vision_node.py` (or `stub_producer.py` for simulated messages).
- Run a consumer locally: `python src/consumers/flash_trigger.py`.
- Run Spark analytics locally: `spark-submit --py-files src/processors/gaze_udf.py --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 src/processors/spark_analytics.py` (requires Kafka running and Java 17+).
- Run unit tests: `pytest -q tests/test_udf.py` (project uses pytest for core UDF tests).

### Project-specific conventions & patterns
- Messaging-first design: most cross-component calls use Kafka topics (see `kafka_config.py`). When editing producers/consumers, update topic names in that file.
- UDF-first processing: `gaze_udf.py` and `spark_analytics.py` are reference implementations—follow their function signatures and small, pure-function style for testability.
- Models and weights: training code expects model artifacts under `models/`. Don't hardcode absolute paths; prefer relative paths under `models/` or `ml/data/`.
- Tests target processors/UDFs directly (see `tests/test_udf.py`) rather than end-to-end Kafka flows — add unit tests for new UDFs following that pattern.

### Integration & external dependencies
- External services: Kafka (brought up via `docker/docker-compose.yml`), optional Spark for `spark_analytics.py`, and YOLO model weights under `models/`.
- Docker: use `docker/docker-compose.yml` to replicate the integration environment; inspect service names and exposed ports before wiring producers/consumers.

### When editing code, be mindful of
- Backwards compatibility for message schemas — search for topic names in `src/utils/kafka_config.py` and message keys/serializers in `src/producers` and `src/consumers`.
- Keep UDFs pure and small so `tests/test_udf.py` can exercise them without Kafka or Spark.

### Spark & Kafka runtime setup (macOS Homebrew)
- Install Java 17: `brew install --cask temurin@17`
- Install Spark: `brew install apache-spark`
- Set env vars (in terminal or `~/.zshrc`):
  ```bash
  export JAVA_HOME=$(/usr/libexec/java_home -v 17)
  export SPARK_HOME=$(brew --prefix)/opt/apache-spark/libexec
  export PATH="$JAVA_HOME/bin:$SPARK_HOME/bin:$PATH"
  ```
- Create Kafka topics (if not auto-created):
  ```bash
  docker exec docker-kafka-1 kafka-topics --create --topic gaze_events --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1
  docker exec docker-kafka-1 kafka-topics --create --topic intervention_trigger --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1
  ```
- Kafka connector: for Spark 4.1.1 use `org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1` (Scala 2.13); UDF modules must be shipped via `--py-files` or zipped package.

### Examples (copyable)
- Install deps: `pip install -r requirements.txt`
- Up infra: `docker-compose -f docker/docker-compose.yml up`
- Run producer: `python src/producers/vision_node.py`
- Run consumer: `python src/consumers/flash_trigger.py`
- Run Spark: `spark-submit --py-files src/processors/gaze_udf.py --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 src/processors/spark_analytics.py`
- Run UDF tests: `pytest -q tests/test_udf.py`

### If unsure / missing details
- Open `README.md` and `docker/docker-compose.yml` for service wiring and ports.
- Inspect `src/utils/kafka_config.py` for topic names and broker addresses used across the repo.


