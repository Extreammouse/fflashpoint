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
- Run Spark analytics locally: inspect `src/processors/spark_analytics.py`; use `spark-submit` if required by your environment, otherwise `python` for small-scale runs.
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

### Examples (copyable)
- Install deps: `pip install -r requirements.txt`
- Up infra: `docker-compose -f docker/docker-compose.yml up`
- Run producer: `python src/producers/vision_node.py`
- Run consumer: `python src/consumers/flash_trigger.py`
- Run UDF tests: `pytest -q tests/test_udf.py`

### If unsure / missing details
- Open `README.md` and `docker/docker-compose.yml` for service wiring and ports.
- Inspect `src/utils/kafka_config.py` for topic names and broker addresses used across the repo.

---
If you want, I can (1) merge this with any existing `.github/copilot-instructions.md` if present, (2) expand examples (exact spark-submit flags), or (3) run small repository scans to pull exact topic names and entrypoint docstrings into this file—which would require reading the files. Tell me which you'd prefer.
