"""Centralized Kafka configuration for the FLASH-POINT pipeline."""

from __future__ import annotations

BOOTSTRAP_SERVERS = "localhost:9092"
GAZE_EVENTS_TOPIC = "gaze_events"
TRIGGER_TOPIC = "intervention_trigger"
CONSUMER_GROUP = "flash_point_group"

PRODUCER_CONFIG = {
    "bootstrap_servers": BOOTSTRAP_SERVERS,
    "value_serializer": None,  # set manually — json.dumps().encode()
    "acks": "all",
    "retries": 3,
    "linger_ms": 5,
}

CONSUMER_CONFIG = {
    "bootstrap_servers": BOOTSTRAP_SERVERS,
    "group_id": CONSUMER_GROUP,
    "auto_offset_reset": "latest",
    "enable_auto_commit": True,
    "value_deserializer": None,  # set manually — json.loads()
}

FLASH_DURATION_MS = 2000
