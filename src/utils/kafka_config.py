# kafka_config.py
# Shared Kafka settings for producers and consumers
# This file centralizes Kafka configuration to ensure consistency across producers and consumers.


import json

# Broker
KAFKA_BROKER = "localhost:9092"

# Topics
GAZE_EVENTS_TOPIC = "gaze_events"
TRIGGER_TOPIC = "intervention_trigger"

# Producer config — used by stub_producer.py and vision_node.py
PRODUCER_CONFIG = {
    "bootstrap_servers": KAFKA_BROKER,
    "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
}

# Consumer config — used by stub_consumer.py and flash_trigger.py
CONSUMER_CONFIG = {
    "bootstrap_servers": KAFKA_BROKER,
    "auto_offset_reset": "latest",
    "value_deserializer": lambda v: json.loads(v.decode("utf-8")),
}

# Flash settings
FLASH_DURATION_MS = 200  # how long the flash stays white
