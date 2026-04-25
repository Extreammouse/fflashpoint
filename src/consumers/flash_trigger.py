"""Kafka consumer: listens on intervention_trigger topic and fires the screen flash."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from src.screen_flash import ScreenFlashOverlay, resolve_screen_flash_style
from src.utils.kafka_config import (
    BOOTSTRAP_SERVERS,
    TRIGGER_TOPIC,
    CONSUMER_GROUP,
    FLASH_DURATION_MS,
)


def run():
    try:
        from kafka import KafkaConsumer
    except ImportError:
        raise SystemExit("kafka-python not installed. Run: pip install kafka-python")

    _, flash_alpha, flash_color = resolve_screen_flash_style(
        "white",
        overlay_alpha=config.SCREEN_FLASH_OVERLAY_ALPHA,
        overlay_color=config.SCREEN_FLASH_OVERLAY_COLOR,
        white_color=config.SCREEN_FLASH_WHITE_COLOR,
    )

    flash = ScreenFlashOverlay(
        enabled=True,
        duration_ms=FLASH_DURATION_MS,
        mode="white",
        alpha=flash_alpha,
        color=flash_color,
    )

    consumer = KafkaConsumer(
        TRIGGER_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=CONSUMER_GROUP,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    print(f"flash_trigger: consuming from '{TRIGGER_TOPIC}' on {BOOTSTRAP_SERVERS}")
    print("Press Ctrl+C to stop.")

    try:
        for message in consumer:
            payload = message.value
            label = payload.get("matched_label", "unknown")
            print(f"flash_trigger: trigger received — label={label}, firing flash")
            flash.trigger(label)
    except KeyboardInterrupt:
        print("\nflash_trigger: shutting down.")
    finally:
        consumer.close()
        flash.close()


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    run()
