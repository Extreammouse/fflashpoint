"""Transient full-monitor screen capture helpers.

Frames returned by this module are in-memory NumPy arrays. The module does not
write screenshots, videos, or frame dumps.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScreenFrame:
    image_bgr: np.ndarray
    screen_size: tuple[int, int]
    timestamp: float
    monitor: dict[str, int]


def monitor_size(monitor: Mapping[str, Any]) -> tuple[int, int]:
    width = int(monitor.get("width", 0))
    height = int(monitor.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid monitor dimensions: {monitor}")
    return width, height


class PrimaryMonitorCapture:
    """Capture the configured full primary monitor using mss."""

    def __init__(self, monitor_index: int = 1) -> None:
        try:
            import mss
        except ImportError as exc:  # pragma: no cover - depends on environment.
            raise RuntimeError(
                "Missing dependency: mss. Install requirements.txt first."
            ) from exc

        self._mss = mss.mss()
        monitors = self._mss.monitors
        if monitor_index < 0 or monitor_index >= len(monitors):
            raise RuntimeError(
                f"Monitor index {monitor_index} is unavailable. "
                f"mss exposed {len(monitors)} monitor entries."
            )

        self.monitor = dict(monitors[monitor_index])
        self.screen_size = monitor_size(self.monitor)

    def read(self) -> ScreenFrame:
        raw = self._mss.grab(self.monitor)
        bgra = np.asarray(raw, dtype=np.uint8)
        image_bgr = bgra[:, :, :3].copy()
        return ScreenFrame(
            image_bgr=image_bgr,
            screen_size=self.screen_size,
            timestamp=time.time(),
            monitor=dict(self.monitor),
        )

    def close(self) -> None:
        self._mss.close()
