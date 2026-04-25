"""Controlled visual alert state for gaze-target hits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class AlertSnapshot:
    enabled: bool
    active: bool
    triggered: bool
    matched_label: str | None
    matched_confidence: float | None
    active_label: str | None
    dwell_ms: int
    dwell_progress_ms: int
    cooldown_remaining_s: float

    def as_event(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "active": self.active,
            "triggered": self.triggered,
            "matched_label": self.matched_label,
            "matched_confidence": self.matched_confidence,
            "active_label": self.active_label,
            "dwell_ms": self.dwell_ms,
            "dwell_progress_ms": self.dwell_progress_ms,
            "cooldown_remaining_s": round(self.cooldown_remaining_s, 3),
        }


class AlertController:
    """Dwell/cooldown gate for visual-only target alerts."""

    def __init__(
        self,
        enabled: bool,
        dwell_ms: int,
        duration_ms: int,
        cooldown_seconds: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.dwell_ms = max(0, int(dwell_ms))
        self.duration_seconds = max(0.0, float(duration_ms) / 1000.0)
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._hit_started_at: float | None = None
        self._hit_key: tuple[str, tuple[float, float, float, float]] | None = None
        self._last_triggered_at: float | None = None
        self._active_until: float = 0.0
        self._active_label: str | None = None

    def update(self, timestamp: float, hits: Sequence[Mapping[str, Any]]) -> AlertSnapshot:
        now = float(timestamp)
        active = now < self._active_until
        hit = _first_hit(hits)
        hit_key = _hit_key(hit) if hit is not None else None
        matched_label = str(hit.get("label")) if hit is not None else None
        matched_confidence = _safe_confidence(hit.get("confidence")) if hit is not None else None
        triggered = False

        if not self.enabled:
            self._hit_started_at = None
            self._hit_key = None
            return self._snapshot(
                active=False,
                triggered=False,
                matched_label=matched_label,
                matched_confidence=matched_confidence,
                now=now,
                dwell_progress_ms=0,
            )

        if hit is None:
            self._hit_started_at = None
            self._hit_key = None
            return self._snapshot(
                active=active,
                triggered=False,
                matched_label=None,
                matched_confidence=None,
                now=now,
                dwell_progress_ms=0,
            )

        if hit_key != self._hit_key:
            self._hit_key = hit_key
            self._hit_started_at = now

        hit_started_at = self._hit_started_at if self._hit_started_at is not None else now
        dwell_progress_ms = int(max(0.0, now - hit_started_at) * 1000.0)
        cooldown_remaining_s = self._cooldown_remaining(now)
        dwell_ready = dwell_progress_ms >= self.dwell_ms
        cooldown_ready = cooldown_remaining_s <= 0.0

        if dwell_ready and cooldown_ready and not active:
            triggered = True
            self._last_triggered_at = now
            self._active_until = now + self.duration_seconds
            self._active_label = matched_label
            active = self.duration_seconds > 0.0

        return self._snapshot(
            active=active,
            triggered=triggered,
            matched_label=matched_label,
            matched_confidence=matched_confidence,
            now=now,
            dwell_progress_ms=dwell_progress_ms,
        )

    def _cooldown_remaining(self, now: float) -> float:
        if self._last_triggered_at is None:
            return 0.0
        elapsed = now - self._last_triggered_at
        return max(0.0, self.cooldown_seconds - elapsed)

    def _snapshot(
        self,
        active: bool,
        triggered: bool,
        matched_label: str | None,
        matched_confidence: float | None,
        now: float,
        dwell_progress_ms: int,
    ) -> AlertSnapshot:
        if not active and now >= self._active_until:
            self._active_label = None
        return AlertSnapshot(
            enabled=self.enabled,
            active=active,
            triggered=triggered,
            matched_label=matched_label,
            matched_confidence=matched_confidence,
            active_label=self._active_label,
            dwell_ms=self.dwell_ms,
            dwell_progress_ms=min(max(0, dwell_progress_ms), self.dwell_ms),
            cooldown_remaining_s=self._cooldown_remaining(now),
        )


def _first_hit(hits: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
        return None
    for hit in hits:
        if isinstance(hit, Mapping):
            return hit
    return None


def _hit_key(hit: Mapping[str, Any] | None) -> tuple[str, tuple[float, float, float, float]] | None:
    if hit is None:
        return None
    bbox = hit.get("bbox")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        return (str(hit.get("label")), (0.0, 0.0, 0.0, 0.0))
    try:
        bbox_key = tuple(round(float(value), 1) for value in bbox)
    except (TypeError, ValueError):
        bbox_key = (0.0, 0.0, 0.0, 0.0)
    return str(hit.get("label")), bbox_key


def _safe_confidence(value: Any) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None
