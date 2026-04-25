from src.alert import AlertController


def _hit(label="upper_torso", confidence=0.9, bbox=None):
    return {
        "label": label,
        "bbox": bbox or [10, 20, 30, 40],
        "confidence": confidence,
    }


def test_alert_controller_stays_inactive_when_disabled():
    controller = AlertController(
        enabled=False,
        dwell_ms=100,
        duration_ms=200,
        cooldown_seconds=1.0,
    )

    state = controller.update(1.0, [_hit()])

    assert not state.enabled
    assert not state.active
    assert not state.triggered


def test_alert_controller_requires_dwell_before_triggering():
    controller = AlertController(
        enabled=True,
        dwell_ms=350,
        duration_ms=200,
        cooldown_seconds=1.0,
    )

    initial = controller.update(10.0, [_hit()])
    before_dwell = controller.update(10.3, [_hit()])
    after_dwell = controller.update(10.36, [_hit()])

    assert not initial.triggered
    assert not before_dwell.triggered
    assert after_dwell.triggered
    assert after_dwell.active
    assert after_dwell.active_label == "upper_torso"


def test_alert_controller_resets_dwell_when_hit_disappears():
    controller = AlertController(
        enabled=True,
        dwell_ms=350,
        duration_ms=200,
        cooldown_seconds=1.0,
    )

    controller.update(10.0, [_hit()])
    controller.update(10.2, [])
    state = controller.update(10.36, [_hit()])

    assert not state.triggered
    assert state.dwell_progress_ms == 0


def test_alert_controller_respects_cooldown():
    controller = AlertController(
        enabled=True,
        dwell_ms=100,
        duration_ms=100,
        cooldown_seconds=1.0,
    )

    first = controller.update(5.0, [_hit()])
    second = controller.update(5.11, [_hit()])
    during_cooldown = controller.update(5.7, [_hit()])
    after_cooldown = controller.update(6.12, [_hit()])

    assert not first.triggered
    assert second.triggered
    assert not during_cooldown.triggered
    assert after_cooldown.triggered
