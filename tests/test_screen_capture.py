import pytest

from src.screen_capture import monitor_size


def test_monitor_size_returns_width_height():
    assert monitor_size({"left": 0, "top": 0, "width": 2560, "height": 1600}) == (
        2560,
        1600,
    )


def test_monitor_size_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        monitor_size({"width": 0, "height": 1600})
