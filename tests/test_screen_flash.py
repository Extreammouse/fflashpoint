from src.screen_flash import (
    ScreenFlashOverlay,
    normalize_screen_flash_bounds,
    resolve_screen_flash_style,
)


def test_disabled_screen_flash_does_not_start_gui_thread():
    flash = ScreenFlashOverlay(
        enabled=False,
        duration_ms=1500,
        mode="overlay",
        alpha=0.22,
        color="#ff3b30",
    )

    flash.trigger("upper_torso")
    metadata = flash.as_metadata()
    flash.close()

    assert metadata == {
        "enabled": False,
        "status": "disabled",
        "mode": "disabled",
        "duration_ms": 0,
        "alpha": 0.0,
        "color": "",
    }


def test_resolve_overlay_screen_flash_style():
    mode, alpha, color = resolve_screen_flash_style(
        "overlay",
        overlay_alpha=0.22,
        overlay_color="#ff3b30",
    )

    assert mode == "overlay"
    assert alpha == 0.22
    assert color == "#ff3b30"


def test_resolve_white_screen_flash_style_is_opaque():
    mode, alpha, color = resolve_screen_flash_style(
        "white",
        overlay_alpha=0.22,
        overlay_color="#ff3b30",
    )

    assert mode == "white"
    assert alpha == 1.0
    assert color == "#ffffff"


def test_invalid_screen_flash_style_falls_back_to_overlay():
    mode, alpha, color = resolve_screen_flash_style(
        "unknown",
        overlay_alpha=0.33,
        overlay_color="#123456",
    )

    assert mode == "overlay"
    assert alpha == 0.33
    assert color == "#123456"


def test_normalize_screen_flash_bounds_accepts_monitor_mapping():
    assert normalize_screen_flash_bounds(
        {"left": 0, "top": 0, "width": 2560, "height": 1600}
    ) == (0, 0, 2560, 1600)


def test_normalize_screen_flash_bounds_rejects_invalid_sizes():
    assert normalize_screen_flash_bounds({"left": 0, "top": 0, "width": 0, "height": 1600}) is None
