"""Optional full-screen visual flash overlay — Mac and Windows compatible."""

from __future__ import annotations

from collections.abc import Mapping
import multiprocessing
import platform
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScreenFlashSettings:
    enabled: bool
    mode: str
    duration_ms: int
    alpha: float
    color: str
    bounds: tuple[int, int, int, int] | None


VALID_SCREEN_FLASH_MODES = {"overlay", "white"}


def resolve_screen_flash_style(
    mode: str,
    *,
    overlay_alpha: float,
    overlay_color: str,
    white_color: str = "#ffffff",
) -> tuple[str, float, str]:
    """Resolve a user-facing flash mode to Tk alpha and color values."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in VALID_SCREEN_FLASH_MODES:
        normalized_mode = "overlay"

    if normalized_mode == "white":
        return "white", 1.0, str(white_color)
    return "overlay", _clamp_alpha(overlay_alpha), str(overlay_color)


def _clamp_alpha(alpha: float) -> float:
    return min(max(0.05, float(alpha)), 1.0)


def normalize_screen_flash_bounds(
    bounds: Mapping[str, Any] | None,
) -> tuple[int, int, int, int] | None:
    if not isinstance(bounds, Mapping):
        return None
    try:
        left = int(bounds.get("left", 0))
        top = int(bounds.get("top", 0))
        width = int(bounds.get("width", 0))
        height = int(bounds.get("height", 0))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return left, top, width, height


def _run_flash_process(
    command_queue: multiprocessing.Queue,
    duration_ms: int,
    color: str,
    alpha: float,
    bounds: tuple[int, int, int, int] | None,
) -> None:
    """Run Tkinter flash window in a separate process (main thread of that process)."""
    try:
        import tkinter as tk
    except Exception:
        return

    root = tk.Tk()
    root.withdraw()
    root.overrideredirect(True)
    root.configure(bg=color)
    root.attributes("-topmost", True)
    root.attributes("-alpha", alpha)

    if bounds is None:
        left, top = 0, 0
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
    else:
        left, top, width, height = bounds
    root.geometry(f"{width}x{height}{left:+d}{top:+d}")

    active_until = 0.0
    close_requested = False

    def poll() -> None:
        nonlocal active_until, close_requested
        try:
            while True:
                command = command_queue.get_nowait()
                if command == "close":
                    close_requested = True
                    root.withdraw()
                    root.quit()
                    return
                if command == "flash":
                    active_until = max(
                        active_until,
                        time.monotonic() + duration_ms / 1000.0,
                    )
                    root.deiconify()
                    root.lift()
        except Exception:
            pass

        if active_until and time.monotonic() >= active_until:
            root.withdraw()
            active_until = 0.0

        if not close_requested:
            root.after(20, poll)

    root.after(0, poll)
    root.mainloop()


class ScreenFlashOverlay:
    """Topmost flash window, Mac and Windows compatible."""

    def __init__(
        self,
        enabled: bool,
        duration_ms: int,
        mode: str,
        alpha: float,
        color: str,
        bounds: Mapping[str, Any] | None = None,
    ) -> None:
        self.settings = ScreenFlashSettings(
            enabled=bool(enabled),
            mode=str(mode),
            duration_ms=max(1, int(duration_ms)),
            alpha=_clamp_alpha(alpha),
            color=str(color),
            bounds=normalize_screen_flash_bounds(bounds),
        )
        self.status = "disabled"
        self._is_mac = platform.system() == "Darwin"
        self._command_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._process: multiprocessing.Process | None = None
        self._thread: threading.Thread | None = None
        if self.settings.enabled:
            self._ensure_running()

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def trigger(self, label: str | None = None) -> None:
        if not self.enabled:
            return
        self._ensure_running()
        self._command_queue.put("flash")

    def close(self) -> None:
        if not self.enabled:
            return
        self._command_queue.put("close")
        if self._is_mac and self._process is not None:
            self._process.join(timeout=3.0)
            self._process = None
        elif self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.status = "closed"

    def as_metadata(self) -> dict[str, Any]:
        metadata = {
            "enabled": self.enabled,
            "status": self.status,
            "mode": self.settings.mode if self.enabled else "disabled",
            "duration_ms": self.settings.duration_ms if self.enabled else 0,
            "alpha": self.settings.alpha if self.enabled else 0.0,
            "color": self.settings.color if self.enabled else "",
        }
        if self.enabled:
            metadata["bounds"] = (
                {
                    "left": self.settings.bounds[0],
                    "top": self.settings.bounds[1],
                    "width": self.settings.bounds[2],
                    "height": self.settings.bounds[3],
                }
                if self.settings.bounds is not None
                else None
            )
        return metadata

    def _ensure_running(self) -> None:
        if not self.enabled:
            return
        if self._is_mac:
            if self._process is not None and self._process.is_alive():
                return
            self.status = "starting"
            self._process = multiprocessing.Process(
                target=_run_flash_process,
                args=(
                    self._command_queue,
                    self.settings.duration_ms,
                    self.settings.color,
                    self.settings.alpha,
                    self.settings.bounds,
                ),
                daemon=True,
            )
            self._process.start()
            self.status = "ready"
        else:
            if self._thread is not None and self._thread.is_alive():
                return
            self.status = "starting"
            self._thread = threading.Thread(
                target=self._run_tk_loop,
                name="screen-flash-overlay",
                daemon=True,
            )
            self._thread.start()

    def _run_tk_loop(self) -> None:
        try:
            import tkinter as tk
        except Exception as exc:
            self.status = f"unavailable: {exc}"
            return

        root: tk.Tk | None = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.overrideredirect(True)
            root.configure(bg=self.settings.color)
            root.attributes("-topmost", True)
            root.attributes("-alpha", self.settings.alpha)

            bounds = self.settings.bounds
            if bounds is None:
                left, top = 0, 0
                width = root.winfo_screenwidth()
                height = root.winfo_screenheight()
            else:
                left, top, width, height = bounds
            root.geometry(f"{width}x{height}{left:+d}{top:+d}")

            active_until = 0.0
            close_requested = False
            self.status = "ready"

            def poll() -> None:
                nonlocal active_until, close_requested
                try:
                    while True:
                        command = self._command_queue.get_nowait()
                        if command == "close":
                            close_requested = True
                            self.status = "closed"
                            root.withdraw()
                            root.quit()
                            return
                        if command == "flash":
                            active_until = max(
                                active_until,
                                time.monotonic() + self.settings.duration_ms / 1000.0,
                            )
                            self.status = "active"
                            root.deiconify()
                            root.lift()
                except Exception:
                    pass

                if active_until and time.monotonic() >= active_until:
                    root.withdraw()
                    active_until = 0.0
                    self.status = "ready"

                if not close_requested:
                    root.after(20, poll)

            root.after(0, poll)
            root.mainloop()
        except Exception as exc:
            self.status = f"error: {exc}"
        finally:
            if root is not None:
                try:
                    root.withdraw()
                except Exception:
                    pass
                try:
                    root.destroy()
                except Exception:
                    pass
