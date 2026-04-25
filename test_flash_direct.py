"""Quick standalone test: fires the screen flash directly without needing gaze or YOLO."""

import multiprocessing
import time

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    from src.screen_flash import ScreenFlashOverlay

    print("Starting flash overlay process...")
    flash = ScreenFlashOverlay(
        enabled=True,
        duration_ms=1500,
        mode="white",
        alpha=1.0,
        color="#ffffff",
    )

    time.sleep(1.5)  # let subprocess start
    print("Firing flash now! You should see a white screen for 1.5 seconds.")
    flash.trigger()
    time.sleep(2.5)

    print("Firing again...")
    flash.trigger()
    time.sleep(2.5)

    flash.close()
    print("Done — if you saw two white flashes, the Mac fix is working.")
