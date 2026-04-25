"""Download or normalize local model assets required by the prototype.

This is the single entry point for model acquisition. Runtime code should read
models from paths in config.py and should not download weights on its own.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import config


MODEL_ROOT = config.BASE_DIR / "models"


def main() -> None:
    args = parse_args()
    requested = set(args.only)
    if "all" in requested:
        requested = {"l2cs", "body-regions"}

    if "l2cs" in requested:
        ensure_l2cs_weights(force=args.force)
    if "body-regions" in requested:
        ensure_body_region_model()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or normalize local model assets for the prototype."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=("all", "l2cs", "body-regions"),
        default=["all"],
        help="Model group to prepare. Default: all required runtime models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload direct assets and ignore existing root L2CS target.",
    )
    return parser.parse_args()


def ensure_body_region_model() -> None:
    destination = Path(config.BODY_REGION_MODEL_PATH)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Body-region YOLO segmentation model exists: {destination}")
        return
    raise SystemExit(
        "Missing trained body-region segmentation model:\n"
        f"{destination}\n"
        "Copy the Turing export torso_lowerbody_yolov8s_seg_best.pt to that path as "
        "body_regions_yolov8s_seg.pt."
    )


def ensure_l2cs_weights(force: bool = False) -> None:
    destination = Path(config.L2CS_WEIGHTS_PATH)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"L2CS Gaze360 weights already exist: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    local_candidate = find_local_l2cs_candidate(exclude=destination)
    if local_candidate is not None and not force:
        shutil.copy2(local_candidate, destination)
        print(f"Copied L2CS Gaze360 weights from {local_candidate} to {destination}")
        return

    download_l2cs_with_gdown()
    local_candidate = find_local_l2cs_candidate(exclude=destination)
    if local_candidate is None:
        raise SystemExit(
            "L2CS download completed, but L2CSNet_gaze360.pkl was not found. "
            f"Expected runtime path: {destination}"
        )
    if local_candidate.resolve() != destination.resolve():
        shutil.copy2(local_candidate, destination)
    if not destination.exists() or destination.stat().st_size <= 0:
        raise SystemExit(f"L2CS weights are missing after download: {destination}")
    print(f"Ready L2CS Gaze360 weights: {destination}")


def find_local_l2cs_candidate(exclude: Path | None = None) -> Path | None:
    exclude_resolved = exclude.resolve() if exclude is not None and exclude.exists() else None
    candidates = sorted(MODEL_ROOT.rglob("L2CSNet_gaze360.pkl"))
    for candidate in candidates:
        if exclude_resolved is not None and candidate.resolve() == exclude_resolved:
            continue
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def download_l2cs_with_gdown() -> None:
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: gdown. Install it with:\n"
            ".\\.venv\\Scripts\\python.exe -m pip install gdown\n"
            "Then rerun download_models.py."
        ) from exc

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    print("Downloading L2CS pretrained model folder with gdown...")
    print(f"URL: {config.L2CS_GDRIVE_FOLDER_URL}")
    print(f"Destination folder: {MODEL_ROOT}")
    try:
        gdown.download_folder(
            url=config.L2CS_GDRIVE_FOLDER_URL,
            output=str(MODEL_ROOT),
            quiet=False,
        )
    except TypeError:
        gdown.download_folder(
            config.L2CS_GDRIVE_FOLDER_URL,
            output=str(MODEL_ROOT),
            quiet=False,
        )


if __name__ == "__main__":
    main()
