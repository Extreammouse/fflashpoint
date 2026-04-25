"""Create a printable A4 chessboard camera-calibration target."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    args = parse_args()
    dpi = args.dpi
    width_px = round(args.paper_width_mm / 25.4 * dpi)
    height_px = round(args.paper_height_mm / 25.4 * dpi)
    square_px = round(args.square_mm / 25.4 * dpi)
    square_cols = args.inner_cols + 1
    square_rows = args.inner_rows + 1
    board_width = square_cols * square_px
    board_height = square_rows * square_px

    if board_width >= width_px or board_height >= height_px:
        raise SystemExit("Board does not fit on the selected paper size.")

    image = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    x0 = (width_px - board_width) // 2
    y0 = (height_px - board_height) // 2

    for row in range(square_rows):
        for col in range(square_cols):
            if (row + col) % 2 == 0:
                cv2.rectangle(
                    image,
                    (x0 + col * square_px, y0 + row * square_px),
                    (x0 + (col + 1) * square_px, y0 + (row + 1) * square_px),
                    (0, 0, 0),
                    thickness=-1,
                )

    cv2.rectangle(image, (x0, y0), (x0 + board_width, y0 + board_height), (0, 0, 0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = (
        f"{args.inner_cols}x{args.inner_rows} inner corners | "
        f"{args.square_mm:g} mm squares | Print A4 landscape at 100%"
    )
    cv2.putText(image, label, (110, 70), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(
        image,
        f"After printing: measure one square. It should be {args.square_mm:g} mm.",
        (110, height_px - 70),
        font,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise SystemExit(f"Could not write {output}")

    print(output.resolve())
    print(
        f"image={width_px}x{height_px}px square={square_px}px "
        f"board={board_width}x{board_height}px"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate printable chessboard PNG.")
    parser.add_argument("--inner-cols", type=int, default=9)
    parser.add_argument("--inner-rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--paper-width-mm", type=float, default=297.0)
    parser.add_argument("--paper-height-mm", type=float, default=210.0)
    parser.add_argument(
        "--output",
        default=(
            "calibration_targets/"
            "chessboard_9x6_inner_25mm_A4_landscape_300dpi.png"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
